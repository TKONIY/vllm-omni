from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_omni.uad.batch import UADBatchInputs, UADBatchItem, UADBatchItemOutput, UADBatchOutputs
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADModel
from vllm_omni.uad.outputs import UADModelRunnerItemOutput, UADModelRunnerOutput
from vllm_omni.uad.request import UADToken
from vllm_omni.uad.scheduler import UADScheduleItem, UADSchedulerOutput

SamplingMetadataBuilder = Callable[[torch.Tensor, UADBatchItem], Any]


class UADRunner:
    """Batch-first UAD execution facade.

    The runner consumes the whole scheduler output for a tick, packs mixed
    AR/DiT items into `UADBatchInputs`, calls the concrete model shell once,
    and scatters raw model results back to scheduled item order. Request state
    transitions remain outside the runner.
    """

    def __init__(
        self,
        model: HunyuanImage3UADModel | None = None,
        sampling_metadata_builder: SamplingMetadataBuilder | None = None,
    ) -> None:
        self.model = model or HunyuanImage3UADModel()
        self.sampling_metadata_builder = sampling_metadata_builder
        self.last_batch_inputs: UADBatchInputs | None = None
        self.last_batch_outputs: UADBatchOutputs | None = None

    def execute_model(
        self,
        scheduler_output: UADSchedulerOutput,
    ) -> UADModelRunnerOutput:
        if not scheduler_output.scheduled_items:
            self.last_batch_inputs = None
            self.last_batch_outputs = None
            return UADModelRunnerOutput()

        batch_inputs = self._build_batch_inputs(scheduler_output)
        self._validate_backend_batch(batch_inputs)
        batch_outputs = self.model(batch_inputs)
        self.last_batch_inputs = batch_inputs
        self.last_batch_outputs = batch_outputs
        return self._scatter_batch_outputs(scheduler_output, batch_outputs)

    def _build_batch_inputs(
        self,
        scheduler_output: UADSchedulerOutput,
    ) -> UADBatchInputs:
        batch_items: list[UADBatchItem] = []
        token_ids: list[int] = []
        token_item_indices: list[int] = []
        token_positions: list[int] = []

        for index, item in enumerate(scheduler_output.scheduled_items):
            token_start = len(token_ids)
            packed_token_ids = self._pack_item_token_ids(item)
            token_ids.extend(packed_token_ids)
            token_item_indices.extend([index] * item.num_scheduled_tokens)
            token_positions.extend(
                range(
                    item.num_computed_tokens,
                    item.num_computed_tokens + item.num_scheduled_tokens,
                )
            )
            token_end = len(token_ids)

            batch_items.append(
                UADBatchItem(
                    request_id=item.request_id,
                    phase=item.phase,
                    output_index=index,
                    num_tokens=item.num_scheduled_tokens,
                    token_start=token_start,
                    token_end=token_end,
                    num_computed_tokens=item.num_computed_tokens,
                    persist=item.persist,
                    dit_step_index=item.dit_step_index,
                    total_dit_steps=item.total_dit_steps,
                    ar_sampler_token_ids=tuple(item.ar_sampler_token_ids),
                    sample_token_offset=item.sample_token_offset,
                )
            )

        return UADBatchInputs(
            items=tuple(batch_items),
            input_token_ids=torch.tensor(token_ids, dtype=torch.long),
            token_item_indices=torch.tensor(token_item_indices, dtype=torch.long),
            token_positions=torch.tensor(token_positions, dtype=torch.long),
        )

    def _pack_item_token_ids(
        self,
        item: UADScheduleItem,
    ) -> list[int]:
        if item.phase in ("ar_prefill", "ar_decode"):
            if len(item.token_ids) != item.num_scheduled_tokens:
                raise ValueError(
                    f"AR item {item.request_id} has {len(item.token_ids)} token_ids "
                    f"but schedules {item.num_scheduled_tokens} tokens"
                )
            return item.token_ids
        if item.phase == "dit_step":
            return [-1] * item.num_scheduled_tokens
        raise NotImplementedError(f"unsupported UAD phase: {item.phase}")

    def _scatter_batch_outputs(
        self,
        scheduler_output: UADSchedulerOutput,
        batch_outputs: UADBatchOutputs,
    ) -> UADModelRunnerOutput:
        outputs_by_index: list[UADModelRunnerItemOutput | None] = [None] * len(
            scheduler_output.scheduled_items
        )
        for item_output in batch_outputs.item_outputs:
            item = scheduler_output.scheduled_items[item_output.output_index]
            if item.request_id != item_output.request_id or item.phase != item_output.phase:
                raise ValueError(
                    "model batch output item mismatch: "
                    f"scheduled ({item.request_id}, {item.phase}) "
                    f"but got ({item_output.request_id}, {item_output.phase})"
                )

            next_token_id = self._resolve_next_token_id(item, item_output)
            sampled_token = None
            if next_token_id is not None:
                sampled_token = UADToken(modality="text", token_id=next_token_id)

            outputs_by_index[item_output.output_index] = UADModelRunnerItemOutput(
                request_id=item_output.request_id,
                phase=item_output.phase,
                num_scheduled_tokens=item_output.num_scheduled_tokens,
                sampled_token=sampled_token,
            )

        if any(output is None for output in outputs_by_index):
            raise ValueError("model batch output did not cover every scheduled item")
        return UADModelRunnerOutput(outputs=[output for output in outputs_by_index if output is not None])

    def _validate_backend_batch(self, batch_inputs: UADBatchInputs) -> None:
        if self.model.ar_model is None:
            return
        num_ar_items = len(batch_inputs.ar_item_indices)
        if num_ar_items > 1:
            raise NotImplementedError(
                "Milestone B real AR backend path is single-request only; "
                f"got {num_ar_items} AR items in one scheduler tick"
            )

    def _resolve_next_token_id(
        self,
        item: UADScheduleItem,
        item_output: UADBatchItemOutput,
    ) -> int | None:
        if item_output.next_token_id is not None:
            return item_output.next_token_id
        if item.phase not in ("ar_prefill", "ar_decode") or self.model.ar_model is None:
            return None
        return self._sample_backend_ar_item(item, item_output)

    def _sample_backend_ar_item(
        self,
        item: UADScheduleItem,
        item_output: UADBatchItemOutput,
    ) -> int:
        hidden_states = item_output.hidden_states
        if hidden_states is None:
            raise ValueError(f"AR backend item {item.request_id} did not return hidden_states")
        if item.sample_token_offset is None:
            raise ValueError(f"AR backend item {item.request_id} is missing sample_token_offset")
        if item.sample_token_offset < 0 or item.sample_token_offset >= hidden_states.shape[0]:
            raise ValueError(
                f"sample_token_offset {item.sample_token_offset} is out of range "
                f"for hidden_states length {hidden_states.shape[0]}"
            )

        sample_hidden_states = hidden_states[item.sample_token_offset : item.sample_token_offset + 1].contiguous()
        logits = self.model.ar_model.compute_logits(sample_hidden_states)
        if logits is None:
            raise ValueError("AR backend compute_logits returned None")

        model_sample = getattr(self.model.ar_model, "sample", None)
        if callable(model_sample):
            sampling_metadata = self._build_sampling_metadata(logits, item)
            sampler_output = model_sample(logits, sampling_metadata)
            sampled_token_id = self._extract_sampled_token_id(sampler_output)
            if sampled_token_id is not None:
                return sampled_token_id

        return int(torch.argmax(logits[0], dim=-1).item())

    def _build_sampling_metadata(
        self,
        logits: torch.Tensor,
        item: UADScheduleItem,
    ) -> Any:
        if self.sampling_metadata_builder is not None:
            batch_item = UADBatchItem(
                request_id=item.request_id,
                phase=item.phase,
                output_index=0,
                num_tokens=item.num_scheduled_tokens,
                token_start=0,
                token_end=item.num_scheduled_tokens,
                num_computed_tokens=item.num_computed_tokens,
                persist=item.persist,
                ar_sampler_token_ids=tuple(item.ar_sampler_token_ids),
                sample_token_offset=item.sample_token_offset,
            )
            return self.sampling_metadata_builder(logits, batch_item)

        from vllm.v1.sample.logits_processor.state import LogitsProcessors
        from vllm.v1.sample.metadata import SamplingMetadata

        device = logits.device
        batch_size = logits.shape[0]
        return SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(batch_size, device=device),
            presence_penalties=torch.zeros(batch_size, device=device),
            repetition_penalties=torch.ones(batch_size, device=device),
            output_token_ids=[list(item.ar_sampler_token_ids)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    @staticmethod
    def _extract_sampled_token_id(sampler_output: Any) -> int | None:
        if sampler_output is None:
            return None
        sampled_token_ids = getattr(sampler_output, "sampled_token_ids", sampler_output)
        if isinstance(sampled_token_ids, torch.Tensor):
            if sampled_token_ids.numel() == 0:
                return None
            return int(sampled_token_ids.reshape(-1)[0].item())
        if isinstance(sampled_token_ids, list):
            if not sampled_token_ids:
                return None
            first = sampled_token_ids[0]
            if isinstance(first, list):
                if not first:
                    return None
                return int(first[0])
            return int(first)
        return int(sampled_token_ids)
