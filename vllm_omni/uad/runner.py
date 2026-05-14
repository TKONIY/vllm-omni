from __future__ import annotations

import torch

from vllm_omni.uad.batch import UADBatchInputs, UADBatchItem, UADBatchOutputs
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADModel
from vllm_omni.uad.outputs import UADRunnerOutput, UADRunnerStepOutput
from vllm_omni.uad.request import UADToken
from vllm_omni.uad.scheduler import UADScheduleItem, UADSchedulerOutput


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
    ) -> None:
        self.model = model or HunyuanImage3UADModel()
        self.last_batch_inputs: UADBatchInputs | None = None
        self.last_batch_outputs: UADBatchOutputs | None = None

    def execute_model(
        self,
        scheduler_output: UADSchedulerOutput,
    ) -> UADRunnerStepOutput:
        if not scheduler_output.scheduled_items:
            self.last_batch_inputs = None
            self.last_batch_outputs = None
            return UADRunnerStepOutput()

        batch_inputs = self._build_batch_inputs(scheduler_output)
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

            input_kind = "token_ids" if item.phase in ("ar_prefill", "ar_decode") else "latent_timestep"
            uses_prefix_attention = True
            uses_chunk_bidirectional_attention = item.phase == "dit_step"
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
                    input_kind=input_kind,
                    uses_prefix_attention=uses_prefix_attention,
                    uses_chunk_bidirectional_attention=uses_chunk_bidirectional_attention,
                    dit_step_index=item.dit_step_index,
                    total_dit_steps=item.total_dit_steps,
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
    ) -> UADRunnerStepOutput:
        outputs_by_index: list[UADRunnerOutput | None] = [None] * len(scheduler_output.scheduled_items)
        for item_output in batch_outputs.item_outputs:
            item = scheduler_output.scheduled_items[item_output.output_index]
            if item.request_id != item_output.request_id or item.phase != item_output.phase:
                raise ValueError(
                    "model batch output item mismatch: "
                    f"scheduled ({item.request_id}, {item.phase}) "
                    f"but got ({item_output.request_id}, {item_output.phase})"
                )

            sampled_token = None
            if item_output.next_token_id is not None:
                sampled_token = UADToken(modality="text", token_id=item_output.next_token_id)

            outputs_by_index[item_output.output_index] = UADRunnerOutput(
                request_id=item_output.request_id,
                phase=item_output.phase,
                num_scheduled_tokens=item_output.num_scheduled_tokens,
                sampled_token=sampled_token,
            )

        if any(output is None for output in outputs_by_index):
            raise ValueError("model batch output did not cover every scheduled item")
        return UADRunnerStepOutput(outputs=[output for output in outputs_by_index if output is not None])
