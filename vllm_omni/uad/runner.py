from __future__ import annotations

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.omni.hunyuan_image3 import HunyuanImage3UADStateConfig
from vllm_omni.uad.outputs import UADModelOutput, UADStepOutput
from vllm_omni.uad.request import UADPhaseUpdate, UADRequestState, UADToken
from vllm_omni.uad.scheduler import UADScheduleItem, UADSchedulerOutput


class UADRunner:
    """Runner-first UAD execution facade.

    Step 3 keeps execution toy-sized, but centralizes item execution in the
    runner instead of a separate model adapter. Later steps replace these toy
    item paths with batched input building, paged metadata, and real AR/DiT
    module calls.
    """

    def __init__(
        self,
        model: HunyuanImage3UADForConditionalGeneration | None = None,
        state_config: HunyuanImage3UADStateConfig | None = None,
    ) -> None:
        self.model = model or HunyuanImage3UADForConditionalGeneration()
        self.state_config = state_config or HunyuanImage3UADStateConfig()

    def execute_model(
        self,
        scheduler_output: UADSchedulerOutput,
        requests: dict[str, UADRequestState],
    ) -> UADStepOutput:
        outputs: list[UADModelOutput] = []
        for item in scheduler_output.scheduled_items:
            request = requests[item.request_id]
            if item.phase in ("ar_prefill", "ar_decode"):
                outputs.append(self._execute_ar_item(item, request))
            elif item.phase == "dit_step":
                outputs.append(self._execute_dit_item(item, request))
            else:
                raise NotImplementedError(f"unsupported UAD phase: {item.phase}")
        return UADStepOutput(outputs=outputs)

    def _execute_ar_item(
        self,
        item: UADScheduleItem,
        request: UADRequestState,
    ) -> UADModelOutput:
        input_ids = torch.tensor(item.token_ids, dtype=torch.long)
        model_output = self.model(input_ids=input_ids, request_id=request.request_id, phase=item.phase)
        sampled_token = UADToken(modality="text", token_id=model_output.next_token_id)

        ratio_index = self.state_config.ratio_index(sampled_token.token_id)
        if ratio_index is not None:
            image_context_tokens = self.state_config.build_toy_image_context_tokens()
            return UADModelOutput(
                request_id=request.request_id,
                new_engine_tokens=[sampled_token] + image_context_tokens,
                new_materialized_tokens=[],
                num_computed_tokens_delta=item.num_scheduled_tokens,
                phase_update=UADPhaseUpdate(
                    phase="dit_step",
                    dit_step_index=0,
                    total_dit_steps=self.state_config.toy_total_dit_steps,
                    image_ratio_token_id=sampled_token.token_id,
                    image_ratio_index=ratio_index,
                    image_context_token_count=len(image_context_tokens),
                    pending_image_context_commit=True,
                ),
                finished=False,
            )

        materialized_tokens = []
        if not self.state_config.is_engine_only_token(sampled_token.token_id):
            materialized_tokens.append(sampled_token)
        return UADModelOutput(
            request_id=request.request_id,
            new_engine_tokens=[sampled_token],
            new_materialized_tokens=materialized_tokens,
            num_computed_tokens_delta=item.num_scheduled_tokens,
            finished=False,
        )

    def _execute_dit_item(
        self,
        item: UADScheduleItem,
        request: UADRequestState,
    ) -> UADModelOutput:
        if request.total_dit_steps <= 0:
            raise ValueError(f"request {request.request_id} entered dit_step without total_dit_steps")

        next_step_index = min(request.dit_step_index + 1, request.total_dit_steps)
        return UADModelOutput(
            request_id=request.request_id,
            num_computed_tokens_delta=0,
            phase_update=UADPhaseUpdate(
                phase="dit_step",
                dit_step_index=next_step_index,
                pending_image_context_commit=True,
            ),
            finished=False,
        )
