from __future__ import annotations

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.outputs import UADModelOutput
from vllm_omni.uad.request import UADRequestState, UADToken
from vllm_omni.uad.scheduler import UADScheduleItem


class HunyuanImage3UADAdapter:
    """Translate Step 0 UAD schedule items into HunyuanImage3 toy AR calls."""

    def __init__(self, model: HunyuanImage3UADForConditionalGeneration | None = None) -> None:
        self.model = model or HunyuanImage3UADForConditionalGeneration()

    def execute_item(
        self,
        item: UADScheduleItem,
        request: UADRequestState,
    ) -> UADModelOutput:
        if item.phase not in ("ar_prefill", "ar_decode"):
            raise NotImplementedError(f"Step 0 only supports toy AR phases, got {item.phase}")

        input_ids = torch.tensor(item.token_ids, dtype=torch.long)
        model_output = self.model(input_ids=input_ids, request_id=request.request_id, phase=item.phase)
        sampled_token = UADToken(modality="text", token_id=model_output.next_token_id)
        return UADModelOutput(
            request_id=request.request_id,
            new_engine_tokens=[sampled_token],
            new_materialized_tokens=[sampled_token],
            num_computed_tokens_delta=item.num_scheduled_tokens,
            finished=False,
        )
