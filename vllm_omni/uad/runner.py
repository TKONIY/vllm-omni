from __future__ import annotations

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.omni.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine
from vllm_omni.uad.outputs import UADModelOutput, UADStepOutput
from vllm_omni.uad.request import UADRequestState, UADToken
from vllm_omni.uad.scheduler import UADScheduleItem, UADSchedulerOutput
from vllm_omni.uad.state_machine import UADModelStateMachine


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
        state_machine: UADModelStateMachine | None = None,
        state_config: HunyuanImage3UADStateConfig | None = None,
    ) -> None:
        self.model = model or HunyuanImage3UADForConditionalGeneration()
        if state_machine is not None and state_config is not None:
            raise ValueError("pass either state_machine or state_config, not both")
        self.state_machine = state_machine or HunyuanImage3UADStateMachine(
            config=state_config or HunyuanImage3UADStateConfig()
        )

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
        return self.state_machine.on_ar_token_sampled(
            request=request,
            sampled_token=sampled_token,
            num_scheduled_tokens=item.num_scheduled_tokens,
        )

    def _execute_dit_item(
        self,
        item: UADScheduleItem,
        request: UADRequestState,
    ) -> UADModelOutput:
        return self.state_machine.on_dit_step_completed(
            request=request,
            num_scheduled_tokens=item.num_scheduled_tokens,
        )
