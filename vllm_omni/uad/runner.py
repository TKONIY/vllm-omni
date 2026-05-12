from __future__ import annotations

from vllm_omni.uad.omni.adapter.hunyuan_image3 import HunyuanImage3UADAdapter
from vllm_omni.uad.outputs import UADModelOutput, UADStepOutput
from vllm_omni.uad.request import UADRequestState
from vllm_omni.uad.scheduler import UADSchedulerOutput


class UADRunner:
    """Step 0 runner facade.

    It keeps the engine independent from HunyuanImage3-specific request
    conversion while still exercising the UAD model entrypoint.
    """

    def __init__(self, adapter: HunyuanImage3UADAdapter | None = None) -> None:
        self.adapter = adapter or HunyuanImage3UADAdapter()

    def execute_model(
        self,
        scheduler_output: UADSchedulerOutput,
        requests: dict[str, UADRequestState],
    ) -> UADStepOutput:
        outputs: list[UADModelOutput] = []
        for item in scheduler_output.scheduled_items:
            request = requests[item.request_id]
            outputs.append(self.adapter.execute_item(item, request))
        return UADStepOutput(outputs=outputs)
