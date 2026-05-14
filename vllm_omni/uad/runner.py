from __future__ import annotations

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.outputs import UADRunnerOutput, UADRunnerStepOutput
from vllm_omni.uad.request import UADToken
from vllm_omni.uad.scheduler import UADScheduleItem, UADSchedulerOutput


class UADRunner:
    """Runner-first UAD execution facade.

    Step 3 keeps execution toy-sized, but centralizes item execution in the
    runner instead of a separate model adapter. The runner consumes the whole
    scheduler output for a tick, groups work by executable phase, and returns
    raw model results. Request state transitions remain outside the runner.
    """

    def __init__(
        self,
        model: HunyuanImage3UADForConditionalGeneration | None = None,
    ) -> None:
        self.model = model or HunyuanImage3UADForConditionalGeneration()

    def execute_model(
        self,
        scheduler_output: UADSchedulerOutput,
    ) -> UADRunnerStepOutput:
        outputs_by_index: list[UADRunnerOutput | None] = [None] * len(scheduler_output.scheduled_items)
        ar_items: list[tuple[int, UADScheduleItem]] = []
        dit_items: list[tuple[int, UADScheduleItem]] = []

        for index, item in enumerate(scheduler_output.scheduled_items):
            if item.phase in ("ar_prefill", "ar_decode"):
                ar_items.append((index, item))
            elif item.phase == "dit_step":
                dit_items.append((index, item))
            else:
                raise NotImplementedError(f"unsupported UAD phase: {item.phase}")

        for index, output in self._execute_ar_items(ar_items):
            outputs_by_index[index] = output
        for index, output in self._execute_dit_items(dit_items):
            outputs_by_index[index] = output

        return UADRunnerStepOutput(outputs=[output for output in outputs_by_index if output is not None])

    def _execute_ar_items(
        self,
        indexed_items: list[tuple[int, UADScheduleItem]],
    ) -> list[tuple[int, UADRunnerOutput]]:
        outputs: list[tuple[int, UADRunnerOutput]] = []
        for index, item in indexed_items:
            input_ids = torch.tensor(item.token_ids, dtype=torch.long)
            model_output = self.model(input_ids=input_ids, request_id=item.request_id, phase=item.phase)
            sampled_token = UADToken(modality="text", token_id=model_output.next_token_id)
            outputs.append(
                (
                    index,
                    UADRunnerOutput(
                        request_id=item.request_id,
                        phase=item.phase,
                        num_scheduled_tokens=item.num_scheduled_tokens,
                        sampled_token=sampled_token,
                    ),
                )
            )
        return outputs

    def _execute_dit_items(
        self,
        indexed_items: list[tuple[int, UADScheduleItem]],
    ) -> list[tuple[int, UADRunnerOutput]]:
        return [
            (
                index,
                UADRunnerOutput(
                    request_id=item.request_id,
                    phase=item.phase,
                    num_scheduled_tokens=item.num_scheduled_tokens,
                ),
            )
            for index, item in indexed_items
        ]
