from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.outputs import UADModelRunnerOutput
from uad_vllm.scheduler import UADSchedulerOutput


class UADRunner:
    """Output processor boundary for UAD model execution."""

    def process_outputs(
        self,
        scheduler_output: UADSchedulerOutput,
        raw_model_output: Any,
    ) -> UADModelRunnerOutput:
        del scheduler_output
        # TODO: split raw AR sampler output, DiT denoise predictions, and
        # artifact decode results into UADPhaseOutput entries.
        if isinstance(raw_model_output, UADModelRunnerOutput):
            return raw_model_output
        if isinstance(raw_model_output, ModelRunnerOutput):
            return UADModelRunnerOutput.from_base(raw_model_output, phase_outputs=[])
        return UADModelRunnerOutput.make_empty(phase_outputs=[])


class UADGPUModelRunner:
    """Placeholder interface for the future UAD-native GPU model runner."""

    def __init__(self, vllm_config: Any, device: Any) -> None:
        self.vllm_config = vllm_config
        self.device = device

    def execute_model(self, scheduler_output: SchedulerOutput, intermediate_tensors: Any | None = None) -> Any:
        del scheduler_output, intermediate_tensors
        # TODO: build UAD mixed inputs and run AR/DiT layers.
        return None
