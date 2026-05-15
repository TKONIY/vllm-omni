from __future__ import annotations

from vllm_omni.uad.outputs import UADEngineCoreOutputs
from vllm_omni.uad.request import UADRequestState
from vllm_omni.uad.runner import UADRunner
from vllm_omni.uad.scheduler import UADToyScheduler
from vllm_omni.uad.state.base import UADModelStateMachine


class UADEngine:
    """Minimal Unified AR + DiT engine shell for Step 0."""

    def __init__(
        self,
        scheduler: UADToyScheduler | None = None,
        runner: UADRunner | None = None,
        state_machine: UADModelStateMachine | None = None,
    ) -> None:
        if scheduler is not None and state_machine is not None:
            raise ValueError("pass either scheduler or state_machine, not both")
        self.scheduler = scheduler or UADToyScheduler(state_machine=state_machine)
        self.runner = runner or UADRunner()

    @property
    def requests(self) -> dict[str, UADRequestState]:
        return self.scheduler.requests

    def add_request(self, request_id: str, prompt_token_ids: list[int]) -> UADRequestState:
        return self.scheduler.add_request(request_id, prompt_token_ids)

    def step(self) -> UADEngineCoreOutputs:
        scheduler_output = self.scheduler.schedule()
        runner_output = self.runner.execute_model(scheduler_output)
        return self.scheduler.update_from_output(scheduler_output, runner_output)

    def get_request(self, request_id: str) -> UADRequestState:
        return self.scheduler.get_request(request_id)


class AsyncUADEngine:
    """Async wrapper matching the shape of the production async engine."""

    def __init__(self, engine: UADEngine | None = None) -> None:
        self.engine = engine or UADEngine()

    async def add_request(self, request_id: str, prompt_token_ids: list[int]) -> UADRequestState:
        return self.engine.add_request(request_id, prompt_token_ids)

    async def step(self) -> UADEngineCoreOutputs:
        return self.engine.step()
