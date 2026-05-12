from __future__ import annotations

from vllm_omni.uad.outputs import UADModelOutput, UADStepOutput
from vllm_omni.uad.request import UADRequestState
from vllm_omni.uad.runner import UADRunner
from vllm_omni.uad.scheduler import UADToyScheduler


class UADEngine:
    """Minimal Unified AR + DiT engine shell for Step 0."""

    def __init__(
        self,
        scheduler: UADToyScheduler | None = None,
        runner: UADRunner | None = None,
    ) -> None:
        self.scheduler = scheduler or UADToyScheduler()
        self.runner = runner or UADRunner()
        self.requests: dict[str, UADRequestState] = {}

    def add_request(self, request_id: str, prompt_token_ids: list[int]) -> UADRequestState:
        if request_id in self.requests:
            raise ValueError(f"duplicate UAD request_id: {request_id}")
        request = UADRequestState.from_prompt_token_ids(request_id, prompt_token_ids)
        self.requests[request_id] = request
        return request

    def step(self) -> UADStepOutput:
        scheduler_output = self.scheduler.schedule(list(self.requests.values()))
        step_output = self.runner.execute_model(scheduler_output, self.requests)
        for output in step_output.outputs:
            self._apply_model_output(output)
        return step_output

    def get_request(self, request_id: str) -> UADRequestState:
        return self.requests[request_id]

    def _apply_model_output(self, output: UADModelOutput) -> None:
        request = self.requests[output.request_id]
        request.advance_computed_tokens(output.num_computed_tokens_delta)
        request.append_engine_tokens(output.new_engine_tokens)
        request.append_materialized_tokens(output.new_materialized_tokens)
        request.finished = output.finished


class AsyncUADEngine:
    """Async wrapper matching the shape of the production async engine."""

    def __init__(self, engine: UADEngine | None = None) -> None:
        self.engine = engine or UADEngine()

    async def add_request(self, request_id: str, prompt_token_ids: list[int]) -> UADRequestState:
        return self.engine.add_request(request_id, prompt_token_ids)

    async def step(self) -> UADStepOutput:
        return self.engine.step()
