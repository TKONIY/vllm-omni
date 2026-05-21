from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor import Executor
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.outputs import UADModelRunnerOutput


class UADExecutor(Executor):
    """UAD-native executor scaffold with a v1-shaped interface."""

    def __init__(self) -> None:
        self.last_scheduler_output: SchedulerOutput | None = None

    def _init_executor(self) -> None:
        # TODO: initialize UAD worker/model-runner backend.
        return None

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> Any:
        del method, timeout, args, kwargs, non_block
        raise NotImplementedError("UADExecutor.collective_rpc is not implemented in the scaffold.")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        self.last_scheduler_output = scheduler_output
        # TODO: execute UAD AR/DiT/artifact items in the same worker group.
        if non_block:
            future: Future[ModelRunnerOutput | None] = Future()
            future.set_result(None)
            return future
        return None

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
        non_block: bool = False,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        del grammar_output
        output = UADModelRunnerOutput.make_empty()
        if non_block:
            future: Future[ModelRunnerOutput] = Future()
            future.set_result(output)
            return future
        return output

    def check_health(self) -> None:
        # TODO: check the UAD worker/model-runner backend once it exists.
        return None

    @property
    def max_concurrent_batches(self) -> int:
        # UADEngineCore disables the upstream batch_queue for now.
        return 1
