from __future__ import annotations

from concurrent.futures import Future
from typing import Any, NoReturn

from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor import Executor
from vllm.v1.outputs import ModelRunnerOutput


class UADExecutor(Executor):
    """Minimal UAD executor facade over a base v1 executor.

    It inherits Executor for interface visibility and ABC checks, but it is not
    installed as EngineCoreProc.model_executor.  Only UADEngineCore.step() uses
    it in this scaffold.
    """

    def __init__(self, base_executor: Any) -> None:
        self.base_executor = base_executor

    def _unsupported_v1_method(self, name: str) -> NoReturn:
        raise NotImplementedError(
            f"UADExecutor.{name} is outside the UAD step scaffold. "
            "Use EngineCoreProc.model_executor for the v1 executor lifecycle."
        )

    def _init_executor(self) -> None:
        # The wrapped executor was initialized by EngineCoreProc before this
        # facade was created.
        return None

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> Any:
        self._unsupported_v1_method("collective_rpc")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # TODO: execute UAD DiT/artifact items in the same worker group.
        return self.base_executor.execute_model(scheduler_output, non_block=non_block)

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
        non_block: bool = False,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.base_executor.sample_tokens(grammar_output, non_block=non_block)

    def check_health(self) -> None:
        self._unsupported_v1_method("check_health")

    @property
    def max_concurrent_batches(self) -> int:
        # UADEngineCore disables the upstream batch_queue for now.
        return 1
