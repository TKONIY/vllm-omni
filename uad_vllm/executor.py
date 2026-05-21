from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor import Executor
from vllm.v1.outputs import ModelRunnerOutput


class UADExecutor(Executor):
    """Executor implementation that delegates to an existing v1 executor.

    The scaffold delegates AR work to the existing vLLM executor.  Future UAD
    steps can replace this facade with a full executor/worker stack while
    keeping UADEngineCore's step orchestration stable.
    """

    def __init__(self, base_executor: Any) -> None:
        self.base_executor = base_executor
        for attr in (
            "vllm_config",
            "model_config",
            "cache_config",
            "lora_config",
            "load_config",
            "parallel_config",
            "scheduler_config",
            "device_config",
            "speculative_config",
            "observability_config",
            "kv_output_aggregator",
            "is_sleeping",
            "sleeping_tags",
        ):
            if hasattr(base_executor, attr):
                setattr(self, attr, getattr(base_executor, attr))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_executor, name)

    def _init_executor(self) -> None:
        # The wrapped executor is already initialized by EngineCoreProc.
        return None

    def collective_rpc(
        self,
        method: str | Callable[..., Any],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> Any:
        return self.base_executor.collective_rpc(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            non_block=non_block,
        )

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
        return self.base_executor.check_health()

    @property
    def max_concurrent_batches(self) -> int:
        return self.base_executor.max_concurrent_batches
