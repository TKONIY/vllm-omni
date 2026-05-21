from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from uad_vllm.executor import UADExecutor
from uad_vllm.runner import UADRunner
from uad_vllm.scheduler import UADScheduler
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

logger = init_logger(__name__)


class UADEngineCore(StageEngineCoreProc):
    """EngineCoreProc subclass with UAD-aware step orchestration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.uad_scheduler = UADScheduler(self.scheduler)
        self.uad_executor = UADExecutor(self.model_executor)
        self.uad_runner = UADRunner()

        if self.batch_queue is not None:
            # TODO: add a UAD-aware async batch queue.  The upstream queue assumes
            # every executable item follows AR execute -> sample semantics.
            logger.warning("UAD engine disables upstream EngineCore batch_queue until UAD queue semantics are added.")
            self.batch_queue = None
            self.step_fn = self.step

    def step(self) -> tuple[dict[int, Any], bool]:
        """Run one UAD scheduler/executor/update iteration.

        This scaffold preserves AR passthrough behavior while routing control
        through UAD modules.  DiT and artifact work items will be added behind
        the TODOs in UADScheduler, UADExecutor, and UADRunner.
        """

        if not self.uad_scheduler.has_requests():
            return {}, False

        scheduler_output = self.uad_scheduler.schedule()
        future = self.uad_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.uad_scheduler.get_grammar_bitmask(scheduler_output)

        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            raw_model_output = future.result()
            if raw_model_output is None and scheduler_output.has_ar_sample_items:
                raw_model_output = self.uad_executor.sample_tokens(grammar_output)

        self._process_aborts_queue()
        runner_output = self.uad_runner.process_outputs(scheduler_output, raw_model_output)
        engine_core_outputs = self.uad_scheduler.update_from_output(scheduler_output, runner_output)

        return engine_core_outputs, scheduler_output.total_num_scheduled_work > 0

    def step_with_batch_queue(self) -> tuple[dict[int, Any] | None, bool]:
        # TODO: implement UAD-aware async batch queue semantics.
        return self.step()
