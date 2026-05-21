from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Literal, NoReturn

from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.request import UADPhase

UADInputKind = Literal["ar_tokens", "dit_latent_step", "artifact_decode"]
UADOutputKind = Literal["sample_tokens", "denoise_pred", "artifact", "none"]


@dataclass(frozen=True)
class UADScheduleItem:
    """One UAD work item scheduled in the current EngineCore tick."""

    request_id: str
    phase: UADPhase
    num_scheduled_tokens: int
    input_kind: UADInputKind
    output_kind: UADOutputKind
    needs_kv_slots: bool = False
    num_persistent_tokens: int = 0
    step_index: int | None = None
    total_steps: int | None = None
    shape_bucket: tuple[int, ...] | None = None


@dataclass
class UADSchedulerOutput(SchedulerOutput):
    """v1 SchedulerOutput with UAD work items attached.

    The inherited SchedulerOutput fields keep their normal v1 meaning.  UAD
    metadata is additive, so existing v1 executor/worker paths can still treat
    this object as a regular SchedulerOutput.
    """

    uad_items: list[UADScheduleItem] = field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        base_output: SchedulerOutput,
        uad_items: list[UADScheduleItem] | None = None,
    ) -> UADSchedulerOutput:
        base_values = {
            field_info.name: getattr(base_output, field_info.name)
            for field_info in dataclass_fields(SchedulerOutput)
        }
        return cls(**base_values, uad_items=list(uad_items or []))

    @property
    def total_num_scheduled_uad_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.uad_items)

    @property
    def total_num_persistent_uad_tokens(self) -> int:
        return sum(item.num_persistent_tokens for item in self.uad_items)

    @property
    def total_num_scheduled_work(self) -> int:
        return self.total_num_scheduled_tokens + self.total_num_scheduled_uad_tokens

    @property
    def has_ar_sample_items(self) -> bool:
        return self.total_num_scheduled_tokens > 0


class UADScheduler(SchedulerInterface):
    """Minimal UAD scheduler facade over a base v1 scheduler.

    This class inherits SchedulerInterface for interface visibility and ABC
    checks, but it is not a full replacement for EngineCoreProc.scheduler.
    EngineCoreProc.scheduler remains the base scheduler; this object is used
    only by UADEngineCore.step().
    """

    def __init__(self, base_scheduler: SchedulerInterface) -> None:
        self.base_scheduler = base_scheduler

    def _unsupported_v1_method(self, name: str) -> NoReturn:
        raise NotImplementedError(
            f"UADScheduler.{name} is outside the UAD step scaffold. "
            "Use EngineCoreProc.scheduler for the v1 scheduler lifecycle."
        )

    def has_requests(self) -> bool:
        return self.base_scheduler.has_requests()

    def schedule(self) -> UADSchedulerOutput:
        base_output = self.base_scheduler.schedule()
        # TODO: inspect attached UAD request states and append DiT/artifact items.
        return UADSchedulerOutput.from_base(base_output, uad_items=[])

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput) -> GrammarOutput | None:
        return self.base_scheduler.get_grammar_bitmask(scheduler_output)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # TODO: apply UADModelRunnerOutput.phase_outputs before/after base scheduler update.
        return self.base_scheduler.update_from_output(scheduler_output, model_runner_output)

    def update_draft_token_ids(self, draft_token_ids: Any) -> None:
        self._unsupported_v1_method("update_draft_token_ids")

    def update_draft_token_ids_in_output(self, draft_token_ids: Any, scheduler_output: SchedulerOutput) -> None:
        self._unsupported_v1_method("update_draft_token_ids_in_output")

    def add_request(self, request: Any) -> None:
        self._unsupported_v1_method("add_request")

    def finish_requests(self, request_ids: Any, finished_status: Any) -> list[tuple[str, int]]:
        self._unsupported_v1_method("finish_requests")

    def get_num_unfinished_requests(self) -> int:
        self._unsupported_v1_method("get_num_unfinished_requests")

    def has_finished_requests(self) -> bool:
        self._unsupported_v1_method("has_finished_requests")

    @property
    def pause_state(self) -> Any:
        self._unsupported_v1_method("pause_state")

    def set_pause_state(self, pause_state: Any) -> None:
        self._unsupported_v1_method("set_pause_state")

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        self._unsupported_v1_method("reset_prefix_cache")

    def reset_encoder_cache(self) -> None:
        self._unsupported_v1_method("reset_encoder_cache")

    def get_request_counts(self) -> tuple[int, int]:
        self._unsupported_v1_method("get_request_counts")

    def make_stats(self) -> Any:
        self._unsupported_v1_method("make_stats")

    def shutdown(self) -> None:
        self._unsupported_v1_method("shutdown")
