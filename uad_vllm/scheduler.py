from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Literal, NoReturn

from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.request import UADPhase, UADRequestState

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
    """UAD-native scheduler scaffold with a v1-shaped interface."""

    def __init__(self) -> None:
        self.request_states: dict[str, UADRequestState] = {}
        self.finished_request_ids: set[str] = set()
        self._pause_state = PauseState.UNPAUSED

    def _unsupported_v1_method(self, name: str) -> NoReturn:
        raise NotImplementedError(
            f"UADScheduler.{name} is not part of the UAD scheduler scaffold. "
            "The v1 EngineCore lifecycle keeps using EngineCoreProc.scheduler."
        )

    def has_requests(self) -> bool:
        # TODO: return true when UAD request states can produce AR/DiT work.
        return False

    def schedule(self) -> UADSchedulerOutput:
        # TODO: schedule AR/DiT/artifact items from self.request_states.
        return UADSchedulerOutput.make_empty()

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput) -> GrammarOutput | None:
        del scheduler_output
        # TODO: add UAD-aware structured output support for AR phases.
        return None

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        del scheduler_output, model_runner_output
        # TODO: apply UADModelRunnerOutput.phase_outputs to request_states and
        # materialize EngineCoreOutputs.
        return {}

    def update_draft_token_ids(self, draft_token_ids: Any) -> None:
        del draft_token_ids
        self._unsupported_v1_method("update_draft_token_ids")

    def update_draft_token_ids_in_output(self, draft_token_ids: Any, scheduler_output: SchedulerOutput) -> None:
        del draft_token_ids, scheduler_output
        self._unsupported_v1_method("update_draft_token_ids_in_output")

    def add_request(self, request: Any) -> None:
        request_id = str(request.request_id)
        self.request_states[request_id] = UADRequestState(request_id=request_id)

    def finish_requests(self, request_ids: Any, finished_status: Any) -> list[tuple[str, int]]:
        del finished_status
        if request_ids is None:
            request_ids = list(self.request_states)
        elif isinstance(request_ids, str):
            request_ids = [request_ids]

        finished: list[tuple[str, int]] = []
        for request_id in request_ids:
            if request_id in self.request_states:
                del self.request_states[request_id]
                self.finished_request_ids.add(request_id)
                finished.append((request_id, 0))
        return finished

    def get_num_unfinished_requests(self) -> int:
        return len(self.request_states)

    def has_finished_requests(self) -> bool:
        return bool(self.finished_request_ids)

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: Any) -> None:
        self._pause_state = PauseState(pause_state)

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        del reset_running_requests, reset_connector
        self._unsupported_v1_method("reset_prefix_cache")

    def reset_encoder_cache(self) -> None:
        self._unsupported_v1_method("reset_encoder_cache")

    def get_request_counts(self) -> tuple[int, int]:
        return 0, len(self.request_states)

    def make_stats(self) -> Any:
        # TODO: add UAD scheduler stats.
        return None

    def shutdown(self) -> None:
        self.request_states.clear()
        self.finished_request_ids.clear()
