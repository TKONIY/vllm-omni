from __future__ import annotations

import pytest
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.request import UADPhase
from uad_vllm.scheduler import UADScheduleItem, UADScheduler, UADSchedulerOutput

pytestmark = pytest.mark.cpu


class _BaseScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.output = SchedulerOutput.make_empty()
        self.output.num_scheduled_tokens = {"req": 2}
        self.output.total_num_scheduled_tokens = 2

    def has_requests(self) -> bool:
        self.calls.append(("has_requests", None))
        return True

    def schedule(self) -> object:
        self.calls.append(("schedule", None))
        return self.output

    def get_grammar_bitmask(self, scheduler_output: object) -> str:
        self.calls.append(("get_grammar_bitmask", scheduler_output))
        return "grammar"

    def update_from_output(self, scheduler_output: object, model_output: object) -> dict[int, str]:
        self.calls.append(("update_from_output", (scheduler_output, model_output)))
        return {0: "ok"}


def test_uad_scheduler_wraps_base_scheduler_output() -> None:
    base = _BaseScheduler()
    scheduler = UADScheduler(base)

    assert isinstance(scheduler, SchedulerInterface)
    assert scheduler.has_requests()
    output = scheduler.schedule()

    assert isinstance(output, SchedulerOutput)
    assert isinstance(output, UADSchedulerOutput)
    assert output.uad_items == []
    assert output.total_num_scheduled_tokens == 2
    assert output.total_num_scheduled_uad_tokens == 0
    assert output.total_num_scheduled_work == 2
    assert output.has_ar_sample_items


def test_uad_scheduler_delegates_grammar_and_update() -> None:
    base = _BaseScheduler()
    scheduler = UADScheduler(base)
    output = scheduler.schedule()
    runner_output = ModelRunnerOutput(req_ids=["req"], req_id_to_index={"req": 0})

    assert scheduler.get_grammar_bitmask(output) == "grammar"
    assert scheduler.update_from_output(output, runner_output) == {0: "ok"}
    assert ("get_grammar_bitmask", output) in base.calls
    assert ("update_from_output", (output, runner_output)) in base.calls


def test_uad_scheduler_output_tracks_persistent_uad_tokens() -> None:
    base_output = SchedulerOutput.make_empty()
    item = UADScheduleItem(
        request_id="req",
        phase=UADPhase.DIT_STEP,
        num_scheduled_tokens=4,
        input_kind="dit_latent_step",
        output_kind="denoise_pred",
        needs_kv_slots=True,
        num_persistent_tokens=2,
    )

    output = UADSchedulerOutput.from_base(base_output, uad_items=[item])

    assert output.total_num_scheduled_uad_tokens == 4
    assert output.total_num_persistent_uad_tokens == 2
    assert output.total_num_scheduled_work == 4


def test_uad_scheduler_lifecycle_methods_are_not_part_of_step_facade() -> None:
    scheduler = UADScheduler(_BaseScheduler())

    with pytest.raises(NotImplementedError, match="EngineCoreProc.scheduler"):
        scheduler.add_request(object())
