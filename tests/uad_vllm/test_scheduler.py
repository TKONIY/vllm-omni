from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.request import UADPhase
from uad_vllm.scheduler import UADScheduleItem, UADScheduler, UADSchedulerOutput

pytestmark = pytest.mark.cpu


def test_uad_scheduler_is_standalone_v1_shaped_scaffold() -> None:
    scheduler = UADScheduler()

    assert isinstance(scheduler, SchedulerInterface)
    assert not hasattr(scheduler, "base_scheduler")
    assert not scheduler.has_requests()
    output = scheduler.schedule()

    assert isinstance(output, SchedulerOutput)
    assert isinstance(output, UADSchedulerOutput)
    assert output.uad_items == []
    assert output.total_num_scheduled_tokens == 0
    assert output.total_num_scheduled_uad_tokens == 0
    assert output.total_num_scheduled_work == 0
    assert not output.has_ar_sample_items


def test_uad_scheduler_returns_empty_step_outputs() -> None:
    scheduler = UADScheduler()
    output = scheduler.schedule()
    runner_output = ModelRunnerOutput(req_ids=["req"], req_id_to_index={"req": 0})

    assert scheduler.get_grammar_bitmask(output) is None
    assert scheduler.update_from_output(output, runner_output) == {}


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


def test_uad_scheduler_tracks_request_state_without_base_scheduler() -> None:
    scheduler = UADScheduler()

    scheduler.add_request(SimpleNamespace(request_id="req"))

    assert scheduler.get_num_unfinished_requests() == 1
    assert scheduler.get_request_counts() == (0, 1)
    assert scheduler.pause_state == PauseState.UNPAUSED

    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert scheduler.pause_state == PauseState.PAUSED_ALL

    assert scheduler.finish_requests("req", object()) == [("req", 0)]
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.has_finished_requests()


def test_uad_scheduler_marks_unowned_v1_methods_unsupported() -> None:
    scheduler = UADScheduler()

    with pytest.raises(NotImplementedError, match="not part of the UAD scheduler scaffold"):
        scheduler.update_draft_token_ids(object())
    with pytest.raises(NotImplementedError, match="not part of the UAD scheduler scaffold"):
        scheduler.reset_prefix_cache()
    with pytest.raises(NotImplementedError, match="not part of the UAD scheduler scaffold"):
        scheduler.reset_encoder_cache()
