from __future__ import annotations

import pytest

from vllm_omni.uad.request import UADToken
from vllm_omni.uad.scheduler import UADToyScheduler

pytestmark = pytest.mark.cpu


def test_step1_shadow_scheduler_builds_prefill_item() -> None:
    scheduler = UADToyScheduler()
    scheduler.add_request("req-prefill", [10, 11, 12])

    output = scheduler.schedule(base_output={"source": "base"})
    item = output.scheduled_items[0]

    assert output.base_output == {"source": "base"}
    assert output.request_ids == ["req-prefill"]
    assert item.request_id == "req-prefill"
    assert item.phase == "ar_prefill"
    assert item.token_ids == [10, 11, 12]
    assert item.num_scheduled_tokens == 3
    assert item.num_computed_tokens == 0


def test_step1_shadow_scheduler_builds_decode_item() -> None:
    scheduler = UADToyScheduler()
    request = scheduler.add_request("req-decode", [7])
    request.advance_computed_tokens(1)
    request.append_engine_tokens([UADToken(modality="text", token_id=8)])

    output = scheduler.schedule()
    item = output.scheduled_items[0]

    assert item.phase == "ar_decode"
    assert item.token_ids == [8]
    assert item.num_scheduled_tokens == 1
    assert item.num_computed_tokens == 1


def test_step1_shadow_scheduler_skips_finished_requests() -> None:
    scheduler = UADToyScheduler()
    finished = scheduler.add_request("finished", [1])
    finished.finished = True
    scheduler.add_request("active", [2])

    output = scheduler.schedule()

    assert output.request_ids == ["active"]
    assert output.num_scheduled_tokens_by_request == {"active": 1}
    assert output.total_num_scheduled_tokens == 1


def test_step1_shadow_scheduler_totals_match_items() -> None:
    scheduler = UADToyScheduler()
    scheduler.add_request("req-0", [1, 2])
    scheduler.add_request("req-1", [3])

    output = scheduler.schedule()

    assert output.num_scheduled_tokens_by_request == {"req-0": 2, "req-1": 1}
    assert output.total_num_scheduled_tokens == 3
    assert output.total_num_scheduled_tokens == sum(item.num_scheduled_tokens for item in output.scheduled_items)
