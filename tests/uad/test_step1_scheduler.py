from __future__ import annotations

import pytest

from vllm_omni.uad.request import UADRequestState, UADToken
from vllm_omni.uad.scheduler import UADToyScheduler

pytestmark = pytest.mark.cpu


def test_step1_shadow_scheduler_builds_prefill_item() -> None:
    request = UADRequestState.from_prompt_token_ids("req-prefill", [10, 11, 12])

    output = UADToyScheduler().schedule([request], base_output={"source": "base"})
    item = output.scheduled_items[0]

    assert output.base_output == {"source": "base"}
    assert output.request_ids == ["req-prefill"]
    assert item.request_id == "req-prefill"
    assert item.phase == "ar_prefill"
    assert item.token_ids == [10, 11, 12]
    assert item.num_scheduled_tokens == 3
    assert item.num_computed_tokens == 0


def test_step1_shadow_scheduler_builds_decode_item() -> None:
    request = UADRequestState.from_prompt_token_ids("req-decode", [7])
    request.advance_computed_tokens(1)
    request.append_engine_tokens([UADToken(modality="text", token_id=8)])

    output = UADToyScheduler().schedule([request])
    item = output.scheduled_items[0]

    assert item.phase == "ar_decode"
    assert item.token_ids == [8]
    assert item.num_scheduled_tokens == 1
    assert item.num_computed_tokens == 1


def test_step1_shadow_scheduler_skips_finished_requests() -> None:
    finished = UADRequestState.from_prompt_token_ids("finished", [1])
    finished.finished = True
    active = UADRequestState.from_prompt_token_ids("active", [2])

    output = UADToyScheduler().schedule([finished, active])

    assert output.request_ids == ["active"]
    assert output.num_scheduled_tokens_by_request == {"active": 1}
    assert output.total_num_scheduled_tokens == 1


def test_step1_shadow_scheduler_totals_match_items() -> None:
    req0 = UADRequestState.from_prompt_token_ids("req-0", [1, 2])
    req1 = UADRequestState.from_prompt_token_ids("req-1", [3])

    output = UADToyScheduler().schedule([req0, req1])

    assert output.num_scheduled_tokens_by_request == {"req-0": 2, "req-1": 1}
    assert output.total_num_scheduled_tokens == 3
    assert output.total_num_scheduled_tokens == sum(item.num_scheduled_tokens for item in output.scheduled_items)
