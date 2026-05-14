from __future__ import annotations

import pytest

from vllm_omni.uad.engine import UADEngine
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADForConditionalGeneration
from vllm_omni.uad.omni.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine
from vllm_omni.uad.request import UADToken
from vllm_omni.uad.runner import UADRunner

pytestmark = pytest.mark.cpu


def _build_engine() -> UADEngine:
    state_config = HunyuanImage3UADStateConfig(
        img_token_id=90,
        eoi_token_id=91,
        primary_ratio_token_start_id=50,
        primary_ratio_token_end_id=82,
        toy_image_context_token_count=3,
        toy_total_dit_steps=2,
    )
    model = HunyuanImage3UADForConditionalGeneration(vocab_size=300)
    state_machine = HunyuanImage3UADStateMachine(config=state_config)
    return UADEngine(runner=UADRunner(model=model), state_machine=state_machine)


def test_step5_scheduler_marks_only_final_dit_step_as_persisted() -> None:
    engine = _build_engine()
    request = engine.add_request("req-ratio", [49])

    engine.step()
    first_dit_item = engine.scheduler.schedule().scheduled_items[0]
    assert first_dit_item.phase == "dit_step"
    assert first_dit_item.persist is False
    assert first_dit_item.num_scheduled_tokens == request.image_context_token_count

    engine.step()
    final_dit_item = engine.scheduler.schedule().scheduled_items[0]
    assert final_dit_item.phase == "dit_step"
    assert final_dit_item.persist is True
    assert final_dit_item.num_scheduled_tokens == len(request.pending_token_ids())


def test_step5_final_dit_persists_pending_engine_context_for_next_turn() -> None:
    engine = _build_engine()
    request = engine.add_request("req-ratio", [49])

    engine.step()
    engine.step()
    assert request.num_computed_tokens == 1
    assert request.pending_token_ids()

    engine.step()
    assert request.phase == "ar_decode"
    assert request.pending_image_context_commit is False
    assert request.num_computed_tokens == len(request.engine_tokens)
    assert request.pending_token_ids() == []

    request.append_engine_tokens([UADToken(modality="text", token_id=100)])
    output = engine.step()

    assert output.request_ids == ["req-ratio"]
    assert request.num_computed_tokens == len(request.engine_tokens) - 1
    assert [token.token_id for token in output.outputs[0].new_materialized_tokens] == [101]
