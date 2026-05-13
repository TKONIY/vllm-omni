from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.engine import UADEngine
from vllm_omni.uad.omni.adapter.hunyuan_image3 import (
    HunyuanImage3UADAdapter,
    HunyuanImage3UADStateConfig,
)
from vllm_omni.uad.runner import UADRunner
from vllm_omni.uad.scheduler import UADToyScheduler

pytestmark = pytest.mark.cpu


def _build_engine(state_config: HunyuanImage3UADStateConfig) -> UADEngine:
    model = HunyuanImage3UADForConditionalGeneration(vocab_size=300)
    adapter = HunyuanImage3UADAdapter(model=model, state_config=state_config)
    return UADEngine(runner=UADRunner(adapter=adapter))


def test_step2_ratio_token_switches_request_to_dit_phase() -> None:
    state_config = HunyuanImage3UADStateConfig(
        img_token_id=90,
        eoi_token_id=91,
        primary_ratio_token_start_id=50,
        primary_ratio_token_end_id=82,
        toy_image_context_token_count=3,
        toy_total_dit_steps=2,
    )
    engine = _build_engine(state_config)

    request = engine.add_request("req-ratio", [49])
    output = engine.step()

    assert output.request_ids == ["req-ratio"]
    assert [token.token_id for token in output.outputs[0].new_engine_tokens] == [50, 90, 90, 90, 91]
    assert [token.token_id for token in output.outputs[0].new_materialized_tokens] == []
    assert [token.token_id for token in request.engine_tokens] == [49, 50, 90, 90, 90, 91]
    assert [token.token_id for token in request.materialized_tokens] == []
    assert request.num_computed_tokens == 1
    assert request.phase == "dit_step"
    assert request.dit_step_index == 0
    assert request.total_dit_steps == 2
    assert request.image_ratio_token_id == 50
    assert request.image_ratio_index == 0
    assert request.image_context_token_count == 4
    assert request.pending_image_context_commit is True


def test_step2_scheduler_does_not_ar_schedule_dit_pending_tokens() -> None:
    engine = _build_engine(
        HunyuanImage3UADStateConfig(
            img_token_id=90,
            eoi_token_id=91,
            primary_ratio_token_start_id=50,
            primary_ratio_token_end_id=82,
        )
    )
    request = engine.add_request("req-ratio", [49])
    engine.step()

    scheduler_output = UADToyScheduler().schedule([request])

    assert request.phase == "dit_step"
    assert request.pending_token_ids()
    assert scheduler_output.scheduled_items == []


def test_step2_normal_text_token_still_materializes() -> None:
    engine = _build_engine(
        HunyuanImage3UADStateConfig(
            img_token_id=90,
            primary_ratio_token_start_id=50,
            primary_ratio_token_end_id=82,
        )
    )

    request = engine.add_request("req-text", [10])
    output = engine.step()

    assert [token.token_id for token in output.outputs[0].new_engine_tokens] == [11]
    assert [token.token_id for token in output.outputs[0].new_materialized_tokens] == [11]
    assert [token.token_id for token in request.materialized_tokens] == [11]
    assert request.phase == "ar_decode"


def test_step2_control_token_is_engine_only() -> None:
    engine = _build_engine(
        HunyuanImage3UADStateConfig(
            img_token_id=90,
            primary_ratio_token_start_id=150,
            primary_ratio_token_end_id=182,
        )
    )

    request = engine.add_request("req-control", [89])
    output = engine.step()

    assert [token.token_id for token in output.outputs[0].new_engine_tokens] == [90]
    assert output.outputs[0].new_materialized_tokens == []
    assert request.materialized_tokens == []
    assert request.phase == "ar_decode"


def test_step2_hunyuan_stage_transition_rules_match_existing_sampler() -> None:
    state_config = HunyuanImage3UADStateConfig(
        end_think_token_id=20,
        recaption_token_id=21,
        end_recaption_token_id=30,
        answer_token_id=31,
        boi_token_id=32,
        image_size_token_id=33,
        eos_token_id=2,
        primary_ratio_token_start_id=50,
        primary_ratio_token_end_id=82,
    )

    assert state_config.get_forced_token([20]) == 21
    assert state_config.get_forced_token([30]) == 31
    assert state_config.get_forced_token([30, 31]) == 32
    assert state_config.get_forced_token([30, 31, 32]) == 33
    assert state_config.get_forced_token([30, 31, 32, 33]) is None
    assert state_config.ratio_index(50) == 0
    assert state_config.should_force_eos_after(50)
