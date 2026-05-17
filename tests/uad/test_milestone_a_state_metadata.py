from __future__ import annotations

import pytest

from vllm_omni.uad.outputs import UADModelRunnerItemOutput
from vllm_omni.uad.request import UADRequestState, UADToken
from vllm_omni.uad.state.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine

pytestmark = pytest.mark.cpu


class FakeHunyuanTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.token_ids = {
            "<img>": 90,
            "<boi>": 91,
            "<eoi>": 92,
            "<cfg>": 93,
            "</think>": 100,
            "<recaption>": 101,
            "</recaption>": 102,
            "<answer>": 103,
            "</answer>": 104,
            "<img_size_1024>": 110,
            "<img_ratio_0>": 200,
            "<img_ratio_32>": 232,
            "<img_ratio_33>": 300,
            "<img_ratio_36>": 303,
        }

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.token_ids.get(token)


def test_milestone_a_state_config_reads_real_tokenizer_rules() -> None:
    config = HunyuanImage3UADStateConfig.from_tokenizer(
        FakeHunyuanTokenizer(),
        num_inference_steps=7,
        guidance_scale=1.0,
        seed=123,
    )

    assert config.img_token_id == 90
    assert config.stage_transitions == {
        100: [101],
        102: [103, 91, 110],
    }
    assert config.should_restrict_to_ratio_after(110)
    assert config.ratio_index(200) == 0
    assert config.ratio_index(232) == 32
    assert config.ratio_index(300) == 33
    assert config.ratio_index(303) == 36
    assert config.should_force_eos_after(300)


def test_milestone_a_generation_metadata_matches_hunyuan_ratio_bucket_math() -> None:
    config = HunyuanImage3UADStateConfig.from_tokenizer(
        FakeHunyuanTokenizer(),
        num_inference_steps=7,
        seed=123,
    )

    metadata = config.build_generation_metadata(16)

    assert metadata.base_size == 1024
    assert metadata.ratio_index == 16
    assert metadata.image_height == 1024
    assert metadata.image_width == 1024
    assert metadata.token_height == 64
    assert metadata.token_width == 64
    assert metadata.image_token_count == 4096
    assert metadata.image_context_token_count == 4098
    assert metadata.latent_shape == (16, 128, 128)
    assert metadata.num_inference_steps == 7
    assert metadata.seed == 123

    assert config.ratio_index_for_size(width=2048, height=512) == 0
    assert config.build_generation_metadata_from_size(width=1024, height=1024) == metadata


def test_milestone_a_ratio_state_update_records_dit_metadata_on_request() -> None:
    config = HunyuanImage3UADStateConfig.from_tokenizer(
        FakeHunyuanTokenizer(),
        num_inference_steps=7,
        seed=123,
    )
    state_machine = HunyuanImage3UADStateMachine(config=config)
    request = UADRequestState.from_prompt_token_ids("req", [10])

    update = state_machine.update_request_state(
        request=request,
        runner_output=UADModelRunnerItemOutput(
            request_id="req",
            phase="ar_decode",
            num_scheduled_tokens=1,
            sampled_token=UADToken(modality="text", token_id=216),
        ),
    )
    request.append_engine_tokens(update.new_engine_tokens)
    request.append_materialized_tokens(update.new_materialized_tokens)
    assert update.phase_update is not None
    request.apply_phase_update(update.phase_update)

    assert request.phase == "dit_step"
    assert request.image_ratio_token_id == 216
    assert request.image_ratio_index == 16
    assert request.image_width == 1024
    assert request.image_height == 1024
    assert request.image_token_width == 64
    assert request.image_token_height == 64
    assert request.image_context_token_count == 4098
    assert request.latent_shape == (16, 128, 128)
    assert request.total_dit_steps == 7
    assert request.num_inference_steps == 7
    assert request.seed == 123
    assert request.pending_image_context_commit is True
    assert len(update.new_engine_tokens) == 4099
    assert update.new_engine_tokens[0].token_id == 216
    assert update.new_engine_tokens[-1].token_id == 92
    assert update.new_materialized_tokens == []


def test_milestone_a_request_overrides_default_dit_runtime_metadata() -> None:
    config = HunyuanImage3UADStateConfig.from_tokenizer(FakeHunyuanTokenizer())
    state_machine = HunyuanImage3UADStateMachine(config=config)
    request = UADRequestState.from_prompt_token_ids("req", [10])
    request.seed = 9
    request.num_inference_steps = 3
    request.guidance_scale = 1.25

    update = state_machine.update_request_state(
        request=request,
        runner_output=UADModelRunnerItemOutput(
            request_id="req",
            phase="ar_decode",
            num_scheduled_tokens=1,
            sampled_token=UADToken(modality="text", token_id=216),
        ),
    )

    assert update.phase_update is not None
    assert update.phase_update.seed == 9
    assert update.phase_update.num_inference_steps == 3
    assert update.phase_update.total_dit_steps == 3
    assert update.phase_update.guidance_scale == 1.25
