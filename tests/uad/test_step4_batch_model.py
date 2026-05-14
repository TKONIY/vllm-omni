from __future__ import annotations

import pytest

from vllm_omni.uad.engine import UADEngine
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADModel
from vllm_omni.uad.omni.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine
from vllm_omni.uad.runner import UADRunner

pytestmark = pytest.mark.cpu


def _build_engine() -> tuple[UADEngine, UADRunner, HunyuanImage3UADModel]:
    state_config = HunyuanImage3UADStateConfig(
        img_token_id=90,
        eoi_token_id=91,
        primary_ratio_token_start_id=50,
        primary_ratio_token_end_id=82,
        toy_image_context_token_count=3,
        toy_total_dit_steps=2,
    )
    model = HunyuanImage3UADModel(vocab_size=300)
    runner = UADRunner(model=model)
    state_machine = HunyuanImage3UADStateMachine(config=state_config)
    return UADEngine(runner=runner, state_machine=state_machine), runner, model


def test_step4_runner_calls_hunyuan_uad_model_once_for_mixed_tick() -> None:
    engine, runner, model = _build_engine()
    dit_request = engine.add_request("req-dit", [49])
    engine.step()
    ar_request = engine.add_request("req-ar", [10])

    output = engine.step()
    batch_inputs = runner.last_batch_inputs
    batch_outputs = runner.last_batch_outputs

    assert batch_inputs is not None
    assert batch_outputs is not None
    assert model.forward_calls == 2
    assert output.request_ids == ["req-dit", "req-ar"]
    assert [item.request_id for item in batch_inputs.items] == ["req-dit", "req-ar"]
    assert [item.phase for item in batch_inputs.items] == ["dit_step", "ar_prefill"]
    assert batch_inputs.num_dit_tokens == dit_request.image_context_token_count
    assert batch_inputs.num_ar_tokens == 1
    assert batch_inputs.num_ffn_tokens == dit_request.image_context_token_count + 1
    assert batch_outputs.num_ffn_tokens == batch_inputs.num_ffn_tokens
    assert [token.token_id for token in ar_request.materialized_tokens] == [11]


def test_step4_batch_metadata_exposes_causal_and_dit_attention_work() -> None:
    engine, runner, _ = _build_engine()
    dit_request = engine.add_request("req-dit", [49])
    engine.step()
    engine.add_request("req-ar", [10, 11])

    engine.step()
    batch_inputs = runner.last_batch_inputs

    assert batch_inputs is not None
    assert batch_inputs.causal_attention_item_indices == (0, 1)
    assert batch_inputs.bidirectional_attention_item_indices == (0,)
    assert batch_inputs.num_causal_attention_tokens == batch_inputs.num_ffn_tokens
    assert batch_inputs.num_bidirectional_attention_tokens == dit_request.image_context_token_count
    assert batch_inputs.items[0].input_kind == "latent_timestep"
    assert batch_inputs.items[1].input_kind == "token_ids"
    assert batch_inputs.items[0].dit_step_index == 0
    assert batch_inputs.items[0].total_dit_steps == 2
    assert batch_inputs.token_positions.tolist() == [1, 2, 3, 4, 0, 1]


def test_step4_batch_output_scatter_restores_scheduler_item_order() -> None:
    engine, runner, model = _build_engine()
    req_a = engine.add_request("req-a", [1, 2])
    req_b = engine.add_request("req-b", [5])

    output = engine.step()
    batch_inputs = runner.last_batch_inputs

    assert batch_inputs is not None
    assert model.forward_calls == 1
    assert output.request_ids == ["req-a", "req-b"]
    assert [token.token_id for token in output.outputs[0].new_materialized_tokens] == [3]
    assert [token.token_id for token in output.outputs[1].new_materialized_tokens] == [6]
    assert req_a.num_computed_tokens == 2
    assert req_b.num_computed_tokens == 1
    assert [item.output_index for item in batch_inputs.items] == [0, 1]
