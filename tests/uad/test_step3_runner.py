from __future__ import annotations

import pytest

from vllm_omni.uad.engine import UADEngine
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADForConditionalGeneration
from vllm_omni.uad.outputs import UADModelOutput, UADRunnerOutput
from vllm_omni.uad.request import UADRequestState
from vllm_omni.uad.runner import UADRunner
from vllm_omni.uad.state.base import UADModelStateMachine
from vllm_omni.uad.state.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine

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


class RecordingStateMachine(UADModelStateMachine):
    def __init__(self) -> None:
        self.sampled_tokens: list[int] = []

    def update_request_state(
        self,
        *,
        request: UADRequestState,
        runner_output: UADRunnerOutput,
    ) -> UADModelOutput:
        assert runner_output.sampled_token is not None
        sampled_token = runner_output.sampled_token
        self.sampled_tokens.append(sampled_token.token_id)
        return UADModelOutput(
            request_id=request.request_id,
            new_engine_tokens=[sampled_token],
            new_materialized_tokens=[sampled_token],
            finished=False,
        )


def test_step3_runner_executes_fake_dit_steps_and_persists_final_step() -> None:
    engine = _build_engine()
    request = engine.add_request("req-ratio", [49])

    engine.step()
    assert request.phase == "dit_step"
    assert request.dit_step_index == 0
    assert request.num_computed_tokens == 1
    engine_tokens = list(request.engine_tokens)

    first_dit = engine.step()
    assert first_dit.request_ids == ["req-ratio"]
    assert first_dit.outputs[0].new_engine_tokens == []
    assert first_dit.outputs[0].new_materialized_tokens == []
    assert request.dit_step_index == 1
    assert request.num_computed_tokens == 1
    assert request.engine_tokens == engine_tokens
    assert request.pending_image_context_commit is True

    final_dit = engine.step()
    assert final_dit.request_ids == ["req-ratio"]
    assert request.dit_step_index == 2
    assert request.total_dit_steps == 2
    assert request.num_computed_tokens == len(request.engine_tokens)
    assert request.engine_tokens == engine_tokens
    assert request.phase == "ar_decode"
    assert request.pending_image_context_commit is False
    assert request.pending_token_ids() == []

    no_work = engine.step()
    assert no_work.outputs == []
    assert request.phase == "ar_decode"


def test_step3_runner_can_process_ar_and_dit_items_in_one_tick() -> None:
    engine = _build_engine()
    dit_request = engine.add_request("req-dit", [49])
    engine.step()

    ar_request = engine.add_request("req-ar", [10])
    mixed_output = engine.step()

    assert mixed_output.request_ids == ["req-dit", "req-ar"]
    assert dit_request.dit_step_index == 1
    assert dit_request.num_computed_tokens == 1
    assert [token.token_id for token in ar_request.engine_tokens] == [10, 11]
    assert [token.token_id for token in ar_request.materialized_tokens] == [11]
    assert ar_request.num_computed_tokens == 1


def test_step3_scheduler_update_delegates_runner_outputs_to_state_machine() -> None:
    model = HunyuanImage3UADForConditionalGeneration(vocab_size=300)
    state_machine = RecordingStateMachine()
    engine = UADEngine(runner=UADRunner(model=model), state_machine=state_machine)

    request = engine.add_request("req-custom", [49])
    output = engine.step()

    assert state_machine.sampled_tokens == [50]
    assert [token.token_id for token in output.outputs[0].new_engine_tokens] == [50]
    assert [token.token_id for token in request.materialized_tokens] == [50]
    assert request.num_computed_tokens == 1
    assert request.phase == "ar_decode"


def test_step3_runner_has_no_model_state_machine() -> None:
    runner = UADRunner()

    assert not hasattr(runner, "state_machine")


def test_step3_engine_delegates_state_updates_to_scheduler() -> None:
    engine = _build_engine()

    assert hasattr(engine.scheduler, "update_from_output")
    assert not hasattr(engine, "_process_runner_output")
