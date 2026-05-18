from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.uad.engine import UADEngine
from vllm_omni.uad.model.hunyuan_image3 import HunyuanImage3UADModel
from vllm_omni.uad.runner import UADRunner
from vllm_omni.uad.state.hunyuan_image3 import HunyuanImage3UADStateConfig, HunyuanImage3UADStateMachine

pytestmark = pytest.mark.cpu


class FakeReusableHunyuanARModel(nn.Module):
    def __init__(self, sampled_token_ids: list[int], vocab_size: int = 300) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.empty(0))
        self.sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self.forward_input_ids: list[list[int]] = []
        self.forward_positions: list[list[int]] = []
        self.observed_sampling_output_token_ids: list[list[list[int]]] = []
        self.compute_hidden_shapes: list[tuple[int, ...]] = []
        self.sample_calls = 0

    def forward(self, *, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self.forward_input_ids.append(input_ids.detach().cpu().tolist())
        self.forward_positions.append(positions.detach().cpu().tolist())
        hidden = torch.zeros((input_ids.numel(), 4), dtype=torch.float32, device=input_ids.device)
        hidden[:, 0] = input_ids.to(dtype=torch.float32)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_hidden_shapes.append(tuple(hidden_states.shape))
        token_id = self.sampled_token_ids[self.sample_calls]
        logits = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            -1000.0,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        logits[:, token_id] = 1000.0
        return logits

    def sample(self, logits: torch.Tensor, sampling_metadata) -> SimpleNamespace:
        self.observed_sampling_output_token_ids.append([list(row) for row in sampling_metadata.output_token_ids])
        token_id = self.sampled_token_ids[self.sample_calls]
        self.sample_calls += 1
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[token_id]], dtype=torch.int32, device=logits.device)
        )


def _build_engine(ar_model: nn.Module) -> UADEngine:
    state_config = HunyuanImage3UADStateConfig(
        img_token_id=90,
        eoi_token_id=91,
        primary_ratio_token_start_id=50,
        primary_ratio_token_end_id=82,
        toy_image_context_token_count=3,
        toy_total_dit_steps=2,
    )
    model = HunyuanImage3UADModel(vocab_size=300, ar_model=ar_model)
    state_machine = HunyuanImage3UADStateMachine(config=state_config)
    return UADEngine(runner=UADRunner(model=model), state_machine=state_machine)


def test_milestone_b_ar_items_reuse_backend_forward_logits_and_sampler() -> None:
    backend = FakeReusableHunyuanARModel(sampled_token_ids=[177, 178])
    engine = _build_engine(backend)
    request = engine.add_request("req", [10, 11])

    engine.step()
    engine.step()

    assert backend.forward_input_ids == [[10, 11], [177]]
    assert backend.forward_positions == [[0, 1], [2]]
    assert backend.compute_hidden_shapes == [(1, 4), (1, 4)]
    assert backend.observed_sampling_output_token_ids == [[[]], [[177]]]
    assert [token.token_id for token in request.materialized_tokens] == [177, 178]
    assert request.ar_sampler_token_ids == [177, 178]


def test_milestone_b_backend_sampled_ratio_token_still_switches_state_machine_to_dit() -> None:
    backend = FakeReusableHunyuanARModel(sampled_token_ids=[50])
    engine = _build_engine(backend)
    request = engine.add_request("req-ratio", [10])

    output = engine.step()

    assert output.request_ids == ["req-ratio"]
    assert backend.forward_input_ids == [[10]]
    assert request.phase == "dit_step"
    assert request.image_ratio_token_id == 50
    assert request.dit_step_index == 0
    assert request.total_dit_steps == 2
    assert request.ar_sampler_token_ids == [50]
    assert [token.token_id for token in request.engine_tokens] == [10, 50, 90, 90, 90, 91]


def test_milestone_b_real_ar_backend_rejects_multiple_ar_items_in_one_tick() -> None:
    backend = FakeReusableHunyuanARModel(sampled_token_ids=[177])
    engine = _build_engine(backend)
    engine.add_request("req-a", [10])
    engine.add_request("req-b", [20])

    with pytest.raises(NotImplementedError, match="single-request only"):
        engine.step()
