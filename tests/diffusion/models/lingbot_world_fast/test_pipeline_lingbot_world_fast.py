# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast import (
    LingbotWorldFastPipeline,
    _LingbotSourceModules,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class CausalConv3d(torch.nn.Module):
    def forward(self, x, cache_x=None):
        del cache_x
        return x


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = CausalConv3d()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None and feat_idx is not None:
            feat_cache[feat_idx[0]] = True
            feat_idx[0] += 1
        x = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        x = F.interpolate(x, size=(2, 2), mode="nearest")
        return x.unsqueeze(2).repeat(1, 32, 1, 1, 1)


class _FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = CausalConv3d()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            is_first = feat_cache[idx] is None
            feat_cache[idx] = True
            feat_idx[0] += 1
        else:
            is_first = False
        frames = 1 if is_first else 4
        x = x.mean(dim=1, keepdim=False)[:, 0:1]
        x = F.interpolate(x, size=(16, 16), mode="nearest")
        return x.unsqueeze(2).repeat(1, 3, frames, 1, 1)


class _FakeConv1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = x[:, :16]
        logvar = torch.zeros_like(mu)
        return torch.cat([mu, logvar], dim=1)


class _FakeConv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


class _FakeVAEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()
        self.conv1 = _FakeConv1()
        self.conv2 = _FakeConv2()


class _FakeVAE:
    def __init__(self, vae_pth, dtype, device):
        del vae_pth
        self.mean = torch.zeros(16, dtype=dtype, device=device)
        self.std = torch.ones(16, dtype=dtype, device=device)
        self.model = _FakeVAEModel().to(device)


class _FakeT5EncoderModel:
    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, texts, device):
        del texts
        return [torch.ones((4, 8), device=device, dtype=torch.float32)]


class _FakeWanModelFast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_dim=16, num_layers=2, dim=64, num_heads=4)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return cls()

    def forward(self, x, t, **kwargs):
        del t, kwargs
        return [x[0].float() * 0.0 + 0.1]


def _make_request(*, session_id: str, reset: bool) -> OmniDiffusionRequest:
    image = PIL.Image.new("RGB", (16, 16), color=(32, 64, 128))
    prompt = OmniTextPrompt(
        prompt="robot world",
        multi_modal_data={"image": image},
    )
    control = {
        "poses": np.repeat(np.eye(4, dtype=np.float32)[None, ...], 13, axis=0),
        "intrinsics": np.array([[400.0, 401.0, 200.0, 201.0]], dtype=np.float32),
        "control_type": "cam",
    }
    sampling = OmniDiffusionSamplingParams(
        height=16,
        width=16,
        fps=16,
        seed=0,
        extra_args={
            "realtime_video": {
                "session_id": session_id,
                "rendered_prompt": "robot world",
                "text_layers": {"scene": "robot world"},
                "control": [control],
                "chunk_size": 3,
                "shift": 5.0,
                "reset": reset,
            }
        },
    )
    return OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=sampling,
        request_ids=[f"req-{session_id}"],
    )


def test_lingbot_world_fast_pipeline_preserves_runtime_state(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast.get_local_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast._resolve_file",
        lambda repo_or_path, relative_path: f"/tmp/{relative_path}",
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast._load_source_modules",
        lambda: _LingbotSourceModules(
            WanModelFast=_FakeWanModelFast,
            T5EncoderModel=_FakeT5EncoderModel,
            Wan2_1_VAE=_FakeVAE,
        ),
    )

    pipeline = LingbotWorldFastPipeline(
        od_config=OmniDiffusionConfig(
            model="robbyant/lingbot-world-fast",
            model_paths={"base_model": "robbyant/lingbot-world-base-cam"},
            dtype=torch.float32,
        )
    )

    first = pipeline.forward(_make_request(session_id="s1", reset=True))
    second = pipeline.forward(_make_request(session_id="s1", reset=False))

    assert first.custom_output["video_chunk"].shape[0] == 9
    assert second.custom_output["video_chunk"].shape[0] == 12
    assert pipeline.sessions["s1"].generated_chunks == 2
    assert pipeline.sessions["s1"].generated_latent_frames == 6
    assert pipeline.reset_realtime_video_session("s1") is True
    assert "s1" not in pipeline.sessions
    assert pipeline.reset_realtime_video_session("s1") is False
