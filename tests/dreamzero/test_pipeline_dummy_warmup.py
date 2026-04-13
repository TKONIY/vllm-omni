"""Regression tests for DreamZero formal-service warmup behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _make_minimal_pipeline() -> DreamZeroPipeline:
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    nn.Module.__init__(pipe)
    pipe.action_horizon = 24
    pipe.max_action_dim = 32
    return pipe


class _FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(image_size=224)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        assert image.dtype == self._dummy.dtype
        batch = image.shape[0]
        return torch.zeros(batch, 257, 1280, dtype=self._dummy.dtype, device=image.device)


class _FakeLatentDist:
    def __init__(self, latent: torch.Tensor):
        self._latent = latent

    def mode(self):
        return self._latent


class _FakeVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.last_input_dtype = None

    def encode(self, x: torch.Tensor):
        self.last_input_dtype = x.dtype
        latent_t = (x.shape[2] + 3) // 4
        latent = torch.ones(
            x.shape[0],
            16,
            latent_t,
            x.shape[3] // 8,
            x.shape[4] // 8,
            dtype=torch.float32,
            device=x.device,
        )
        return SimpleNamespace(latent_dist=_FakeLatentDist(latent))

    def _encode(self, x: torch.Tensor):
        self.last_input_dtype = x.dtype
        latent_t = (x.shape[2] + 3) // 4
        mu = torch.ones(
            x.shape[0],
            16,
            latent_t,
            x.shape[3] // 8,
            x.shape[4] // 8,
            dtype=torch.float32,
            device=x.device,
        )
        logvar = torch.zeros_like(mu)
        return torch.cat([mu, logvar], dim=1)


def test_forward_allows_dummy_warmup_without_unified_obs():
    pipe = _make_minimal_pipeline()
    req = OmniDiffusionRequest(
        prompts=["dummy run"],
        request_ids=["dummy_req_id"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
        ),
    )

    output = DreamZeroPipeline.forward(pipe, req)

    assert output.error is None
    assert isinstance(output.output, dict)
    assert "actions" in output.output
    assert output.output["actions"].shape == (24, 32)
    assert output.output["actions"].dtype == np.float32


def test_forward_requires_unified_obs_for_real_requests():
    pipe = _make_minimal_pipeline()
    req = OmniDiffusionRequest(
        prompts=["pick up the red block"],
        request_ids=["req-1"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            extra_args={},
        ),
    )

    with pytest.raises(KeyError, match="unified_obs"):
        DreamZeroPipeline.forward(pipe, req)


def test_encode_image_bridges_vae_dtype():
    pipe = _make_minimal_pipeline()
    pipe.image_encoder = _FakeImageEncoder()
    pipe.vae = _FakeVAE()
    pipe.register_buffer(
        "vae_latents_mean",
        torch.zeros(1, 16, 1, 1, 1, dtype=torch.float32),
        persistent=False,
    )
    pipe.register_buffer(
        "vae_latents_inv_std",
        torch.ones(1, 16, 1, 1, 1, dtype=torch.float32),
        persistent=False,
    )

    image = torch.zeros(1, 1, 3, 352, 640, dtype=torch.bfloat16)
    clip_context, ys, new_image = pipe._encode_image(image, num_frames=81, height=352, width=640)

    assert pipe.vae.last_input_dtype == torch.float32
    assert clip_context.dtype == torch.bfloat16
    assert new_image.dtype == torch.bfloat16
    assert ys.shape[2] == 21
