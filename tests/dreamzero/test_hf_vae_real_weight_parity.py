# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-weight parity check for DreamZero VAE encoding semantics.

The DreamZero source VAE and diffusers `AutoencoderKLWan` share the same encode
math in fp32, but bf16 parity depends on one subtle detail:

- source uses `(mu - mean) * (1 / std)` with the reciprocal precomputed in fp32
- a direct bf16 division by `std` is *not* bit-equivalent

This test locks the local path to the exact source behavior on a real DreamZero
checkpoint and real client input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from safetensors import safe_open

from vllm_omni.entrypoints.openai.realtime.robot.transform.roboarena import (
    RoboArenaTransform,
)

DREAMZERO_REPO = Path("~/code/dreamzero").expanduser()
CHECKPOINT_DIR = DREAMZERO_REPO / "checkpoints" / "dreamzero"
VAE_PTH = Path(
    "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/"
    "snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth",
).expanduser()
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
SESSION_ID = "hf-vae-real-weight-parity"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif(not DREAMZERO_REPO.exists(), reason="DreamZero source repo is required at ~/code/dreamzero"),
    pytest.mark.skipif(not CHECKPOINT_DIR.exists(), reason="DreamZero local checkpoint is required"),
    pytest.mark.skipif(not VAE_PTH.exists(), reason="Wan2.1_VAE.pth is required in local HF cache"),
]


def _load_real_vae_input():
    if str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    import test_client_AR as dreamzero_client

    camera_frames = dreamzero_client.load_camera_frames()
    obs = dreamzero_client._make_obs_from_video(camera_frames, [0], PROMPT, SESSION_ID)
    stitched = RoboArenaTransform().transform_input(obs)["images"]

    videos = torch.from_numpy(stitched).unsqueeze(0).permute(0, 4, 1, 2, 3).float() / 255.0
    videos = videos * 2.0 - 1.0
    videos = videos.to(device="cuda:0", dtype=torch.bfloat16)

    image = videos[:, :, :1].transpose(1, 2).contiguous()  # [B, 1, C, H, W]
    image_input = image.transpose(1, 2)  # [B, C, 1, H, W]
    image_zeros = torch.zeros(
        image.shape[0],
        3,
        32,
        image.shape[-2],
        image.shape[-1],
        device=image.device,
        dtype=image.dtype,
    )
    return torch.cat([image_input, image_zeros], dim=2)


def _iter_root_vae_weights():
    with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_file in index["weight_map"].items():
        if key.startswith("action_head.vae."):
            shard_to_keys.setdefault(shard_file, []).append(key)

    for shard_file, keys in sorted(shard_to_keys.items()):
        with safe_open(str(CHECKPOINT_DIR / shard_file), framework="pt", device="cpu") as f:
            for key in keys:
                yield key, f.get_tensor(key)


def test_diffusers_vae_matches_source_when_using_precomputed_inverse_std():
    if str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE

    vae_input = _load_real_vae_input()

    source_vae = WanVideoVAE().to(device=vae_input.device, dtype=torch.bfloat16).eval()
    source_vae.model.load_state_dict(torch.load(VAE_PTH, map_location="cpu"))

    diffusers_vae = (
        AutoencoderKLWan.from_pretrained(
            str(CHECKPOINT_DIR),
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        .to(device=vae_input.device, dtype=torch.bfloat16)
        .eval()
    )

    with torch.no_grad():
        source_latent = source_vae.encode(vae_input)

        hidden = diffusers_vae._encode(vae_input)
        mu, _ = hidden.chunk(2, dim=1)
        mean = torch.tensor(
            diffusers_vae.config.latents_mean,
            device=vae_input.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        inv_std = (
            1.0
            / torch.tensor(
                diffusers_vae.config.latents_std,
                device=vae_input.device,
                dtype=torch.float32,
            )
        ).view(1, -1, 1, 1, 1)
        local_latent = (mu - mean.to(dtype=mu.dtype)) * inv_std.to(dtype=mu.dtype)

    diff = (source_latent.float() - local_latent.float()).abs()
    assert diff.max().item() == 0.0
    assert diff.mean().item() == 0.0


def test_constructor_bootstrapped_distributed_vae_matches_source_after_root_remap():
    if str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    vae_input = _load_real_vae_input()

    source_vae = WanVideoVAE().to(device=vae_input.device, dtype=torch.bfloat16).eval()
    source_vae.model.load_state_dict(torch.load(VAE_PTH, map_location="cpu"))

    local_vae = DistributedAutoencoderKLWan().to(device=vae_input.device, dtype=torch.bfloat16).eval()
    local_params = dict(local_vae.named_parameters())
    loaded = set()

    for key, tensor in _iter_root_vae_weights():
        mapped = DreamZeroPipeline._remap_vae_key(key)
        assert mapped is not None, key
        assert mapped in local_params, mapped
        local_params[mapped].data.copy_(tensor)
        loaded.add(mapped)

    missing = sorted(set(local_params) - loaded)
    assert not missing, missing[:10]

    with torch.no_grad():
        source_latent = source_vae.encode(vae_input)

        hidden = local_vae._encode(vae_input)
        mu, _ = hidden.chunk(2, dim=1)
        mean = torch.tensor(
            local_vae.config.latents_mean,
            device=vae_input.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        inv_std = (
            1.0
            / torch.tensor(
                local_vae.config.latents_std,
                device=vae_input.device,
                dtype=torch.float32,
            )
        ).view(1, -1, 1, 1, 1)
        local_latent = (mu - mean.to(dtype=mu.dtype)) * inv_std.to(dtype=mu.dtype)

    diff = (source_latent.float() - local_latent.float()).abs()
    assert diff.max().item() == 0.0
    assert diff.mean().item() == 0.0
