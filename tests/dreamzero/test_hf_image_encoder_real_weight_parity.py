# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-weight parity checks for the DreamZero image encoder.

This file answers the service-path question we actually care about:

1. Under DreamZero's real bf16 inference input path, which local implementation
   matches upstream `WanImageEncoder.encode_image()` exactly?
2. Why is `CLIPImageProcessor` still not acceptable?

Current conclusion:

- The local source-shaped port `DreamZeroImageEncoder` matches upstream
  exactly on real checkpoint weights + real service input.
- HF `CLIPVisionModel` still drifts on the same service input, even if
  weights are remapped correctly and preprocessing uses the upstream
  bicubic-resize + CLIP-normalize path.
- `CLIPImageProcessor` remains non-equivalent preprocessing for DreamZero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from vllm_omni.diffusion.models.dreamzero.modeling.image_encoder import (
    DreamZeroImageEncoder,
)
from vllm_omni.entrypoints.openai.realtime.robot.transform.roboarena import (
    RoboArenaTransform,
)

DREAMZERO_REPO = Path("~/code/dreamzero").expanduser()
CHECKPOINT_DIR = DREAMZERO_REPO / "checkpoints" / "dreamzero"
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
SESSION_ID = "hf-image-encoder-real-weight-parity"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif(not DREAMZERO_REPO.exists(), reason="DreamZero source repo is required at ~/code/dreamzero"),
    pytest.mark.skipif(not CHECKPOINT_DIR.exists(), reason="DreamZero local checkpoint is required"),
]


def _iter_image_encoder_weights():
    with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    image_keys = [k for k in weight_map if k.startswith("action_head.image_encoder.")]
    shard_to_keys: dict[str, list[str]] = {}
    for key in image_keys:
        shard_to_keys.setdefault(weight_map[key], []).append(key)
    for shard, keys in sorted(shard_to_keys.items()):
        with safe_open(CHECKPOINT_DIR / shard, framework="pt", device="cpu") as sf:
            for key in keys:
                yield key, sf.get_tensor(key)


def _load_real_input():
    sys.path.insert(0, str(DREAMZERO_REPO))
    import test_client_AR as tcar

    camera_frames = tcar.load_camera_frames()
    obs = tcar._make_obs_from_video(camera_frames, [0], PROMPT, SESSION_ID)
    transform = RoboArenaTransform()
    unified = transform.transform_input(obs)

    device = torch.device("cuda:0")
    stitched = unified["images"]  # (T, H, W, C), uint8
    videos = torch.from_numpy(stitched).unsqueeze(0).permute(0, 4, 1, 2, 3)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        videos = videos.to(device=device, dtype=torch.bfloat16)
        batch_size, channels, num_frames, height, width = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        videos = videos * 2.0 - 1.0
        videos = videos.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

    image = videos[:, :, :1].transpose(1, 2).contiguous()  # [B, 1, C, H, W]
    first_frame_hwc = stitched[0]
    first_frame_chw_uint8 = torch.from_numpy(first_frame_hwc).permute(2, 0, 1).contiguous()
    return image, first_frame_chw_uint8


def _load_upstream_encoder():
    sys.path.insert(0, str(DREAMZERO_REPO))
    from groot.vla.model.dreamzero.modules.wan_video_image_encoder import (
        WanImageEncoder,
    )

    device = torch.device("cuda:0")
    encoder = WanImageEncoder().to(device=device, dtype=torch.bfloat16).eval()
    state_dict = {name[len("action_head.image_encoder.") :]: tensor for name, tensor in _iter_image_encoder_weights()}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    assert not missing
    assert not unexpected
    return encoder


def _load_local_port():
    device = torch.device("cuda:0")
    encoder = DreamZeroImageEncoder().to(device=device, dtype=torch.bfloat16).eval()
    state_dict = {name[len("action_head.image_encoder.") :]: tensor for name, tensor in _iter_image_encoder_weights()}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    assert not missing
    assert not unexpected
    return encoder


def _load_hf_clip_model():
    device = torch.device("cuda:0")
    config = CLIPVisionConfig(
        hidden_size=1280,
        intermediate_size=5120,
        projection_dim=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
    )
    model = CLIPVisionModel(config).to(device=device, dtype=torch.bfloat16).eval()
    params = {"image_encoder." + name: param for name, param in model.named_parameters()}
    loaded: set[str] = set()
    for name, tensor in _iter_image_encoder_weights():
        _remap_image_encoder_key_to_hf(name, tensor, params, loaded)
    assert loaded == set(params)
    return model


def _remap_image_encoder_key_to_hf(
    name: str,
    tensor: torch.Tensor,
    params: dict[str, torch.nn.Parameter],
    loaded: set[str],
) -> None:
    if not name.startswith("action_head.image_encoder.model."):
        return

    subkey = name[len("action_head.image_encoder.model.") :]
    if subkey == "log_scale":
        return
    if subkey.startswith("visual.head"):
        return
    if not subkey.startswith("visual."):
        return

    visual_key = subkey[len("visual.") :]
    global_map = {
        "cls_embedding": "vision_model.embeddings.class_embedding",
        "patch_embedding.weight": "vision_model.embeddings.patch_embedding.weight",
        "pos_embedding": "vision_model.embeddings.position_embedding.weight",
        "pre_norm.weight": "vision_model.pre_layrnorm.weight",
        "pre_norm.bias": "vision_model.pre_layrnorm.bias",
        "post_norm.weight": "vision_model.post_layernorm.weight",
        "post_norm.bias": "vision_model.post_layernorm.bias",
    }
    if visual_key in global_map:
        full_name = "image_encoder." + global_map[visual_key]
        if full_name in params:
            while tensor.dim() > params[full_name].dim():
                tensor = tensor.squeeze(0)
            params[full_name].data.copy_(tensor)
            loaded.add(full_name)
        return

    m = __import__("re").match(r"transformer\.(\d+)\.(.*)", visual_key)
    if not m:
        return
    layer_idx, rest = m.group(1), m.group(2)
    layer_prefix = f"vision_model.encoder.layers.{layer_idx}"

    if rest in ("attn.to_qkv.weight", "attn.to_qkv.bias"):
        suffix = rest.split(".")[-1]
        chunks = tensor.chunk(3, dim=0)
        for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            full_name = f"image_encoder.{layer_prefix}.self_attn.{proj}.{suffix}"
            if full_name in params:
                params[full_name].data.copy_(chunks[i])
                loaded.add(full_name)
        return

    simple_map = {
        "attn.proj.weight": f"{layer_prefix}.self_attn.out_proj.weight",
        "attn.proj.bias": f"{layer_prefix}.self_attn.out_proj.bias",
        "mlp.0.weight": f"{layer_prefix}.mlp.fc1.weight",
        "mlp.0.bias": f"{layer_prefix}.mlp.fc1.bias",
        "mlp.2.weight": f"{layer_prefix}.mlp.fc2.weight",
        "mlp.2.bias": f"{layer_prefix}.mlp.fc2.bias",
        "norm1.weight": f"{layer_prefix}.layer_norm1.weight",
        "norm1.bias": f"{layer_prefix}.layer_norm1.bias",
        "norm2.weight": f"{layer_prefix}.layer_norm2.weight",
        "norm2.bias": f"{layer_prefix}.layer_norm2.bias",
    }
    if rest in simple_map:
        full_name = "image_encoder." + simple_map[rest]
        if full_name in params:
            params[full_name].data.copy_(tensor)
            loaded.add(full_name)


def _source_preprocess(image: torch.Tensor) -> torch.Tensor:
    """Exact source preprocessing for DreamZero `WanImageEncoder.encode_image()`."""
    size = (224, 224)
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    videos = torch.cat(
        [F.interpolate(frame_batch, size=size, mode="bicubic", align_corners=False) for frame_batch in image]
    )
    return normalize(videos.mul_(0.5).add_(0.5))


def _source_preprocess_frame(frame_chw_uint8: torch.Tensor) -> torch.Tensor:
    pixels = frame_chw_uint8.unsqueeze(0).float() / 255.0
    pixels = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _max_mean_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def test_local_image_encoder_matches_upstream_on_real_service_input():
    image, _ = _load_real_input()
    upstream = _load_upstream_encoder()
    local_port = _load_local_port()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        upstream_out = upstream.encode_image(image.clone())
        local_out = local_port.encode_image(image.clone())

    local_max, local_mean = _max_mean_diff(local_out, upstream_out)
    assert local_max == 0.0 and local_mean == 0.0


def test_hf_clip_vision_model_drifts_on_real_service_input():
    image, _ = _load_real_input()
    upstream = _load_upstream_encoder()
    hf = _load_hf_clip_model()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        upstream_out = upstream.encode_image(image.clone())
        hf_pixels = _source_preprocess(image).to(device=image.device, dtype=torch.bfloat16)
        hf_out = hf(hf_pixels, output_hidden_states=True).hidden_states[-2].clone()

    hf_max, hf_mean = _max_mean_diff(hf_out, upstream_out)
    assert hf_max > 0.5
    assert hf_mean > 1e-3


def test_clip_image_processor_is_not_source_equivalent_for_dreamzero():
    image, frame_chw_uint8 = _load_real_input()
    hf = _load_hf_clip_model()

    source_pixels = _source_preprocess_frame(frame_chw_uint8)

    processor_default = CLIPImageProcessor()
    default_pixels = processor_default(
        images=Image.fromarray(frame_chw_uint8.permute(1, 2, 0).cpu().numpy()),
        return_tensors="pt",
    ).pixel_values

    processor_match_cfg = CLIPImageProcessor(
        do_center_crop=False,
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=3,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    match_pixels = processor_match_cfg(images=frame_chw_uint8, return_tensors="pt").pixel_values

    default_pixels_max, _ = _max_mean_diff(default_pixels, source_pixels)
    match_pixels_max, _ = _max_mean_diff(match_pixels, source_pixels)

    with torch.no_grad():
        base_out = (
            hf(
                source_pixels.to(device=image.device, dtype=torch.bfloat16),
                output_hidden_states=True,
            )
            .hidden_states[-2]
            .clone()
        )
        default_out = (
            hf(
                default_pixels.to(device=image.device, dtype=torch.bfloat16),
                output_hidden_states=True,
            )
            .hidden_states[-2]
            .clone()
        )
        match_out = (
            hf(
                match_pixels.to(device=image.device, dtype=torch.bfloat16),
                output_hidden_states=True,
            )
            .hidden_states[-2]
            .clone()
        )

    default_out_max, _ = _max_mean_diff(default_out, base_out)
    match_out_max, _ = _max_mean_diff(match_out, base_out)

    assert default_pixels_max > 1.0
    assert match_pixels_max > 1e-2
    assert default_out_max > 1.0
    assert match_out_max > 1.0
