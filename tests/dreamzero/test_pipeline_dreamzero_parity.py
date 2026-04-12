# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pipeline-level DreamZero parity tests against upstream eager source.

This test exercises the full vLLM-Omni DreamZero pipeline path:

- `DroidTransform.transform_input()`
- `DreamZeroPipeline.forward()`
- `DreamZeroState` frame accumulation / KV cache reuse
- CFG sequential / CFG parallel execution
- TP=1 / TP=2 execution

Reference path:

- Upstream `WANPolicyHead.lazy_joint_video_action()`
- Upstream `CausalWanModel`
- Upstream scheduler imported with `torch.compile` disabled
- Tiny fake tokenizer / text encoder / image encoder / VAE so the test stays
  lightweight while preserving the DreamZero control flow

Run:
    PYTHONPATH=. .venv/bin/python -m pytest \
        tests/dreamzero/test_pipeline_dreamzero_parity.py -v -s

Or run directly:
    PYTHONPATH=. .venv/bin/python tests/dreamzero/test_pipeline_dreamzero_parity.py
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import os
import socket
import sys
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.openai.realtime.robot.transform.droid import DroidTransform
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

DTYPE = torch.bfloat16
ATOL = 1e-5
RTOL = 1e-5
# End-to-end observable outputs must match bit-for-bit after bf16→fp32 cast.
OUTPUT_ATOL = 0.0
OUTPUT_RTOL = 0.0
# Internal DreamZero state must also match exactly.
STATE_CACHE_ATOL = 0.0
STATE_CACHE_RTOL = 0.0

NUM_INFERENCE_STEPS = 2
NUM_FRAMES = 5
NUM_FRAME_PER_BLOCK = 1
ACTION_HORIZON = 4
NEGATIVE_PROMPT = "bad quality"
CLIP_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
CLIP_IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)

TINY_CFG = dict(
    model_type="i2v",
    patch_size=(1, 2, 2),
    frame_seqlen=4,
    text_len=16,
    in_dim=36,
    dim=64,
    ffn_dim=128,
    freq_dim=32,
    text_dim=64,
    out_dim=16,
    num_heads=4,
    num_layers=2,
    qk_norm=True,
    cross_attn_norm=True,
    num_frame_per_block=1,
    action_dim=8,
    num_action_per_block=4,
    num_state_per_block=1,
    max_num_embodiments=4,
    hidden_size=32,
)

ACTION_STATS = {
    "oxe_droid": {
        "q01": torch.tensor([-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10], dtype=torch.float32),
        "q99": torch.tensor([0.35, 0.42, 0.49, 0.56, 0.63, 0.70, 0.77, 0.84], dtype=torch.float32),
    }
}

COMBOS = [
    (1, 1),
    pytest.param(
        2,
        1,
        marks=pytest.mark.xfail(
            reason="Known RowParallelLinear bf16 drift under TP>1; see docs/models/dreamzero/todo.md appendix.",
            strict=False,
        ),
    ),
    (1, 2),
    pytest.param(
        2,
        2,
        marks=pytest.mark.xfail(
            reason="Known RowParallelLinear bf16 drift under TP>1; see docs/models/dreamzero/todo.md appendix.",
            strict=False,
        ),
    ),
]

COMBO_IDS = [
    "tp1_cfgp1",
    "tp2_cfgp1",
    "tp1_cfgp2",
    "tp2_cfgp2",
]


def _install_optional_dependency_stubs() -> None:
    """Stub optional DreamZero dependencies missing in `.venv`.

    The parity test only calls inference helpers, so lightweight stubs for
    `hydra` and `peft` are sufficient to import upstream `WANPolicyHead`.
    """

    if "hydra" not in sys.modules:
        hydra = type(sys)("hydra")
        hydra.__spec__ = importlib.machinery.ModuleSpec("hydra", loader=None)
        hydra_utils = type(sys)("hydra.utils")
        hydra_utils.__spec__ = importlib.machinery.ModuleSpec("hydra.utils", loader=None)
        hydra_utils.instantiate = lambda *args, **kwargs: None
        hydra.utils = hydra_utils
        sys.modules["hydra"] = hydra
        sys.modules["hydra.utils"] = hydra_utils

    if "peft" not in sys.modules:
        peft = type(sys)("peft")
        peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

        class LoraConfig:  # noqa: D401 - tiny import stub
            pass

        peft.LoraConfig = LoraConfig
        peft.get_peft_model = lambda model, *args, **kwargs: model
        sys.modules["peft"] = peft


@contextlib.contextmanager
def _patch_dreamzero_compile_to_eager():
    """Import DreamZero reference modules with `torch.compile` disabled.

    For this parity test we intentionally target the upstream eager scheduler
    path, because the current vLLM scheduler implementation is also eager.
    """

    compile_orig = torch.compile
    module_names = [
        "groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler",
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    def identity_compile(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    torch.compile = identity_compile
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        yield
    finally:
        torch.compile = compile_orig
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


@contextlib.contextmanager
def _patch_dreamzero_attention():
    """Patch upstream attention to SDPA, same strategy as model parity tests."""

    import groot.vla.model.dreamzero.modules.attention as base_attention
    import groot.vla.model.dreamzero.modules.wan2_1_attention as wan_attention
    import groot.vla.model.dreamzero.modules.wan2_1_submodule as wan_submodule
    import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as wan_chunk

    old_backend = os.environ.get("ATTENTION_BACKEND")
    os.environ["ATTENTION_BACKEND"] = "torch"

    original_base_flash = base_attention.flash_attention
    original_wan_attention_flash = wan_attention.flash_attention
    original_wan_submodule_flash = wan_submodule.flash_attention
    original_attention_init = wan_attention.AttentionModule.__init__
    original_t2v_forward = wan_submodule.WanT2VCrossAttention.forward

    def sdpa_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens=None,
        k_lens=None,
        dropout_p: float = 0.0,
        softmax_scale=None,
        q_scale=None,
        causal: bool = False,
        window_size=(-1, -1),
        deterministic: bool = False,
        dtype: torch.dtype = torch.float32,
        version=None,
    ) -> torch.Tensor:
        del window_size, deterministic, dtype, version
        assert q_lens is None and k_lens is None, "varlen attention is not covered in this test"
        out_dtype = q.dtype
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.transpose(1, 2).to(torch.bfloat16)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def patched_attention_init(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        softmax_scale=None,
        q_scale=None,
        causal: bool = False,
        window_size=None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: str | None = None,
    ) -> None:
        del dtype, backend
        original_attention_init(
            self,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=torch.bfloat16,
            backend="torch",
        )

    def patched_t2v_forward(self, x, context, context_lens=None, crossattn_cache=None):
        return original_t2v_forward(self, x, context, context_lens, crossattn_cache)

    base_attention.flash_attention = sdpa_flash_attention
    wan_attention.flash_attention = sdpa_flash_attention
    wan_submodule.flash_attention = sdpa_flash_attention
    wan_attention.AttentionModule.__init__ = patched_attention_init
    wan_submodule.WanT2VCrossAttention.forward = patched_t2v_forward
    wan_chunk.WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"].forward = patched_t2v_forward

    try:
        yield
    finally:
        base_attention.flash_attention = original_base_flash
        wan_attention.flash_attention = original_wan_attention_flash
        wan_submodule.flash_attention = original_wan_submodule_flash
        wan_attention.AttentionModule.__init__ = original_attention_init
        wan_submodule.WanT2VCrossAttention.forward = original_t2v_forward
        wan_chunk.WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"].forward = original_t2v_forward
        if old_backend is None:
            os.environ.pop("ATTENTION_BACKEND", None)
        else:
            os.environ["ATTENTION_BACKEND"] = old_backend


class FakeTokenizer:
    def __init__(self, max_length: int = 16) -> None:
        self.max_length = max_length

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        masks = []
        for text in texts:
            toks = [((ord(ch) % 97) + 1) for ch in text[: self.max_length]]
            mask = [1] * len(toks)
            toks += [0] * (self.max_length - len(toks))
            mask += [0] * (self.max_length - len(mask))
            ids.append(toks)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeTextEncoderVLLM(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        ids = input_ids.float().unsqueeze(-1)
        basis = torch.arange(self.dim, device=input_ids.device, dtype=torch.float32).view(1, 1, -1)
        hidden = torch.sin(ids * (basis + 1) / 17.0) + torch.cos(ids * (basis + 1) / 29.0)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTextEncoderDreamZero(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        ids = input_ids.float().unsqueeze(-1)
        basis = torch.arange(self.dim, device=input_ids.device, dtype=torch.float32).view(1, 1, -1)
        return torch.sin(ids * (basis + 1) / 17.0) + torch.cos(ids * (basis + 1) / 29.0)


class FakeImageProcessor:
    def __call__(self, images, return_tensors="pt", do_rescale=False):
        del return_tensors, do_rescale
        return SimpleNamespace(pixel_values=images)


def _fake_clip_hidden_from_pixel_values(pixel_values: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    batch_size, channels, _, _ = pixel_values.shape
    down = F.interpolate(pixel_values.float(), size=(16, 16), mode="bilinear", align_corners=False)
    tokens = down.permute(0, 2, 3, 1).reshape(batch_size, 256, channels)
    mean = tokens.mean(dim=-1, keepdim=True)
    basis = torch.arange(1280, device=down.device, dtype=torch.float32).view(1, 1, -1)
    feats = torch.sin(mean * (basis + 1) / 37.0)
    cls = feats.mean(dim=1, keepdim=True)
    return torch.cat([cls, feats], dim=1).to(dtype=dtype)


class FakeClipVision(nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self._dtype = dtype
        self.config = SimpleNamespace(image_size=224)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True):
        del output_hidden_states
        hidden = _fake_clip_hidden_from_pixel_values(pixel_values, dtype=self._dtype)
        return SimpleNamespace(hidden_states=[torch.zeros_like(hidden), hidden, torch.zeros_like(hidden)])


class FakeDreamZeroImageEncoder(nn.Module):
    def __init__(self, *, dtype: torch.dtype = DTYPE) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy.dtype

    def encode_image(self, videos: torch.Tensor) -> torch.Tensor:
        size = (224, 224)
        pixel_values = torch.cat([
            F.interpolate(frame_batch.float(), size=size, mode="bicubic", align_corners=False)
            for frame_batch in videos
        ])
        pixel_values = pixel_values.mul(0.5).add(0.5)
        pixel_values = (
            pixel_values
            - CLIP_IMAGE_MEAN.to(device=pixel_values.device, dtype=pixel_values.dtype)
        ) / CLIP_IMAGE_STD.to(device=pixel_values.device, dtype=pixel_values.dtype)
        pixel_values = pixel_values.to(dtype=self.dtype)
        return _fake_clip_hidden_from_pixel_values(pixel_values, dtype=self.dtype)


class FakeVAE(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy.dtype

    def _latent_mu(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, _, _ = video.shape
        frame_indices = [0] + list(range(4, num_frames, 4))
        selected = video[:, :, frame_indices]
        pooled = F.avg_pool3d(selected.float(), kernel_size=(1, 8, 8), stride=(1, 8, 8))
        repeated = pooled.repeat(1, (16 + channels - 1) // channels, 1, 1, 1)[:, :16]
        return repeated.to(dtype=self.dtype)

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        mu = self._latent_mu(video)
        logvar = torch.zeros_like(mu)
        return torch.cat([mu, logvar], dim=1)

    def encode(self, video: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return self._latent_mu(video)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class RefFrameAccumulator:
    """Mirror DreamZeroState.accumulate_frames() for the upstream reference."""

    def __init__(self) -> None:
        self.buf: list[np.ndarray] = []
        self.calls = 0

    def add(self, stitched: np.ndarray) -> np.ndarray:
        if stitched.ndim == 4:
            self.buf.extend(list(stitched))
        else:
            self.buf.append(stitched)
        need = 1 if self.calls == 0 else 4
        frames = self.buf[-need:] if len(self.buf) >= need else list(self.buf)
        while len(frames) < need:
            frames.insert(0, self.buf[0])
        self.calls += 1
        return np.stack(frames, axis=0)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _device(local_rank: int) -> torch.device:
    return torch.device(f"cuda:{local_rank}")


def _set_deterministic() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _init_parallel_env(rank: int, world_size: int, tp_size: int, cfg_parallel_size: int, master_port: int) -> torch.device:
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.platforms import current_omni_platform

    device = _device(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    torch.cuda.set_device(device)
    current_omni_platform.set_device(device)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_parallel_size=tp_size,
        cfg_parallel_size=cfg_parallel_size,
    )
    return device


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_error: torch.Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    if actual_f.shape != expected_f.shape:
        raise AssertionError(f"{name}: shape mismatch actual={tuple(actual_f.shape)}, expected={tuple(expected_f.shape)}")
    diff = 0.0 if actual_f.numel() == 0 else (actual_f - expected_f).abs().max().item()
    max_error.fill_(max(max_error.item(), diff))
    if not torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max_diff={diff:.3e}, atol={atol}, rtol={rtol}")


def _load_vllm_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight.to(device=param.device, dtype=param.dtype))


def _broadcast_module(module: torch.nn.Module) -> None:
    if not torch.distributed.is_initialized():
        return
    for param in module.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for buffer in module.buffers():
        torch.distributed.broadcast(buffer.data, src=0)


def _sync_module(vllm_module: torch.nn.Module, dreamzero_module: torch.nn.Module) -> None:
    _broadcast_module(dreamzero_module)
    vllm_params = dict(vllm_module.named_parameters())
    dreamzero_params_raw = dict(dreamzero_module.named_parameters())

    name_map = {
        "img_emb.proj.0.weight": "img_emb.norm1.weight",
        "img_emb.proj.0.bias": "img_emb.norm1.bias",
        "img_emb.proj.1.weight": "img_emb.fc1.weight",
        "img_emb.proj.1.bias": "img_emb.fc1.bias",
        "img_emb.proj.3.weight": "img_emb.fc2.weight",
        "img_emb.proj.3.bias": "img_emb.fc2.bias",
        "img_emb.proj.4.weight": "img_emb.norm2.weight",
        "img_emb.proj.4.bias": "img_emb.norm2.bias",
    }
    dreamzero_params = {
        name_map.get(name, name): param
        for name, param in dreamzero_params_raw.items()
    }

    missing = sorted(set(dreamzero_params) - set(vllm_params))
    extra = sorted(set(vllm_params) - set(dreamzero_params))
    assert not missing, f"Missing params in vllm module: {missing}"
    assert not extra, f"Unexpected params in vllm module: {extra}"

    for name, dz_param in dreamzero_params.items():
        _load_vllm_param(vllm_params[name], dz_param.detach())


def _slice_dim(tensor: torch.Tensor, rank: int, world_size: int, dim: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dim %= tensor.ndim
    assert tensor.shape[dim] % world_size == 0, (
        f"Cannot shard dim={dim} with size={tensor.shape[dim]} across world_size={world_size}"
    )
    shard = tensor.shape[dim] // world_size
    return tensor.narrow(dim, rank * shard, shard).contiguous()


def _slice_heads(tensor: torch.Tensor, rank: int, world_size: int, *, head_dim: int) -> torch.Tensor:
    return _slice_dim(tensor, rank, world_size, head_dim)


def _build_vllm_pipeline(transformer: nn.Module, device: torch.device) -> DreamZeroPipeline:
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    nn.Module.__init__(pipe)

    pipe.tokenizer = FakeTokenizer()
    pipe.text_encoder = FakeTextEncoderVLLM(64).to(device)
    pipe.image_processor = FakeImageProcessor()
    pipe.image_encoder = FakeDreamZeroImageEncoder(dtype=DTYPE).to(device)
    pipe.register_buffer(
        "clip_image_mean",
        torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
        persistent=False,
    )
    pipe.register_buffer(
        "clip_image_std",
        torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
        persistent=False,
    )
    pipe.vae = FakeVAE().to(device)
    pipe.register_buffer(
        "vae_latents_mean",
        torch.zeros(1, 16, 1, 1, 1, dtype=torch.float32, device=device),
        persistent=False,
    )
    pipe.register_buffer(
        "vae_latents_inv_std",
        torch.ones(1, 16, 1, 1, 1, dtype=torch.float32, device=device),
        persistent=False,
    )
    pipe.transformer = transformer
    pipe.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    pipe.state = DreamZeroState()
    pipe.num_inference_steps = NUM_INFERENCE_STEPS
    pipe.cfg_scale = 5.0
    pipe.sigma_shift = 5.0
    pipe.num_frames = NUM_FRAMES
    pipe.num_frame_per_block = NUM_FRAME_PER_BLOCK
    pipe.action_horizon = ACTION_HORIZON
    pipe.decouple_inference_noise = False
    pipe.video_inference_final_noise = 0.8
    pipe.seed = 1140
    pipe.max_state_dim = 64
    pipe.max_action_dim = 8
    pipe.negative_prompt = NEGATIVE_PROMPT
    pipe.embodiment_name_to_id = {"oxe_droid": 0}
    pipe.action_norm_stats = {
        name: {key: value.clone() for key, value in stats.items()}
        for name, stats in ACTION_STATS.items()
    }
    pipe.relative_action = True
    pipe.relative_action_dim = 7
    return pipe


def _build_reference_head(source_model: nn.Module, device: torch.device):
    _install_optional_dependency_stubs()
    with _patch_dreamzero_compile_to_eager():
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead

    head = WANPolicyHead.__new__(WANPolicyHead)
    nn.Module.__init__(head)

    head.tiled = False
    head.tile_size_height = 34
    head.tile_size_width = 34
    head.tile_stride_height = 18
    head.tile_stride_width = 16
    head.num_frame_per_block = NUM_FRAME_PER_BLOCK
    head.hidden_size = TINY_CFG["hidden_size"]
    head.num_frames = NUM_FRAMES
    head.text_encoder = FakeTextEncoderDreamZero(64).to(device)
    head.image_encoder = FakeDreamZeroImageEncoder().to(device)
    head.vae = FakeVAE().to(device)
    head.scheduler = SimpleNamespace(num_train_timesteps=1000)
    head.num_inference_steps = NUM_INFERENCE_STEPS
    head.seed = 1140
    head.cfg_scale = 5.0
    head.sigma_shift = 5.0
    head.kv_cache1 = None
    head.kv_cache_neg = None
    head.crossattn_cache = None
    head.crossattn_cache_neg = None
    head.clip_feas = None
    head.ys = None
    head.current_start_frame = 0
    head.language = None
    head.ip_rank = 0
    head.ip_size = 1
    head.ip_group = None
    head._device = str(device)
    head.dynamic_cache_schedule = False
    head.dit_step_mask = [True] * NUM_INFERENCE_STEPS
    head.skip_countdown = 0
    head.normalize_video = lambda x: x * 2.0 - 1.0
    head.model = source_model
    head.action_dim = TINY_CFG["action_dim"]
    head.action_horizon = ACTION_HORIZON
    head.config = SimpleNamespace(
        decouple_inference_noise=False,
        video_inference_final_noise=0.8,
    )
    head.trt_engine = None
    head._vae_device_ready = False
    head.set_frozen_modules_to_eval_mode = lambda: None

    def create_kv(self, batch_size: int, dtype: torch.dtype, device: torch.device, frame_seqlen: int):
        del frame_seqlen
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_pos = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(self.model.num_layers)
        ]
        kv_neg = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(self.model.num_layers)
        ]
        return kv_pos, kv_neg

    def create_cross(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        del batch_size, dtype, device
        cross_pos = [{"is_init": False, "k": None, "v": None} for _ in range(self.model.num_layers)]
        cross_neg = [{"is_init": False, "k": None, "v": None} for _ in range(self.model.num_layers)]
        return cross_pos, cross_neg

    head._create_kv_caches = MethodType(create_kv, head)
    head._create_crossattn_caches = MethodType(create_cross, head)
    return head


def _make_request(unified_obs: dict[str, Any], *, reset: bool) -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={
            "reset": reset,
            "unified_obs": unified_obs,
        }
    )
    return OmniDiffusionRequest(
        prompts=[unified_obs["prompt"]],
        sampling_params=sampling_params,
        request_ids=["dreamzero-parity"],
    )


def _pad_state(raw_state: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = np.asarray(raw_state, dtype=np.float64).flatten()
    padded = np.zeros(64, dtype=np.float64)
    padded[: min(len(raw), 64)] = raw[:64]
    return torch.from_numpy(padded).reshape(1, 1, 64).to(device=device, dtype=torch.bfloat16)


def _make_raw_obs_sequence() -> list[dict[str, Any]]:
    obs_seq = []
    prompt = "pick up the block"
    for step in range(2):
        base = 20 + step * 11
        obs_seq.append(
            {
                "observation/exterior_image_1_left": np.full((16, 16, 3), base, dtype=np.uint8),
                "observation/exterior_image_2_left": np.full((16, 16, 3), base + 1, dtype=np.uint8),
                "observation/wrist_image_left": np.full((16, 16, 3), base + 2, dtype=np.uint8),
                "observation/joint_position": np.linspace(-0.3, 0.3, 7, dtype=np.float64) + step * 0.01,
                "observation/gripper_position": np.array([0.2 + step * 0.05], dtype=np.float64),
                "prompt": prompt,
            }
        )
    return obs_seq


def _build_reference_batch(
    tokenizer: FakeTokenizer,
    ref_frames: RefFrameAccumulator,
    unified_obs: dict[str, Any],
    device: torch.device,
) -> AttrDict:
    prompt_tokens = tokenizer(unified_obs["prompt"])
    negative_tokens = tokenizer(NEGATIVE_PROMPT)
    return AttrDict(
        images=torch.from_numpy(ref_frames.add(unified_obs["images"])).unsqueeze(0).to(device=device),
        text=prompt_tokens["input_ids"].to(device=device),
        text_attention_mask=prompt_tokens["attention_mask"].to(device=device),
        text_negative=negative_tokens["input_ids"].to(device=device),
        text_attention_mask_negative=negative_tokens["attention_mask"].to(device=device),
        state=_pad_state(unified_obs["state"], device),
        embodiment_id=torch.tensor([0], device=device, dtype=torch.long),
    )


def _denormalize_reference_action(action: torch.Tensor, embodiment_name: str) -> torch.Tensor:
    q01 = ACTION_STATS[embodiment_name]["q01"].to(device=action.device, dtype=action.dtype)
    q99 = ACTION_STATS[embodiment_name]["q99"].to(device=action.device, dtype=action.dtype)
    action_real = action.clone()
    action_real[..., : q01.shape[0]] = (action[..., : q01.shape[0]] + 1) / 2 * (q99 - q01) + q01
    return action_real


def _postprocess_reference(
    head_out: Any,
    raw_state: np.ndarray | torch.Tensor,
    embodiment_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror upstream `sim_policy.py` postprocess exactly.

    Source:
    - `sim_policy.py:814-820` casts `model_pred["action_pred"]` to float before
      calling `unapply()`
    - `sim_policy.py:539-604` denormalizes and then adds the original raw state
      back for the relative joint dimensions
    """
    action = _denormalize_reference_action(head_out.action_pred.float().clone(), embodiment_name)
    if not torch.is_tensor(raw_state):
        raw_state = torch.from_numpy(np.asarray(raw_state, dtype=np.float64))
    raw_state = raw_state.to(device=action.device, dtype=torch.float32).reshape(-1)
    action[..., :7] = action[..., :7] + raw_state[:7].view(1, 1, 7)
    return action.squeeze(0), head_out.video_pred


def _assert_cache_inactive(
    name: str,
    kv_cache: list[torch.Tensor],
    crossattn_cache: list[dict[str, object]],
) -> None:
    for layer_idx, cache in enumerate(kv_cache):
        if cache.shape[2] != 0:
            raise AssertionError(f"{name}.kv[{layer_idx}] should stay empty, got seq={cache.shape[2]}")
    for layer_idx, cache in enumerate(crossattn_cache):
        if cache["is_init"] or cache["k"] is not None or cache["v"] is not None:
            raise AssertionError(f"{name}.crossattn[{layer_idx}] should stay uninitialized")


def _assert_cache_matches(
    name: str,
    actual_kv: list[torch.Tensor],
    expected_kv: list[torch.Tensor],
    actual_crossattn: list[dict[str, object]],
    expected_crossattn: list[dict[str, object]],
    *,
    tp_rank: int,
    tp_size: int,
    max_error: torch.Tensor,
) -> None:
    for layer_idx, (actual, expected) in enumerate(zip(actual_kv, expected_kv, strict=True)):
        _assert_close(
            f"{name}.kv[{layer_idx}]",
            actual,
            _slice_heads(expected, tp_rank, tp_size, head_dim=3).to(device=actual.device, dtype=actual.dtype),
            max_error,
            atol=STATE_CACHE_ATOL,
            rtol=STATE_CACHE_RTOL,
        )

    for layer_idx, (actual, expected) in enumerate(zip(actual_crossattn, expected_crossattn, strict=True)):
        if actual["is_init"] != expected["is_init"]:
            raise AssertionError(
                f"{name}.crossattn[{layer_idx}].is_init mismatch: actual={actual['is_init']}, expected={expected['is_init']}"
            )
        if not actual["is_init"]:
            continue
        assert isinstance(actual["k"], torch.Tensor)
        assert isinstance(actual["v"], torch.Tensor)
        assert isinstance(expected["k"], torch.Tensor)
        assert isinstance(expected["v"], torch.Tensor)
        _assert_close(
            f"{name}.crossattn[{layer_idx}].k",
            actual["k"],
            _slice_heads(expected["k"], tp_rank, tp_size, head_dim=2).to(device=actual["k"].device, dtype=actual["k"].dtype),
            max_error,
            atol=STATE_CACHE_ATOL,
            rtol=STATE_CACHE_RTOL,
        )
        _assert_close(
            f"{name}.crossattn[{layer_idx}].v",
            actual["v"],
            _slice_heads(expected["v"], tp_rank, tp_size, head_dim=2).to(device=actual["v"].device, dtype=actual["v"].dtype),
            max_error,
            atol=STATE_CACHE_ATOL,
            rtol=STATE_CACHE_RTOL,
        )


def _assert_pipeline_state_matches_reference(
    pipe: DreamZeroPipeline,
    ref_head: Any,
    *,
    tp_rank: int,
    tp_size: int,
    cfg_parallel_size: int,
    max_error: torch.Tensor,
) -> None:
    from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_rank

    assert pipe.state.current_start_frame == ref_head.current_start_frame
    assert torch.equal(pipe.state.language, ref_head.language)
    assert pipe.state.clip_feas is not None and ref_head.clip_feas is not None
    assert pipe.state.ys is not None and ref_head.ys is not None
    _assert_close(
        "state.clip_feas",
        pipe.state.clip_feas,
        ref_head.clip_feas,
        max_error,
        atol=STATE_CACHE_ATOL,
        rtol=STATE_CACHE_RTOL,
    )
    _assert_close(
        "state.ys",
        pipe.state.ys,
        ref_head.ys,
        max_error,
        atol=STATE_CACHE_ATOL,
        rtol=STATE_CACHE_RTOL,
    )

    if cfg_parallel_size == 1:
        _assert_cache_matches(
            "state.positive",
            pipe.state.kv_cache,
            ref_head.kv_cache1,
            pipe.state.crossattn_cache,
            ref_head.crossattn_cache,
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_error=max_error,
        )
        _assert_cache_matches(
            "state.negative",
            pipe.state.kv_cache_neg,
            ref_head.kv_cache_neg,
            pipe.state.crossattn_cache_neg,
            ref_head.crossattn_cache_neg,
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_error=max_error,
        )
        return

    cfg_rank = get_classifier_free_guidance_rank()
    if cfg_rank == 0:
        _assert_cache_matches(
            "state.positive",
            pipe.state.kv_cache,
            ref_head.kv_cache1,
            pipe.state.crossattn_cache,
            ref_head.crossattn_cache,
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_error=max_error,
        )
        _assert_cache_inactive("state.negative", pipe.state.kv_cache_neg, pipe.state.crossattn_cache_neg)
    else:
        _assert_cache_matches(
            "state.negative",
            pipe.state.kv_cache_neg,
            ref_head.kv_cache_neg,
            pipe.state.crossattn_cache_neg,
            ref_head.crossattn_cache_neg,
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_error=max_error,
        )
        _assert_cache_inactive("state.positive", pipe.state.kv_cache, pipe.state.crossattn_cache)


def _run_combo_worker(rank: int, world_size: int, tp_size: int, cfg_parallel_size: int, master_port: int) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm_omni.diffusion.distributed.parallel_state import destroy_distributed_env

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        device = _init_parallel_env(rank, world_size, tp_size, cfg_parallel_size, master_port)
        _set_deterministic()
        _install_optional_dependency_stubs()

        try:
            with _patch_dreamzero_attention():
                from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
                    CausalWanModel as DreamZeroCausalWanModel,
                )
                from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
                    CausalWanModel as VLLMCausalWanModel,
                )

                source_model = DreamZeroCausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE).eval()
                vllm_model = VLLMCausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE).eval()
                # Source `lazy_joint_video_action()` resets on
                # `current_start_frame >= local_attn_size` even when upstream
                # uses `-1` as the "disabled" sentinel. Keep the model math in
                # full-attention mode (blocks still hold their original value),
                # but override the top-level reset check so the 2-step AR path
                # stays comparable.
                source_model.local_attn_size = 1024
                vllm_model.local_attn_size = 1024
                _sync_module(vllm_model, source_model)

                pipe = _build_vllm_pipeline(vllm_model, device)
                ref_head = _build_reference_head(source_model, device)
                transform = DroidTransform()
                ref_frames = RefFrameAccumulator()
                max_output_error = torch.zeros(1, device=device, dtype=torch.float32)
                max_state_error = torch.zeros(1, device=device, dtype=torch.float32)

                for step_idx, raw_obs in enumerate(_make_raw_obs_sequence()):
                    unified_obs = transform.transform_input(raw_obs)
                    output = pipe.forward(_make_request(unified_obs, reset=(step_idx == 0)))
                    ref_batch = _build_reference_batch(pipe.tokenizer, ref_frames, unified_obs, device)

                    with torch.no_grad():
                        reference = ref_head.lazy_joint_video_action(AttrDict(), ref_batch)

                    ref_action, ref_video = _postprocess_reference(
                        reference,
                        unified_obs["state"],
                        "oxe_droid",
                    )
                    actual_action = torch.from_numpy(output.output["actions"]).to(device=device, dtype=torch.float32)
                    actual_video = output.output["video"].to(device=device, dtype=torch.float32)

                    _assert_close(
                        f"step{step_idx}.actions",
                        actual_action,
                        ref_action.to(dtype=torch.float32),
                        max_output_error,
                        atol=OUTPUT_ATOL,
                        rtol=OUTPUT_RTOL,
                    )
                    _assert_close(
                        f"step{step_idx}.video",
                        actual_video,
                        ref_video.to(dtype=torch.float32),
                        max_output_error,
                        atol=OUTPUT_ATOL,
                        rtol=OUTPUT_RTOL,
                    )
                    _assert_pipeline_state_matches_reference(
                        pipe,
                        ref_head,
                        tp_rank=get_tensor_model_parallel_rank(),
                        tp_size=tp_size,
                        cfg_parallel_size=cfg_parallel_size,
                        max_error=max_state_error,
                    )

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(max_output_error, op=torch.distributed.ReduceOp.MAX)
                    torch.distributed.all_reduce(max_state_error, op=torch.distributed.ReduceOp.MAX)
                if rank == 0:
                    print(
                        f"DreamZero pipeline parity PASS: tp={tp_size}, cfg_p={cfg_parallel_size}, "
                        f"output_max={max_output_error.item():.3e}, "
                        f"state_max={max_state_error.item():.3e}"
                    )
        finally:
            destroy_distributed_env()
            torch.cuda.empty_cache()


def _run_combo(tp_size: int, cfg_parallel_size: int) -> None:
    world_size = tp_size * cfg_parallel_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs for tp={tp_size}, cfg_p={cfg_parallel_size}")
    master_port = _find_free_port()
    mp.spawn(
        _run_combo_worker,
        args=(world_size, tp_size, cfg_parallel_size, master_port),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize(
    ("tp_size", "cfg_parallel_size"),
    COMBOS,
    ids=COMBO_IDS,
)
def test_dreamzero_pipeline_matches_source(tp_size: int, cfg_parallel_size: int) -> None:
    _run_combo(tp_size, cfg_parallel_size)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")
    for tp_size, cfg_parallel_size in COMBOS:
        if torch.cuda.device_count() < tp_size * cfg_parallel_size:
            print(f"skip tp={tp_size}, cfg_p={cfg_parallel_size}: insufficient GPUs")
            continue
        _run_combo(tp_size, cfg_parallel_size)


if __name__ == "__main__":
    main()
