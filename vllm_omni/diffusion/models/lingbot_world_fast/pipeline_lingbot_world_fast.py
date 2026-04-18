# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from huggingface_hub import hf_hub_download
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .cam_utils import (
    compute_relative_poses,
    get_intrinsics_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from .runtime import LingbotWorldFastRuntimeConfig, LingbotWorldFastRuntimeState
from .state import normalize_lingbot_control_chunk

logger = init_logger(__name__)

_BASE_MODEL_REPO = "robbyant/lingbot-world-base-cam"
_TOKENIZER_REPO = "google/umt5-xxl"
_TIMESTEP_INDEX = (0, 179, 358, 679)


@dataclass
class _LingbotSourceModules:
    WanModelFast: Any
    T5EncoderModel: Any
    Wan2_1_VAE: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _add_third_party_lingbot_path() -> None:
    candidate = _repo_root() / "third_party" / "lingbot-world"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _load_source_modules() -> _LingbotSourceModules:
    _add_third_party_lingbot_path()
    try:
        from wan.modules.model_fast import WanModelFast
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE
    except ImportError as exc:  # pragma: no cover - runtime-only path
        raise ImportError(
            "Lingbot-world source is required at `third_party/lingbot-world` for LingbotWorldFastPipeline."
        ) from exc
    return _LingbotSourceModules(
        WanModelFast=WanModelFast,
        T5EncoderModel=T5EncoderModel,
        Wan2_1_VAE=Wan2_1_VAE,
    )


def _resolve_file(repo_or_path: str, relative_path: str) -> str:
    if os.path.isdir(repo_or_path):
        resolved = os.path.join(repo_or_path, relative_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Missing required file: {resolved}")
        return resolved
    return hf_hub_download(repo_or_path, relative_path)


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _count_causal_conv3d(module: nn.Module) -> int:
    return sum(1 for submodule in module.modules() if submodule.__class__.__name__ == "CausalConv3d")


class LingbotWorldFastPipeline(nn.Module, SupportImageInput):
    """Realtime causal Lingbot-World-Fast pipeline backed by source modules."""

    support_image_input = True
    color_format = "RGB"

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.base_model = od_config.model_paths.get("base_model", _BASE_MODEL_REPO)
        self.fast_model = od_config.model or ""
        self.source = _load_source_modules()

        text_encoder_ckpt = _resolve_file(self.base_model, "models_t5_umt5-xxl-enc-bf16.pth")
        vae_ckpt = _resolve_file(self.base_model, "Wan2.1_VAE.pth")

        self.text_encoder = self.source.T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=text_encoder_ckpt,
            tokenizer_path=_TOKENIZER_REPO,
            shard_fn=None,
        )

        self.vae = self.source.Wan2_1_VAE(
            vae_pth=vae_ckpt,
            dtype=self.dtype,
            device=self.device,
        )
        self.transformer = self.source.WanModelFast.from_pretrained(
            self.fast_model,
            torch_dtype=self.dtype,
            control_type="cam",
        ).eval().requires_grad_(False).to(self.device)

        self.num_train_timesteps = 1000
        self.text_len = 512
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.enc_conv_count = _count_causal_conv3d(self.vae.model.encoder)
        self.dec_conv_count = _count_causal_conv3d(self.vae.model.decoder)

        self.sessions: dict[str, LingbotWorldFastRuntimeState] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        del weights
        return None

    def reset_realtime_video_session(self, session_id: str) -> bool:
        removed = self.sessions.pop(session_id, None) is not None
        logger.info("Lingbot realtime runtime session reset: %s removed=%s", session_id, removed)
        return removed

    def _build_runtime_config(
        self,
        request: OmniDiffusionRequest,
        realtime_video: Mapping[str, Any],
    ) -> LingbotWorldFastRuntimeConfig:
        sampling = request.sampling_params
        return LingbotWorldFastRuntimeConfig(
            session_id=str(realtime_video["session_id"]),
            rendered_prompt=str(realtime_video["rendered_prompt"]),
            text_layers=dict(realtime_video.get("text_layers", {})),
            width=int(sampling.width or 832),
            height=int(sampling.height or 480),
            fps=int(sampling.fps or 16),
            chunk_size=int(realtime_video["chunk_size"]),
            seed=sampling.seed,
            shift=float(realtime_video.get("shift") or 5.0),
            max_attention_size=realtime_video.get("max_attention_size"),
        )

    def _extract_image(self, request: OmniDiffusionRequest) -> PIL.Image.Image:
        prompt = request.prompts[0]
        if isinstance(prompt, str):
            raise ValueError("LingbotWorldFastPipeline requires image multimodal input.")
        image = prompt.get("multi_modal_data", {}).get("image")
        if isinstance(image, str):
            return PIL.Image.open(image).convert("RGB")
        if isinstance(image, PIL.Image.Image):
            return image.convert("RGB")
        raise ValueError("LingbotWorldFastPipeline requires a single PIL image input.")

    def _encode_prompt(self, rendered_prompt: str) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder([rendered_prompt], self.device)[0].to(device=self.device, dtype=self.dtype)

    def _normalize_image_tensor(self, image: PIL.Image.Image, *, height: int, width: int) -> torch.Tensor:
        image = image.resize((width, height), PIL.Image.Resampling.BICUBIC)
        return TF.to_tensor(image).sub_(0.5).div_(0.5).to(device=self.device, dtype=torch.float32)

    def _encode_condition_first_latent(
        self,
        *,
        image_tensor: torch.Tensor,
        height: int,
        width: int,
        encoder_feat_cache: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_frame = image_tensor.unsqueeze(0).unsqueeze(2)
        first_frame = first_frame.to(device=self.device, dtype=self.vae.model.conv1.weight.dtype)
        feat_idx = [0]
        encoded = self.vae.model.encoder(first_frame, feat_cache=encoder_feat_cache, feat_idx=feat_idx)
        mu, _ = self.vae.model.conv1(encoded).chunk(2, dim=1)
        mean = self.vae.mean.view(1, -1, 1, 1, 1).to(device=mu.device, dtype=mu.dtype)
        inv_std = (1.0 / self.vae.std).view(1, -1, 1, 1, 1).to(device=mu.device, dtype=mu.dtype)
        mu = (mu - mean) * inv_std

        zero_condition_video = torch.zeros(
            1,
            3,
            self.vae_stride[0],
            height,
            width,
            device=self.device,
            dtype=self.vae.model.conv1.weight.dtype,
        )
        return mu.to(dtype=torch.float32), zero_condition_video

    def _make_scheduler(self, shift: float) -> FlowUniPCMultistepScheduler:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(self.num_train_timesteps, device=self.device, shift=shift)
        return scheduler

    def _make_generator(self, seed: int | None) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed if seed is not None else 0))
        return generator

    def _create_kv_cache(
        self,
        *,
        num_layers: int,
        capacity_tokens: int,
        num_heads: int,
        head_dim: int,
    ) -> list[dict[str, torch.Tensor]]:
        kv_cache: list[dict[str, torch.Tensor]] = []
        for _ in range(num_layers):
            kv_cache.append(
                {
                    "k": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=self.dtype, device=self.device),
                    "v": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=self.dtype, device=self.device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                }
            )
        return kv_cache

    def _create_crossattn_cache(
        self,
        *,
        num_layers: int,
        max_sequence_length: int,
        num_heads: int,
        head_dim: int,
    ) -> list[dict[str, Any]]:
        crossattn_cache: list[dict[str, Any]] = []
        for _ in range(num_layers):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        (1, max_sequence_length, num_heads, head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        (1, max_sequence_length, num_heads, head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "is_init": False,
                }
            )
        return crossattn_cache

    def _init_runtime_state(
        self,
        request: OmniDiffusionRequest,
        realtime_video: Mapping[str, Any],
        runtime_config: LingbotWorldFastRuntimeConfig,
    ) -> LingbotWorldFastRuntimeState:
        image = self._extract_image(request)
        image_tensor = self._normalize_image_tensor(
            image,
            height=runtime_config.height,
            width=runtime_config.width,
        )

        latent_height = runtime_config.height // self.vae_stride[1]
        latent_width = runtime_config.width // self.vae_stride[2]
        frame_seqlen = (latent_height * latent_width) // (self.patch_size[1] * self.patch_size[2])
        prompt_context = self._encode_prompt(runtime_config.rendered_prompt)
        encoder_feat_cache = [None] * self.enc_conv_count
        decoder_feat_cache = [None] * self.dec_conv_count
        condition_latents, zero_condition_video = self._encode_condition_first_latent(
            image_tensor=image_tensor,
            height=runtime_config.height,
            width=runtime_config.width,
            encoder_feat_cache=encoder_feat_cache,
        )

        model_args = self.transformer.config
        head_dim = model_args.dim // model_args.num_heads
        capacity_tokens = runtime_config.chunk_size * frame_seqlen
        kv_cache = self._create_kv_cache(
            num_layers=model_args.num_layers,
            capacity_tokens=capacity_tokens,
            num_heads=model_args.num_heads,
            head_dim=head_dim,
        )
        crossattn_cache = self._create_crossattn_cache(
            num_layers=model_args.num_layers,
            max_sequence_length=self.text_len,
            num_heads=model_args.num_heads,
            head_dim=head_dim,
        )

        return LingbotWorldFastRuntimeState(
            config=runtime_config,
            prompt_context=prompt_context,
            condition_latents=condition_latents,
            zero_condition_video=zero_condition_video,
            generator=self._make_generator(runtime_config.seed),
            scheduler=self._make_scheduler(runtime_config.shift),
            encoder_feat_cache=encoder_feat_cache,
            decoder_feat_cache=decoder_feat_cache,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            latent_height=latent_height,
            latent_width=latent_width,
            frame_seqlen=frame_seqlen,
            max_sequence_length=self.text_len,
            kv_capacity_tokens=capacity_tokens,
        )

    def _get_runtime_state(
        self,
        request: OmniDiffusionRequest,
        realtime_video: Mapping[str, Any],
    ) -> LingbotWorldFastRuntimeState:
        runtime_config = self._build_runtime_config(request, realtime_video)
        request_reset = bool(realtime_video.get("reset"))
        session = self.sessions.get(runtime_config.session_id)
        if session is None or session.config.signature != runtime_config.signature or request_reset:
            session = self._init_runtime_state(request, realtime_video, runtime_config)
            self.sessions[runtime_config.session_id] = session
        return session

    def _ensure_condition_latents(self, state: LingbotWorldFastRuntimeState, target_latent_frames: int) -> None:
        while state.condition_latents.shape[2] < target_latent_frames:
            feat_idx = [0]
            encoded = self.vae.model.encoder(
                state.zero_condition_video,
                feat_cache=state.encoder_feat_cache,
                feat_idx=feat_idx,
            )
            mu, _ = self.vae.model.conv1(encoded).chunk(2, dim=1)
            mean = self.vae.mean.view(1, -1, 1, 1, 1).to(device=mu.device, dtype=mu.dtype)
            inv_std = (1.0 / self.vae.std).view(1, -1, 1, 1, 1).to(device=mu.device, dtype=mu.dtype)
            mu = (mu - mean) * inv_std
            state.condition_latents = torch.cat([state.condition_latents, mu.to(dtype=torch.float32)], dim=2)

    def _ensure_kv_capacity(self, state: LingbotWorldFastRuntimeState, required_tokens: int) -> None:
        if required_tokens <= state.kv_capacity_tokens:
            return
        new_capacity = max(required_tokens, state.kv_capacity_tokens * 2)
        for cache in state.kv_cache:
            new_k = torch.zeros(
                (cache["k"].shape[0], new_capacity, cache["k"].shape[2], cache["k"].shape[3]),
                dtype=cache["k"].dtype,
                device=cache["k"].device,
            )
            new_v = torch.zeros_like(new_k)
            current_len = cache["k"].shape[1]
            new_k[:, :current_len] = cache["k"]
            new_v[:, :current_len] = cache["v"]
            cache["k"] = new_k
            cache["v"] = new_v
        state.kv_capacity_tokens = new_capacity

    def _prepare_intrinsics(self, intrinsics: np.ndarray, target_count: int) -> torch.Tensor:
        if intrinsics.shape[0] == 1:
            interpolated = np.repeat(intrinsics, target_count, axis=0)
        else:
            src = np.linspace(0, intrinsics.shape[0] - 1, intrinsics.shape[0], dtype=np.float32)
            tgt = np.linspace(0, intrinsics.shape[0] - 1, target_count, dtype=np.float32)
            interpolated = np.stack(
                [np.interp(tgt, src, intrinsics[:, column]) for column in range(intrinsics.shape[1])],
                axis=-1,
            )
        return torch.from_numpy(interpolated.astype(np.float32))

    def _prepare_control_chunks(
        self,
        state: LingbotWorldFastRuntimeState,
        control_payload: Mapping[str, Any],
    ) -> list[torch.Tensor]:
        chunk = normalize_lingbot_control_chunk(dict(control_payload))
        if chunk.control_type != "cam":
            raise NotImplementedError("LingbotWorldFastPipeline currently supports camera-control chunks only.")

        poses_np = chunk.poses.astype(np.float32)
        latent_frames = ((poses_np.shape[0] - 1) // self.vae_stride[0]) + 1
        latent_frames = int(latent_frames - (latent_frames % state.config.chunk_size))
        if latent_frames <= 0:
            raise ValueError(
                f"Control chunk must provide at least {state.config.chunk_size * self.vae_stride[0] + 1} frames."
            )

        src_indices = np.linspace(0, poses_np.shape[0] - 1, poses_np.shape[0], dtype=np.float32)
        tgt_indices = np.linspace(0, poses_np.shape[0] - 1, latent_frames, dtype=np.float32)
        c2ws_infer = interpolate_camera_poses(
            src_indices=src_indices,
            src_rot_mat=poses_np[:, :3, :3],
            src_trans_vec=poses_np[:, :3, 3],
            tgt_indices=tgt_indices,
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True).to(self.device)

        intrinsics = self._prepare_intrinsics(chunk.intrinsics, latent_frames)
        intrinsics = get_intrinsics_transformed(
            intrinsics,
            height_org=480,
            width_org=832,
            height_resize=state.config.height,
            width_resize=state.config.width,
            height_final=state.config.height,
            width_final=state.config.width,
        ).to(self.device)

        plucker = get_plucker_embeddings(
            c2ws_infer,
            intrinsics,
            state.config.height,
            state.config.width,
            only_rays_d=False,
        )
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(state.config.height // state.latent_height),
            c2=int(state.config.width // state.latent_width),
        )
        plucker = plucker[None, ...]
        plucker = rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=latent_frames,
            h=state.latent_height,
            w=state.latent_width,
        ).to(device=self.device, dtype=self.dtype)
        return list(plucker.split(state.config.chunk_size, dim=2))

    def _build_condition_chunk(
        self,
        state: LingbotWorldFastRuntimeState,
        *,
        start_frame: int,
    ) -> torch.Tensor:
        end_frame = start_frame + state.config.chunk_size
        self._ensure_condition_latents(state, end_frame)
        mask = torch.zeros(
            (1, 4, state.config.chunk_size, state.latent_height, state.latent_width),
            device=self.device,
            dtype=torch.float32,
        )
        if start_frame == 0:
            mask[:, :, 0:1] = 1.0
        condition_latents = state.condition_latents[:, :, start_frame:end_frame]
        return torch.cat([mask, condition_latents], dim=1)[0]

    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda tensor: tensor.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _decode_latent_chunk(self, state: LingbotWorldFastRuntimeState, latent_chunk: torch.Tensor) -> np.ndarray:
        z = latent_chunk.to(device=self.device, dtype=self.vae.model.conv2.weight.dtype)
        mean = self.vae.mean.view(1, -1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        inv_std = (1.0 / self.vae.std).view(1, -1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        z = z.unsqueeze(0) / inv_std + mean

        with torch.no_grad(), _autocast_context(self.device, self.dtype):
            features = self.vae.model.conv2(z)
            outputs: list[torch.Tensor] = []
            for frame_index in range(features.shape[2]):
                feat_idx = [0]
                outputs.append(
                    self.vae.model.decoder(
                        features[:, :, frame_index : frame_index + 1],
                        feat_cache=state.decoder_feat_cache,
                        feat_idx=feat_idx,
                    )
                )
            decoded = torch.cat(outputs, dim=2).float().clamp_(-1, 1)
        return decoded[0].permute(1, 2, 3, 0).detach().cpu().numpy()

    def _generate_one_chunk(
        self,
        state: LingbotWorldFastRuntimeState,
        *,
        control_chunk: torch.Tensor,
    ) -> np.ndarray:
        required_tokens = state.current_start + state.config.chunk_size * state.frame_seqlen
        self._ensure_kv_capacity(state, required_tokens)

        current_latent = torch.randn(
            (self.transformer.config.out_dim, state.config.chunk_size, state.latent_height, state.latent_width),
            dtype=torch.float32,
            generator=state.generator,
            device=self.device,
        )
        condition = self._build_condition_chunk(state, start_frame=state.generated_latent_frames)
        timesteps = state.scheduler.timesteps[list(_TIMESTEP_INDEX)]
        max_attention_size = state.config.max_attention_size or state.kv_capacity_tokens

        kwargs = {
            "context": [state.prompt_context],
            "seq_len": state.config.chunk_size * state.frame_seqlen,
            "y": [condition],
            "dit_cond_dict": {"c2ws_plucker_emb": [control_chunk]},
            "kv_cache": state.kv_cache,
            "crossattn_cache": state.crossattn_cache,
            "current_start": state.current_start,
            "max_attention_size": max_attention_size,
        }

        with torch.no_grad(), _autocast_context(self.device, self.dtype):
            for timestep_index, timestep_value in enumerate(timesteps):
                timestep = timestep_value.view(1).to(self.device)
                noise_pred = self.transformer(
                    x=[current_latent.to(dtype=self.dtype)],
                    t=timestep,
                    **kwargs,
                )[0]
                x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, timestep_value, state.scheduler)
                if timestep_index < len(timesteps) - 1:
                    next_timestep = timesteps[timestep_index + 1].view(1)
                    current_latent = state.scheduler.add_noise(
                        x0,
                        torch.randn(x0.shape, generator=state.generator, device=x0.device, dtype=x0.dtype),
                        next_timestep,
                    )

            zero_timestep = torch.zeros_like(timesteps[-1]).view(1).to(self.device)
            self.transformer(
                x=[x0.to(dtype=self.dtype)],
                t=zero_timestep,
                **kwargs,
            )

        state.generated_latent_frames += state.config.chunk_size
        state.generated_chunks += 1
        state.current_start += state.config.chunk_size * state.frame_seqlen
        return self._decode_latent_chunk(state, x0)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        del kwargs
        if len(req.prompts) != 1:
            raise ValueError("LingbotWorldFastPipeline only supports a single realtime request at a time.")

        realtime_video = req.sampling_params.extra_args.get("realtime_video")
        if not isinstance(realtime_video, Mapping):
            raise ValueError("LingbotWorldFastPipeline requires sampling_params.extra_args['realtime_video'].")
        state = self._get_runtime_state(req, realtime_video)
        all_video_chunks: list[np.ndarray] = []
        for control_payload in realtime_video.get("control", []):
            for control_chunk in self._prepare_control_chunks(state, control_payload):
                all_video_chunks.append(self._generate_one_chunk(state, control_chunk=control_chunk))

        if not all_video_chunks:
            raise ValueError("LingbotWorldFastPipeline received no control chunks to generate.")

        video_chunk = np.concatenate(all_video_chunks, axis=0)
        return DiffusionOutput(
            output=video_chunk,
            custom_output={
                "video_chunk": video_chunk,
                "realtime_video": {
                    "session_id": state.config.session_id,
                    "generated_chunks": state.generated_chunks,
                    "generated_latent_frames": state.generated_latent_frames,
                    "text_layers": state.config.text_layers,
                    "rendered_prompt": state.config.rendered_prompt,
                },
            },
        )
