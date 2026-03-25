# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
DreamZero world model pipeline for vllm-omni.

DreamZero is a joint video-action diffusion model (Wan2.1-I2V-14B based)
that predicts both video frames AND robot actions simultaneously.
CFG is applied to video only; actions use the positive (conditioned) branch.

Adapted from: third_party/dreamzero/groot/vla/model/dreamzero/action_head/
Reference PR: #2160 (LTX2 multi-output CFG parallel adaptation pattern)
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import UMT5EncoderModel
from vllm_omni.diffusion.request import OmniDiffusionRequest


class VideoActionScheduler:
    """Composite scheduler dispatching to video and action schedulers.

    Following the pattern from PR #2160 (LTX2 VideoAudioScheduler).
    Each scheduler tracks its own step index internally.
    """

    def __init__(self, video_scheduler, action_scheduler):
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(noise_pred[0], t[0], latents[0], return_dict=False, generator=generator)[
            0
        ]
        action_out = self.action_scheduler.step(
            noise_pred[1], t[1], latents[1], return_dict=False, generator=generator
        )[0]
        return ((video_out, action_out),)


class DreamZeroPipeline(nn.Module, CFGParallelMixin):
    """DreamZero world model pipeline with CFG parallel support.

    Multi-output model: predict_noise() returns (video_pred, action_pred).
    CFG is applied to video only; actions use positive branch only.
    Uses VideoActionScheduler composite for scheduler_step_maybe_with_cfg().
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # DreamZero config defaults
        self.cfg_scale = 5.0
        self.num_inference_steps = 16
        self.sigma_shift = 5.0
        self.action_dim = 8  # 7 joints + 1 gripper
        self.action_horizon = 24
        self.num_frame_per_block = 1

        # KV cache state (model-internal, per CFG rank)
        self.kv_cache: list | None = None
        self.kv_cache_neg: list | None = None
        self.crossattn_cache: list | None = None
        self.crossattn_cache_neg: list | None = None
        self.current_start_frame = 0

        # Cached encodings (reused across AR steps within a session)
        self.clip_feas: torch.Tensor | None = None
        self.ys: torch.Tensor | None = None

        # --- Text encoder (same as Wan2.2: umt5-xxl) ---
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        # --- VAE (same as Wan2.2: DistributedAutoencoderKLWan) ---
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial

        # --- Scheduler (FlowUniPCMultistepScheduler, same as Wan2.2) ---
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

        # --- Transformer (CausalWanModel — weights loaded via load_weights) ---
        self.transformer = CausalWanModel(
            model_type="i2v",
            patch_size=(1, 2, 2),
            frame_seqlen=880,  # Tokens per frame for 14B
            text_len=512,
            in_dim=16,  # VAE latent channels
            dim=5120,  # Hidden dimension (14B)
            ffn_dim=13824,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=40,
            num_layers=40,
            qk_norm=True,
            cross_attn_norm=True,
            num_frame_per_block=self.num_frame_per_block,
            action_dim=self.action_dim,
            num_action_per_block=32,
            num_state_per_block=1,
            max_num_embodiments=32,
            hidden_size=64,
            concat_first_frame_latent=True,
        )

        # --- Image encoder (CLIP — to be initialized) ---
        # TODO(PR4): Initialize CLIP image encoder (open-clip-xlm-roberta-large-vit-huge-14)
        # For now, image encoding is handled within the transformer's img_emb
        self.image_encoder = None  # Will be added when CLIP integration is ready

        # Weight sources for load_weights
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

    def predict_noise(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CausalWanModel.

        Expects kwargs to contain hidden_states, action, timestep_video,
        timestep_action, encoder_hidden_states, kv_caches, crossattn_caches,
        and other model-specific arguments.

        Returns:
            (video_pred, action_pred) tuple for multi-output CFG handling.
        """
        video_pred, action_pred, _updated_kv = self.transformer(
            x=kwargs["hidden_states"].transpose(1, 2),  # [B,C,T,H,W] → [B,T,C,H,W] for patchify
            timestep=kwargs["timestep_video"],
            action=kwargs.get("action"),
            timestep_action=kwargs.get("timestep_action"),
            context=kwargs["encoder_hidden_states"],
            seq_len=kwargs.get("seq_len", self.transformer.config.frame_seqlen),
            y=kwargs.get("y"),
            clip_feature=kwargs.get("clip_feature"),
            kv_cache=kwargs.get("kv_caches"),
            crossattn_cache=kwargs.get("crossattn_caches"),
            current_start_frame=kwargs.get("current_start_frame", self.current_start_frame),
            state=kwargs.get("state"),
            embodiment_id=kwargs.get("embodiment_id"),
        )
        return (video_pred, action_pred)

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """CFG combine: video gets standard CFG, action takes positive only.

        DreamZero applies CFG formula only to video predictions.
        Action predictions use the conditioned (positive) branch directly,
        since actions should follow the language instruction without
        unconditional blending.
        """
        (video_pos, action_pos) = positive_noise_pred
        (video_neg, _action_neg) = negative_noise_pred

        # Video: standard CFG formula
        video_combined = super().combine_cfg_noise(video_pos, video_neg, true_cfg_scale, cfg_normalize)

        # Action: positive branch only (no CFG)
        return (video_combined, action_pos)

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-step synchronization for CFG parallel numerical stability.

        Following LTX2 pattern from PR #2160: .contiguous() + cuda.synchronize()
        ensures bit-identical results across CFG ranks after all_gather + local step.
        """
        latents = tuple(tensor.contiguous() for tensor in latents)
        if not self._is_cfg_parallel_enabled(do_true_cfg):
            return latents

        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents

    def _is_cfg_parallel_enabled(self, do_true_cfg: bool) -> bool:
        return do_true_cfg and get_classifier_free_guidance_world_size() > 1

    def _reset_session(self):
        """Clear all session state (KV caches, cached encodings, frame counter)."""
        self.kv_cache = None
        self.kv_cache_neg = None
        self.crossattn_cache = None
        self.crossattn_cache_neg = None
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list, list]:
        """Initialize per-layer KV caches for conditioned and unconditioned branches.

        Each cache entry: [2, B, 0, num_heads, head_dim] (seq_len starts at 0, grows).
        """
        num_heads = self.transformer.config.num_heads
        head_dim = self.transformer.config.dim // num_heads
        num_layers = self.transformer.config.num_layers
        kv_cache: list = []
        kv_cache_neg: list = []
        for _ in range(num_layers):
            kv_cache.append(torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device))
            kv_cache_neg.append(torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device))
        return kv_cache, kv_cache_neg

    def _create_crossattn_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list, list]:
        """Initialize per-layer cross-attention caches (text conditioning).

        Each cache entry: [2, B, 512, num_heads, head_dim] (fixed text seq_len=512).
        """
        num_heads = self.transformer.config.num_heads
        head_dim = self.transformer.config.dim // num_heads
        num_layers = self.transformer.config.num_layers
        crossattn_cache: list = []
        crossattn_cache_neg: list = []
        for _ in range(num_layers):
            crossattn_cache.append(torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device))
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device)
            )
        return crossattn_cache, crossattn_cache_neg

    def _prefill_kv_cache(
        self,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        frame_seqlen: int,
    ) -> None:
        """Prefill KV cache with first-frame encoding and AR observation.

        This is called before the denoising loop to populate the KV cache
        with context from the conditioning image and current observation.
        The return values from predict_noise are discarded — only the
        KV cache side-effects matter.
        """
        batch_size = image_latents.shape[0]

        # First AR step: initialize KV caches
        if self.current_start_frame == 0:
            self.kv_cache, self.kv_cache_neg = self._create_kv_caches(
                batch_size,
                image_latents.dtype,
                image_latents.device,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size,
                image_latents.dtype,
                image_latents.device,
            )

            # Prefill: encode first frame into KV cache (timestep=0, no action)
            zero_timestep = torch.zeros([batch_size, 1], device=image_latents.device, dtype=torch.long)
            first_frame = image_latents.transpose(1, 2)  # [B,C,T,H,W] → [B,T,C,H,W]

            # Run cond branch prefill
            _vid, _act, updated_kv = self.transformer(
                x=first_frame,
                timestep=zero_timestep,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embeds,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1] if self.ys is not None else None,
                clip_feature=self.clip_feas,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start_frame=0,
            )
            if updated_kv is not None:
                for i, kv in enumerate(updated_kv):
                    self.kv_cache[i] = kv.clone()

            # Run uncond branch prefill (if CFG enabled)
            if negative_prompt_embeds is not None:
                _vid, _act, updated_kv_neg = self.transformer(
                    x=first_frame,
                    timestep=zero_timestep,
                    action=None,
                    timestep_action=None,
                    state=None,
                    embodiment_id=None,
                    context=negative_prompt_embeds,
                    seq_len=frame_seqlen,
                    y=self.ys[:, :, 0:1] if self.ys is not None else None,
                    clip_feature=self.clip_feas,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start_frame=0,
                )
                if updated_kv_neg is not None:
                    for i, kv in enumerate(updated_kv_neg):
                        self.kv_cache_neg[i] = kv.clone()

            self.current_start_frame = 1

    def diffuse(
        self,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        timesteps_video: torch.Tensor,
        timesteps_action: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        video_action_scheduler: VideoActionScheduler,
        do_true_cfg: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoising loop with KV cache and CFG parallel support.

        Follows the standard CFGParallelMixin pattern:
        1. predict_noise_maybe_with_cfg() — handles rank distribution + all_gather
        2. scheduler_step_maybe_with_cfg() — all ranks step locally
        3. _synchronize_cfg_parallel_step_output() — contiguous + cuda sync
        """
        for i, (t_video, t_action) in enumerate(zip(timesteps_video, timesteps_action)):
            # Build kwargs for positive (conditioned) and negative (unconditioned)
            positive_kwargs = {
                "hidden_states": video_latents,
                "action": action_latents,
                "timestep_video": t_video,
                "timestep_action": t_action,
                "encoder_hidden_states": prompt_embeds,
                # KV cache for conditioned branch
                "kv_caches": self.kv_cache,
                "crossattn_caches": self.crossattn_cache,
                **kwargs,
            }

            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": video_latents,
                    "action": action_latents,
                    "timestep_video": t_video,
                    "timestep_action": t_action,
                    "encoder_hidden_states": negative_prompt_embeds,
                    # KV cache for unconditioned branch
                    "kv_caches": self.kv_cache_neg,
                    "crossattn_caches": self.crossattn_cache_neg,
                    **kwargs,
                }
            else:
                negative_kwargs = None

            # Multi-output CFG: returns (video_pred, action_pred)
            video_pred, action_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

            # Composite scheduler step: video + action in one call
            video_latents, action_latents = self.scheduler_step_maybe_with_cfg(
                (video_pred, action_pred),
                (t_video, t_action),
                (video_latents, action_latents),
                do_true_cfg=do_true_cfg,
                per_request_scheduler=video_action_scheduler,
            )

            # Post-step sync for CFG parallel numerical stability
            video_latents, action_latents = self._synchronize_cfg_parallel_step_output(
                (video_latents, action_latents),
                do_true_cfg=do_true_cfg,
            )

        return video_latents, action_latents

    def _encode_prompt(self, prompt: str, negative_prompt: str | None = None):
        """Encode text prompt using umt5-xxl text encoder."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_embeds = self.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask)[0].to(
            self.dtype
        )

        negative_prompt_embeds = None
        if negative_prompt is not None:
            neg_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            negative_prompt_embeds = self.text_encoder(neg_inputs.input_ids, attention_mask=neg_inputs.attention_mask)[
                0
            ].to(self.dtype)
        elif self.cfg_scale > 1.0:
            # Empty prompt for unconditional branch
            neg_inputs = self.tokenizer(
                "",
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            negative_prompt_embeds = self.text_encoder(neg_inputs.input_ids, attention_mask=neg_inputs.attention_mask)[
                0
            ].to(self.dtype)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        **kwargs,
    ) -> DiffusionOutput:
        """Main entry point for DreamZero inference.

        Handles session management (reset/continue), encoding, denoising,
        and action output. Each call processes one AR step (one frame block).
        """
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}

        # Session management: reset KV cache on new session or explicit reset
        if extra_args.get("reset", False):
            self._reset_session()

        # 1. Extract observation from request
        prompt = req.prompts[0] if req.prompts else ""
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt", "")
        extra_args.get("images", {})
        state = extra_args.get("state", {})

        generator = req.sampling_params.generator
        num_inference_steps = req.sampling_params.num_inference_steps or self.num_inference_steps

        # 2. Encode text (cached across AR steps if prompt unchanged)
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(prompt)

        # 3. Encode video frames via VAE
        video_latents = extra_args.get("video_latents")
        if video_latents is not None:
            video_latents = torch.as_tensor(video_latents, device=self.device, dtype=self.dtype)
        else:
            # Encode observation images via VAE
            obs_images = extra_args.get("observation_images")
            if obs_images is not None:
                obs_tensor = torch.as_tensor(obs_images, device=self.device, dtype=torch.float32)
                # Normalize: [0,255] → [-1,1], [B,T,H,W,C] → [B,C,T,H,W]
                obs_tensor = (obs_tensor / 127.5 - 1.0).permute(0, 4, 1, 2, 3)
                video_latents = self.vae.encode(obs_tensor).latent_dist.sample()
                video_latents = video_latents.to(self.dtype)
            else:
                raise ValueError("Either 'video_latents' or 'observation_images' must be in extra_args.")

        batch_size, num_channels, num_frames, height, width = video_latents.shape
        # Compute frame sequence length (tokens per frame after patch embedding)
        frame_seqlen = (height // 2) * (width // 2)  # patch_size=(1,2,2)

        # 4. Prefill KV cache (first-frame + AR observation)
        self._prefill_kv_cache(
            image_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            frame_seqlen=frame_seqlen,
        )

        # 5. Prepare noise for video and action
        noise_video = torch.randn_like(video_latents, generator=generator)
        noise_action = torch.randn(
            batch_size,
            self.action_horizon,
            self.transformer.config.action_dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        # 6. Create schedulers and set timesteps
        video_scheduler = copy.deepcopy(self.scheduler)
        action_scheduler = copy.deepcopy(self.scheduler)
        video_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.sigma_shift)
        action_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.sigma_shift)
        video_action_scheduler = VideoActionScheduler(video_scheduler, action_scheduler)

        timesteps = video_scheduler.timesteps
        do_true_cfg = self.cfg_scale > 1.0 and negative_prompt_embeds is not None

        # 7. Prepare state features
        joint_pos = state.get("joint_position")
        gripper_pos = state.get("gripper_position")
        state_features = None
        if joint_pos is not None:
            joint_tensor = torch.as_tensor(joint_pos, device=self.device, dtype=self.dtype).unsqueeze(0)
            if gripper_pos is not None:
                grip_tensor = torch.as_tensor(gripper_pos, device=self.device, dtype=self.dtype).unsqueeze(0)
                state_features = torch.cat([joint_tensor, grip_tensor], dim=-1)
            else:
                state_features = joint_tensor

        # 8. Build diffuse kwargs
        y = None
        if self.ys is not None:
            if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
            else:
                y = self.ys[:, :, -self.num_frame_per_block :]

        diffuse_kwargs = {
            "seq_len": frame_seqlen,
            "y": y,
            "clip_feature": self.clip_feas,
            "current_start_frame": self.current_start_frame,
            "state": state_features,
            "embodiment_id": extra_args.get("embodiment_id"),
        }

        # 9. Run denoising loop
        video_latents_out, action_latents = self.diffuse(
            video_latents=noise_video,
            action_latents=noise_action,
            timesteps_video=timesteps,
            timesteps_action=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_action_scheduler=video_action_scheduler,
            do_true_cfg=do_true_cfg,
            **diffuse_kwargs,
        )

        # 10. Update AR frame counter
        self.current_start_frame += self.num_frame_per_block

        # 11. Extract actions and return
        actions_np = action_latents.cpu().float().numpy()

        return DiffusionOutput(
            output=video_latents_out,
            custom_output={"actions": actions_np},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with key remapping for DreamZero model.

        DreamZero checkpoint keys follow the pattern:
            action_head.model.* → transformer.*
            action_head.text_encoder.* → text_encoder.* (already loaded via from_pretrained)
            action_head.vae.* → vae.* (already loaded via from_pretrained)
            action_head.image_encoder.* → image_encoder.* (TODO)

        For LoRA checkpoints, .base_layer. is stripped from keys.
        """
        from vllm.model_executor.model_loader.weight_utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
