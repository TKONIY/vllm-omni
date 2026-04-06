# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline for vllm-omni.

Corresponds to: WANPolicyHead.lazy_joint_video_action (L929-1270)
Entry point for DiffusionEngine.step() → pipeline.forward(req)
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest


# ---------------------------------------------------------------------------
# VideoActionScheduler — composite scheduler (same pattern as LTX2 PR #2160)
# ---------------------------------------------------------------------------

class VideoActionScheduler:
    """Wraps video + action schedulers into single .step() interface.
    Source pattern: LTX2 VideoAudioScheduler (PR #2160)
    """

    def __init__(self, video_scheduler, action_scheduler):
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(noise_pred[0], t[0], latents[0], return_dict=False, generator=generator)[0]
        action_out = self.action_scheduler.step(noise_pred[1], t[1], latents[1], return_dict=False, generator=generator)[0]
        return ((video_out, action_out),)


# ---------------------------------------------------------------------------
# DreamZeroPipeline
# ---------------------------------------------------------------------------

class DreamZeroPipeline(nn.Module, CFGParallelMixin):
    """DreamZero world model pipeline.

    Multi-output: predict_noise() returns (video_pred, action_pred).
    CFG: video gets standard CFG, action takes positive branch only.
    State: DreamZeroState manages KV cache + frame buffer across forward() calls.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        """
        Initialize pipeline components.
        Source: WANPolicyHead.__init__ (L161-235) + lazy_joint_video_action setup

        Components to initialize:
        - self.tokenizer: AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
        - self.text_encoder: UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder")
          → 复用 wan2_2: from transformers import UMT5EncoderModel
        - self.vae: DistributedAutoencoderKLWan.from_pretrained(model, subfolder="vae")
          → 复用 wan2_2
        - self.transformer: CausalWanModel(**config_from_model_dir)
          → 已完成的 modeling/causal_wan_model.py
        - self.image_encoder: WanImageEncoder (CLIP)
          → 暂时 None 占位
        - self.scheduler: FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
          → 复用 wan2_2
        - self.state: DreamZeroState()
          → state_dreamzero.py

        Config defaults (from server startup log):
          cfg_scale=5.0, num_inference_steps=16, sigma_shift=5.0
          action_dim=32, action_horizon=24, num_frame_per_block=2
        """
        super().__init__()
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # CFGParallelMixin overrides
    # -----------------------------------------------------------------------

    def predict_noise(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Call CausalWanModel, return (video_pred, action_pred).
        Source: _run_diffusion_steps (L852-865) single model call

        kwargs will contain (set by diffuse()):
          hidden_states, action, timestep_video, timestep_action,
          encoder_hidden_states, kv_cache, crossattn_cache,
          seq_len, y, clip_feature, current_start_frame,
          state, embodiment_id
        """
        raise NotImplementedError

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Video: standard CFG. Action: positive only (no CFG).
        Source: L1212 — flow_pred = uncond + cfg_scale * (cond - uncond)
                        action = cond only (no uncond blending)
        """
        (video_pos, action_pos) = positive_noise_pred
        (video_neg, _) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, true_cfg_scale, cfg_normalize)
        return (video_combined, action_pos)

    # -----------------------------------------------------------------------
    # CFG parallel sync (PR #2160 pattern)
    # -----------------------------------------------------------------------

    def _synchronize_cfg_parallel_step_output(
        self, latents: tuple[torch.Tensor, torch.Tensor], do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-step sync: .contiguous() + cuda.synchronize()
        Source: PR #2160 LTX2 _synchronize_cfg_parallel_step_output
        """
        latents = tuple(t.contiguous() for t in latents)
        if do_true_cfg and get_classifier_free_guidance_world_size() > 1:
            device = next((t.device for t in latents if t.is_cuda), None)
            if device is not None:
                torch.cuda.current_stream(device).synchronize()
        return latents

    # -----------------------------------------------------------------------
    # Video preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_video(self, videos: torch.Tensor) -> torch.Tensor:
        """uint8 [B,T,H,W,C] → bfloat16 [B,C,T,H,W] normalized to [-1,1].
        Source: lazy_joint_video_action L952-966

        Steps:
        1. rearrange "b t h w c -> b c t h w"          (L952)
        2. if uint8: float()/255 → normalize(mean=0.5, std=0.5) → to bf16  (L954-963)
        3. to bfloat16                                   (L966)
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _encode_text(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text prompt via UMT5.
        Source: encode_prompt (L525-531) + _prepare_text_inputs (L793-805)

        Steps:
        1. self.text_encoder(input_ids, attention_mask) → [B, 512, 4096]
        2. Zero out padding positions (seq_lens onwards)
        3. Cast to bfloat16

        CFG: called twice (cond prompt + empty prompt for uncond).
        In CFG parallel, each rank only encodes its branch.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Image encoding
    # -----------------------------------------------------------------------

    def _encode_image(self, first_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode first frame via CLIP + VAE.
        Source: encode_image (L563-581)

        Steps:
        1. CLIP: self.image_encoder.encode_image(first_frame) → clip_feas [B, 257, 1280]
        2. VAE: self.vae.encode(first_frame + zeros) → y [B, C_latent, T, H, W]
        3. Build mask + concat: ys = [mask; y]  [B, C_latent+4, T, H, W]
        4. image = y[:, :, 0:1]  (first frame latent)

        Returns: (clip_feas, ys, image_latent)
        Cache: state.clip_feas and state.ys (only computed on first call)
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # KV cache prefill
    # -----------------------------------------------------------------------

    def _prefill_kv_cache(
        self,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        frame_seqlen: int,
    ) -> None:
        """Prefill KV cache with first frame and/or current observation.
        Source: lazy_joint_video_action L1078-1125

        First call (state.current_start_frame == 0):
          1. state.create_kv_caches(...)                          (L1051-1063)
          2. CausalWanModel(first_frame, t=0, action=None)         (L1080-1097)
             → updates state.kv_cache (cond) side effect
          3. Same for negative prompt → state.kv_cache_neg          (if CFG)
          4. state.current_start_frame = 1                          (L1098)

        Subsequent calls (state.current_start_frame > 1):
          1. CausalWanModel(current_obs_latent, t=0, action=None)  (L1108-1125)
             → updates KV cache with new observation context

        KV cache passed via kwargs in diffuse():
          positive_kwargs["kv_cache"] = state.kv_cache
          negative_kwargs["kv_cache"] = state.kv_cache_neg
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Denoising loop
    # -----------------------------------------------------------------------

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
        """Denoising loop with CFG parallel support.
        Source: lazy_joint_video_action L1164-1305

        For each timestep:
          1. Build positive_kwargs / negative_kwargs
             - positive: kv_cache=state.kv_cache, encoder_hidden_states=prompt_embeds
             - negative: kv_cache=state.kv_cache_neg, encoder_hidden_states=neg_embeds
          2. predict_noise_maybe_with_cfg()    → (video_pred, action_pred)
          3. scheduler_step_maybe_with_cfg()   → VideoActionScheduler
          4. _synchronize_cfg_parallel_step_output()

        DiT cache skip (optional, from DreamZero L1184):
          - dit_step_mask = [T,T,T,F,F,F,T,F,F,F,T,F,F,T,T,T]
          - When False: reuse previous prediction (no model forward)
          - Controlled by should_run_model() in original
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        """Full inference step. Called by DiffusionEngine.step().
        Source: WANPolicyHead.lazy_joint_video_action (L929-1270)

        Steps:
        1. Extract unified_obs from req.sampling_params.extra_args["unified_obs"]
        2. state.should_reset() → reset if needed                    (L968-981)
        3. state.accumulate_frames(obs) → stacked video              (via state)
        4. _preprocess_video()                                        (L952-966)
        5. _encode_text() → prompt_embeds + neg_embeds                (L986-991)
        6. _encode_image() → clip_feas, ys (first call only)         (L1002-1005)
        7. VAE encode observation frames                              (L1013-1038)
        8. Generate noise: video + action                             (L1041-1042)
        9. _prefill_kv_cache()                                        (L1078-1125)
        10. Create schedulers + VideoActionScheduler                  (L1134-1155)
        11. diffuse()                                                  (L1164-1305)
        12. state.current_start_frame += num_frame_per_block          (L1248)
        13. Return DiffusionOutput(custom_output={"actions": ndarray}) (L1273)
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights via AutoWeightsLoader.
        Source: follows wan2_2 pattern

        Key remapping (DreamZero checkpoint → vllm-omni):
          action_head.model.* → transformer.*
          (text_encoder and vae loaded via from_pretrained, not here)
        """
        from vllm.model_executor.model_loader.weight_utils import AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
