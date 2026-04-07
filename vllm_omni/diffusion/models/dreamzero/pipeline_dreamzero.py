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

import numpy as np
import torch
import torch.nn as nn
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
        video_out = self.video_scheduler.step(
            noise_pred[0], t[0], latents[0], return_dict=False, generator=generator,
        )[0]
        action_out = self.action_scheduler.step(
            noise_pred[1], t[1], latents[1], return_dict=False, generator=generator,
        )[0]
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
        """Initialize pipeline components.
        Source: WANPolicyHead.__init__ (L156-235)
        """
        super().__init__()

        model_config = od_config.model_config
        model_path = od_config.model_path

        # ---- Tokenizer ---- (follows wan2_2 convention: pipeline owns tokenizer)
        from transformers import AutoTokenizer, UMT5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer",
        )

        # ---- Text encoder ---- (L169)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder",
        )

        # ---- Image encoder (CLIP) ---- (L170)
        # Same model as Wan2.2 I2V: open-clip-xlm-roberta-vit-huge-14
        # DreamZero uses use_31_block=True (skip last layer) ≡ hidden_states[-2]
        # Source: wan_video_image_encoder.py L856-887 (WanImageEncoder)
        # Reuse: pipeline_wan2_2_i2v.py L207-213
        from transformers import CLIPImageProcessor, CLIPVisionModel
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="image_processor",
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model_path, subfolder="image_encoder",
            )
        except (OSError, EnvironmentError) as exc:
            raise RuntimeError(
                "DreamZero requires `image_processor` and `image_encoder` subfolders "
                f"under model path `{model_path}`; zero-feature fallback is disabled."
            ) from exc

        # ---- VAE ---- (L171)
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            DistributedAutoencoderKLWan,
        )
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae",
        )

        # ---- Transformer (DiT backbone) ---- (L232)
        self.transformer = CausalWanModel(
            **model_config.get("transformer", {}),
        )

        # ---- Scheduler ---- (L172)
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
        )

        # ---- Pipeline state ---- (L180-195)
        self.state = DreamZeroState()

        # ---- Inference hyperparams ---- (L175-179)
        self.num_inference_steps: int = model_config.get("num_inference_steps", 16)
        self.cfg_scale: float = model_config.get("cfg_scale", 5.0)
        self.sigma_shift: float = model_config.get("sigma_shift", 5.0)
        self.num_frames: int = model_config.get("num_frames", 81)
        self.num_frame_per_block: int = model_config.get("num_frame_per_block", 2)
        self.action_horizon: int = model_config.get("action_horizon", 24)

        # Decoupled inference noise config                               # L112-118
        self.decouple_inference_noise: bool = model_config.get("decouple_inference_noise", False)
        self.video_inference_final_noise: float = model_config.get("video_inference_final_noise", 0.8)

        # Fixed seed for deterministic noise generation                  # L176
        self.seed: int = model_config.get("seed", 1140)

        # Model-level constants for state/action padding                 # dreamzero_cotrain.yaml
        self.max_state_dim: int = model_config.get("max_state_dim", 64)
        self.max_action_dim: int = model_config.get("max_action_dim", 32)

        # Fixed negative prompt for CFG uncond branch                    # dreamzero_cotrain.py L532
        self.negative_prompt: str = (
            "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
            "style, artwork, painting, image, still, grayscale, dull, worst quality, "
            "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
            "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
            "image, cluttered background, three legs, many people in the background, "
            "walking backwards."
        )

        # Embodiment name → numeric ID mapping (model knowledge)
        # Source: dreamzero transform/base.yaml embodiment_tag_to_projector_index
        self.embodiment_name_to_id: dict[str, int] = model_config.get(
            "embodiment_name_to_id", {
                "oxe_droid": 17,
                "agibot": 26,
                "gr1_unified": 24,
                "xdof": 22,
                "yam": 32,
                "mecka_hands": 27,
                "lapa": 27,
                "dream": 31,
            },
        )

        # Action normalization stats (per-embodiment, from checkpoint metadata)
        stats_path = model_config.get("action_norm_stats_path")
        if stats_path:
            self.action_norm_stats = self._load_action_norm_stats(stats_path)
        else:
            self.action_norm_stats: dict[str, dict[str, torch.Tensor]] = {}

        # Whether model uses relative actions (need to add back last state)
        self.relative_action: bool = model_config.get("relative_action", True)
        # Number of action dims that are relative (DROID: 7 = joint only, gripper is absolute)
        # Source: droid_relative.yaml L11 — relative_action_keys: [joint_position]
        self.relative_action_dim: int = model_config.get("relative_action_dim", 7)

        # TODO: DiT cache skip schedule (L201-218, L899-927)
        #   Static mask (dit_step_mask) + dynamic cosine-similarity skip
        #   Skips model forward on certain denoising steps, reuses previous prediction
        #   Not needed for correctness — pure latency optimization

    # -----------------------------------------------------------------------
    # CFGParallelMixin overrides
    # -----------------------------------------------------------------------

    def predict_noise(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Call CausalWanModel, return (video_pred, action_pred).
        Source: _run_diffusion_steps (L852-865) single model call
        """
        video_pred, action_pred, updated_kv_caches = self.transformer(  # L885-899
            x=kwargs["hidden_states"],
            timestep=kwargs["timestep_video"],
            context=kwargs["encoder_hidden_states"],
            seq_len=kwargs["seq_len"],
            kv_cache=kwargs["kv_cache"],
            crossattn_cache=kwargs["crossattn_cache"],
            current_start_frame=kwargs["current_start_frame"],
            y=kwargs.get("y"),
            clip_feature=kwargs.get("clip_feature"),
            action=kwargs.get("action"),
            timestep_action=kwargs.get("timestep_action"),
            state=kwargs.get("state_features"),
            embodiment_id=kwargs.get("embodiment_id"),
        )
        # KV cache update: side effect, write back to state          # L856-858
        if kwargs.get("update_kv_cache", False) and updated_kv_caches:
            is_neg = kwargs.get("is_negative", False)
            for i, kv in enumerate(updated_kv_caches):
                self.state.update_kv_cache(i, kv, is_negative=is_neg)

        video_pred = video_pred.clone()                              # L859
        if action_pred is not None:
            action_pred = action_pred.clone()                        # L861
        else:
            batch_size = kwargs["hidden_states"].shape[0]
            action_pred = torch.empty(
                batch_size,
                0,
                self.transformer.action_dim,
                device=video_pred.device,
                dtype=video_pred.dtype,
            )                                                       # CFG-parallel-safe dummy action pred
        return (video_pred, action_pred)

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
        """
        videos = videos.permute(0, 4, 1, 2, 3)                      # L952: b t h w c → b c t h w
        if videos.dtype == torch.uint8:                              # L954
            videos = videos.float() / 255.0                          # L955
            b, c, t, h, w = videos.shape                             # L957
            videos = videos.permute(0, 2, 1, 3, 4)                   # L958: b c t h w → b t c h w
            videos = videos.reshape(b * t, c, h, w)                  # L959
            # normalize: (x - 0.5) / 0.5 = x * 2 - 1               # L960 (self.normalize_video)
            videos = videos * 2.0 - 1.0
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # L961: back to b c t h w
        return videos.to(dtype=torch.bfloat16)                       # L966

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _encode_text(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text prompt via UMT5.
        Source: encode_prompt (L525-531)
        """
        seq_lens = attention_mask.gt(0).sum(dim=1).long()            # L526
        prompt_emb = self.text_encoder(                                # L527
            text_tokens, attention_mask,
        ).last_hidden_state
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)     # L528
        for i, v in enumerate(seq_lens):                             # L529-530
            prompt_emb[:, v:] = 0
        return prompt_emb

    # -----------------------------------------------------------------------
    # Image encoding
    # -----------------------------------------------------------------------

    def _encode_image(
        self, image: torch.Tensor, num_frames: int, height: int, width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode first frame via CLIP + VAE.
        Source: wan_flow_matching_action_tf.py encode_image (L547-564)
        CLIP source: wan_video_image_encoder.py L869-887 (WanImageEncoder.encode_image)
        Reuse: pipeline_wan2_2_i2v.py L275-288 (encode_image)
        Returns: (clip_feas, ys, image_latent)
        """
        device = image.device
        batch_size = image.shape[0]                                  # L548

        # CLIP encode                                                 # L549
        # DreamZero: model.visual(img, use_31_block=True) → [B, 257, 1280]
        # Equivalent: CLIPVisionModel(output_hidden_states=True).hidden_states[-2]
        # image: [B, T=1, C, H, W] → extract first frame for CLIP
        first_frame = image[:, 0]                                    # [B, C, H, W]
        # Denormalize [-1,1] → [0,1] for CLIPImageProcessor          # L879: mul_(0.5).add_(0.5)
        first_frame_01 = first_frame.float() * 0.5 + 0.5
        # CLIPImageProcessor expects PIL or [0,255] uint8 or [0,1] float
        pixel_values = self.image_processor(                         # L285
            images=first_frame_01, return_tensors="pt", do_rescale=False,
        ).pixel_values.to(device=device, dtype=self.image_encoder.dtype)
        clip_output = self.image_encoder(                            # L287-288
            pixel_values, output_hidden_states=True,
        )
        clip_context = clip_output.hidden_states[-2]                 # [B, 257, 1280]

        # Build mask                                                  # L550-554
        msk = torch.ones(batch_size, num_frames, height // 8, width // 8, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(batch_size, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)

        # VAE encode: first frame + zeros                             # L556-560
        image_input = image.transpose(1, 2)                          # L556: B,T,C,H,W → B,C,T,H,W
        image_zeros = torch.zeros(
            batch_size, 3, num_frames - 1, height, width,
            dtype=torch.bfloat16, device=device,
        )                                                            # L557
        with torch.no_grad():
            y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))  # L560

        new_image = y[:, :, 0:1]                                     # L561
        y = torch.concat([msk, y], dim=1)                            # L563: [B, 4+C_latent, T, H, W]

        return clip_context, y, new_image

    # -----------------------------------------------------------------------
    # KV cache prefill
    # -----------------------------------------------------------------------

    def _prefill_kv_cache(
        self,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        frame_seqlen: int,
        seq_len: int,
        do_true_cfg: bool,
    ) -> None:
        """Prefill KV cache with first frame and/or current observation.
        Source: lazy_joint_video_action L1078-1125

        Uses predict_noise_maybe_with_cfg() for CFG parallel — same path as
        the denoise loop. The mixin handles rank dispatch automatically.
        KV cache update happens as a side effect inside predict_noise().
        """
        batch_size = image_latents.shape[0]
        device = image_latents.device
        dtype = image_latents.dtype
        num_heads = getattr(self.transformer.blocks[0].self_attn, "tp_num_heads", self.transformer.num_heads)
        head_dim = self.transformer.dim // self.transformer.num_heads

        if self.state.current_start_frame == 0:
            # First call: create caches + encode first frame          # L1051-1063
            self.state.create_kv_caches(
                batch_size, dtype, device,
                self.transformer.num_layers, num_heads, head_dim,
            )

            zero_t = torch.zeros([batch_size, 1], device=device, dtype=torch.long)
            y_first = self.state.ys[:, :, 0:1] if self.state.ys is not None else None

            # Prefill via predict_noise_maybe_with_cfg                # L1080-1097
            # KV cache update is a side effect in predict_noise()
            common = dict(
                hidden_states=image_latents.transpose(1, 2),
                timestep_video=zero_t,
                seq_len=frame_seqlen,
                current_start_frame=0,
                y=y_first,
                clip_feature=self.state.clip_feas,
                update_kv_cache=True,
            )
            positive_kwargs = dict(
                encoder_hidden_states=prompt_embeds,
                kv_cache=self.state.get_kv_caches(False),
                crossattn_cache=self.state.get_crossattn_caches(False),
                is_negative=False,
                **common,
            )
            negative_kwargs = dict(
                encoder_hidden_states=negative_prompt_embeds,
                kv_cache=self.state.get_kv_caches(True),
                crossattn_cache=self.state.get_crossattn_caches(True),
                is_negative=True,
                **common,
            ) if negative_prompt_embeds is not None else None

            self.predict_noise_maybe_with_cfg(
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                cfg_normalize=False,
            )
            self.state.current_start_frame = 1                       # L1098

        # Subsequent: encode current observation                      # L1102-1125
        if self.state.current_start_frame != 1:
            csf = self.state.current_start_frame
            nfpb = self.num_frame_per_block
            current_ref = image_latents[:, -nfpb:]
            if self.state.ys is not None and csf <= self.state.ys.shape[2]:
                y = self.state.ys[:, :, csf - nfpb:csf]
            elif self.state.ys is not None:
                y = self.state.ys[:, :, -nfpb:]
            else:
                y = None

            zero_t = torch.zeros([batch_size, nfpb], device=device, dtype=torch.long)
            common = dict(
                hidden_states=current_ref.transpose(1, 2),
                timestep_video=zero_t,
                seq_len=seq_len,
                current_start_frame=csf - nfpb,
                y=y,
                clip_feature=self.state.clip_feas,
                update_kv_cache=True,
            )
            positive_kwargs = dict(
                encoder_hidden_states=prompt_embeds,
                kv_cache=self.state.get_kv_caches(False),
                crossattn_cache=self.state.get_crossattn_caches(False),
                is_negative=False,
                **common,
            )
            negative_kwargs = dict(
                encoder_hidden_states=negative_prompt_embeds,
                kv_cache=self.state.get_kv_caches(True),
                crossattn_cache=self.state.get_crossattn_caches(True),
                is_negative=True,
                **common,
            ) if negative_prompt_embeds is not None else None

            self.predict_noise_maybe_with_cfg(
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                cfg_normalize=False,
            )

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
        Source: lazy_joint_video_action L1164-1241

        For each timestep:
          1. Build positive_kwargs / negative_kwargs
          2. predict_noise_maybe_with_cfg()    → (video_pred, action_pred)
          3. scheduler_step_maybe_with_cfg()   → VideoActionScheduler
          4. _synchronize_cfg_parallel_step_output()
        """
        seq_len = kwargs["seq_len"]                                      # L1046
        state_features = kwargs.get("state_features")                    # L950
        embodiment_id = kwargs.get("embodiment_id")                      # L949

        # Shared kwargs for predict_noise (both cond & uncond branches)
        common_kwargs = dict(
            seq_len=seq_len,
            current_start_frame=self.state.current_start_frame,
            state_features=state_features,
            embodiment_id=embodiment_id,
            update_kv_cache=False,                                       # L1206: denoising steps don't update KV
        )

        noisy_input = video_latents                                      # L1129
        noisy_input_action = action_latents                              # L1130

        for index in range(len(timesteps_video)):                        # L1164
            video_timestep = timesteps_video[index]                      # L1169
            action_timestep = timesteps_action[index]                    # L1168
            batch_size = noisy_input.shape[0]

            # Build per-frame timestep tensors                           # L1172-1181
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noisy_input.device, dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noisy_input.device, dtype=torch.int64,
            ) * action_timestep

            # Compute y (image conditioning) slice                       # L1187-1190
            csf = self.state.current_start_frame
            if csf + self.num_frame_per_block <= self.state.ys.shape[2]:
                y = self.state.ys[:, :, csf:csf + self.num_frame_per_block]  # L1188
            else:
                y = self.state.ys[:, :, -self.num_frame_per_block:]          # L1190

            # Positive (cond) kwargs                                     # L1191-1208
            positive_kwargs = dict(
                hidden_states=noisy_input.transpose(1, 2),               # L1192
                timestep_video=timestep,
                encoder_hidden_states=prompt_embeds,
                kv_cache=self.state.get_kv_caches(False),
                crossattn_cache=self.state.get_crossattn_caches(False),
                y=y,
                clip_feature=self.state.clip_feas,
                action=noisy_input_action,                               # L1194
                timestep_action=timestep_action,                         # L1195
                is_negative=False,
                **common_kwargs,
            )

            # Negative (uncond) kwargs
            if do_true_cfg and negative_prompt_embeds is not None:
                negative_kwargs = dict(
                    hidden_states=noisy_input.transpose(1, 2),
                    timestep_video=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    kv_cache=self.state.get_kv_caches(True),
                    crossattn_cache=self.state.get_crossattn_caches(True),
                    y=y,
                    clip_feature=self.state.clip_feas,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    is_negative=True,
                    **common_kwargs,
                )
            else:
                negative_kwargs = None

            # CFG-parallel predict_noise                                 # L1209-1210
            noise_pred = self.predict_noise_maybe_with_cfg(
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                cfg_normalize=False,
            )
            flow_pred, flow_pred_action = noise_pred

            # Scheduler step: video + action                             # L1225-1240
            latents = (noisy_input, noisy_input_action)
            t = (video_timestep, action_timestep)
            noise_pred_tuple = (flow_pred.transpose(1, 2), flow_pred_action)  # L1226
            step_output = video_action_scheduler.step(
                noise_pred_tuple, t, latents, generator=kwargs.get("generator"),
            )
            noisy_input, noisy_input_action = step_output[0]

            # Post-step sync                                             # PR #2160
            noisy_input, noisy_input_action = self._synchronize_cfg_parallel_step_output(
                (noisy_input, noisy_input_action), do_true_cfg,
            )

        return noisy_input, noisy_input_action                          # L1242-1243

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        """Full inference step. Called by DiffusionEngine.step().
        Source: WANPolicyHead.lazy_joint_video_action (L929-1270)
        """
        extra_args = req.sampling_params.extra_args or {}
        unified_obs = extra_args["unified_obs"]
        device = get_local_device()

        # ---- Step 1: Extract inputs from unified observation ----
        prompt_str = unified_obs["prompt"]                               # str (templated)
        stitched = unified_obs["images"]                                 # ndarray (T,H,W,C) from transform
        if not isinstance(stitched, np.ndarray):
            stitched = np.asarray(stitched)
        embodiment_name = unified_obs.get("embodiment_name", "oxe_droid")
        embodiment_id = torch.tensor(                                    # (B,) tensor for CategorySpecificMLP
            [self.embodiment_name_to_id.get(embodiment_name, 0)],
            dtype=torch.long, device=device,
        )

        # State: raw from transform → pad to (B, state_horizon=1, max_state_dim)
        raw_state = unified_obs.get("state")
        if raw_state is not None:
            if not isinstance(raw_state, np.ndarray):
                raw_state = np.asarray(raw_state, dtype=np.float64)
            raw_state = raw_state.flatten()
            padded = np.zeros(self.max_state_dim, dtype=np.float64)
            n = min(len(raw_state), self.max_state_dim)
            padded[:n] = raw_state[:n]
            state_features = torch.from_numpy(padded).reshape(1, 1, self.max_state_dim).to(
                device=device, dtype=torch.bfloat16,                     # (B=1, state_horizon=1, max_state_dim)
            )
        else:
            state_features = None

        # ---- Step 1b: Tokenize ---- (wan2_2 convention: pipeline owns tokenizer)
        text_inputs = self.tokenizer(
            prompt_str, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt", add_special_tokens=True,
        )
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # ---- Step 2: Check reset + accumulate frames ---- (L968-981)
        # Explicit reset from serving layer (session switch / client request)
        if extra_args.get("reset", False):
            self.state.reset()
        # Auto-reset based on model state (before accumulation)
        if self.state.should_reset(text_tokens, 0, self.transformer.local_attn_size):
            self.state.reset()
        self.state.language = text_tokens                                # L970/975

        # Frame accumulation: stitched single frame → multi-frame video
        video_frames = self.state.accumulate_frames(stitched)            # (T, H, W, C)
        videos = torch.from_numpy(video_frames).unsqueeze(0).to(device)  # (B=1, T, H, W, C)

        # ---- Step 3: Preprocess video ---- (L952-966)
        videos = self._preprocess_video(videos)                          # → [B,C,T,H,W] bf16
        _, _, num_frames_raw, height, width = videos.shape

        # ---- Step 4: Encode text ---- (L986-991)
        prompt_embeds = self._encode_text(text_tokens, attention_mask)
        # Negative prompt for CFG uncond branch (model constant)
        negative_prompt_embeds = None
        if self.cfg_scale > 1.0:
            neg_inputs = self.tokenizer(
                self.negative_prompt, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt", add_special_tokens=True,
            )
            negative_prompt_embeds = self._encode_text(
                neg_inputs["input_ids"].to(device),
                neg_inputs["attention_mask"].to(device),
            )

        # ---- Step 5: Encode image (first call only) ---- (L1002-1005)
        # Extract first/last frame for CLIP + VAE encoding
        if num_frames_raw == 4 or num_frames_raw == 9:                   # L996-999
            image = videos[:, :, -1:].transpose(1, 2)                    # L998: real-world eval
        else:
            image = videos[:, :, :1].transpose(1, 2)                     # L1000

        if self.state.current_start_frame == 0:                          # L1002
            clip_feas, ys, image = self._encode_image(
                image, self.num_frames, height, width,
            )
            self.state.clip_feas = clip_feas.to(dtype=image.dtype)       # L1004
            self.state.ys = ys.to(dtype=image.dtype)                     # L1005

        # ---- Step 6: VAE encode observation frames ---- (L1013-1038)
        if self.state.current_start_frame != 0:                          # L1013-1038
            # Subsequent calls: encode current observation via VAE
            if (num_frames_raw - 1) // 4 == self.num_frame_per_block:
                pass                                                     # L1020: no further action
            elif num_frames_raw // 4 != self.num_frame_per_block:
                # Repeat to match num_frame_per_block                    # L1023-1027
                repeat_factor = self.num_frame_per_block // (num_frames_raw // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                first_frame = videos[:, :, 0:1]
                videos = torch.cat([first_frame, videos], dim=2)
            else:
                first_frame = videos[:, :, 0:1]                          # L1029-1030
                videos = torch.cat([first_frame, videos], dim=2)

            with torch.no_grad():
                image = self.vae.encode(videos)                          # L1032-1038

        # ---- Step 7: Generate noise (deterministic) ---- (L1041-1042, L176, L771)
        batch_size = image.shape[0]
        generator = torch.Generator(device=device).manual_seed(self.seed)  # L771
        noise_obs = torch.randn(
            batch_size, 16, self.num_frame_per_block, height // 8, width // 8,
            device=device, dtype=torch.bfloat16, generator=generator,
        )                                                                # L1041
        generator = torch.Generator(device=device).manual_seed(self.seed)  # L771
        noise_action = torch.randn(
            batch_size, self.action_horizon, self.transformer.action_dim,
            device=device, dtype=torch.bfloat16, generator=generator,
        )                                                                # L1042

        _, num_channels, num_frames, h_latent, w_latent = noise_obs.shape
        frame_seqlen = int(h_latent * w_latent / 4)                      # L1045
        seq_len = frame_seqlen * num_frames                              # L1046

        image = image.transpose(1, 2)                                    # L1048: [B,C,T,H,W]→[B,T,C,H,W]
        noise_obs = noise_obs.transpose(1, 2)                            # L1049

        # ---- Step 8: Prefill KV cache, ---- (L1078-1125)
        do_true_cfg = self.cfg_scale > 1.0 and negative_prompt_embeds is not None
        self._prefill_kv_cache(
            image, prompt_embeds, negative_prompt_embeds,
            frame_seqlen, seq_len, do_true_cfg,
        )

        # ---- Step 9: Create schedulers ---- (L1134-1155)
        sample_scheduler = copy.deepcopy(self.scheduler)                 # L1134-1137
        sample_scheduler_action = copy.deepcopy(self.scheduler)          # L1138-1141
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=device, shift=self.sigma_shift,
        )                                                                # L1142-1143
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=device, shift=self.sigma_shift,
        )                                                                # L1144-1145

        # Decoupled inference: video sigmas end early                    # L1150-1157
        if self.decouple_inference_noise:
            video_final_noise = self.video_inference_final_noise
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)

        video_action_scheduler = VideoActionScheduler(
            sample_scheduler, sample_scheduler_action,
        )

        # ---- Step 10: Denoising loop ---- (L1164-1241)
        video_out, action_out = self.diffuse(
            video_latents=noise_obs,                                     # L1129
            action_latents=noise_action,                                 # L1130
            timesteps_video=sample_scheduler.timesteps,
            timesteps_action=sample_scheduler_action.timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_action_scheduler=video_action_scheduler,
            do_true_cfg=do_true_cfg,
            seq_len=seq_len,
            state_features=state_features,
            embodiment_id=embodiment_id,
        )

        # ---- Step 11: Post-process ---- (L1242-1273)
        if self.state.current_start_frame == 1:                          # L1246-1247
            video_out = torch.cat([image, video_out], dim=1)
        self.state.current_start_frame += self.num_frame_per_block       # L1248

        # ---- Step 12: Action denormalization ---- (sim_policy.py L500-569)
        # q99 denorm: [-1,1] → real values
        action_out = self._denormalize_action(action_out, embodiment_name)

        # Relative → absolute: only for relative_action_keys (joint_position only)
        # Source: droid_relative.yaml L11 — relative_action_keys: [joint_position]
        # gripper_position is NOT relative, so don't add state back to it
        if self.relative_action and state_features is not None:
            n_relative = self.relative_action_dim                        # 7 for DROID (joint only)
            # state_features: (B, 1, max_state_dim) → squeeze state_horizon
            last_state = state_features[:, 0, :n_relative]               # (B, n_relative)
            action_out[..., :n_relative] = (
                action_out[..., :n_relative] + last_state.unsqueeze(1)   # broadcast over horizon
            )

        # Squeeze batch dim for output: (B, horizon, dim) → (horizon, dim)
        actions_np = action_out.squeeze(0).float().cpu().numpy()         # (horizon, max_action_dim)

        return DiffusionOutput(
            custom_output={
                "actions": actions_np,                                   # L1273
                "video_pred": video_out.transpose(1, 2).cpu(),
            },
        )

    # -----------------------------------------------------------------------
    # Action denormalization
    # -----------------------------------------------------------------------

    def _load_action_norm_stats(self, stats_path: str) -> dict[str, dict[str, torch.Tensor]]:
        """Load per-embodiment action normalization stats from metadata.json.
        Source: metadata.json → statistics.action.{joint_position,gripper_position}.{q01,q99}

        Returns: {embodiment_name: {"q01": Tensor(action_dim,), "q99": Tensor(action_dim,)}}
        """
        import json
        with open(stats_path) as f:
            metadata = json.load(f)

        result = {}
        for emb_name, emb_data in metadata.items():
            action_stats = emb_data.get("statistics", {}).get("action", {})
            q01_parts, q99_parts = [], []
            # Concatenate joint_position + gripper_position stats
            for key in ["joint_position", "gripper_position"]:
                if key in action_stats:
                    q01_parts.extend(action_stats[key]["q01"])
                    q99_parts.extend(action_stats[key]["q99"])
            if q01_parts:
                result[emb_name] = {
                    "q01": torch.tensor(q01_parts, dtype=torch.float32),
                    "q99": torch.tensor(q99_parts, dtype=torch.float32),
                }
        return result

    def _denormalize_action(
        self, action: torch.Tensor, embodiment_name: str,
    ) -> torch.Tensor:
        """Denormalize action from [-1,1] to real values using q99 mode.
        Source: state_action.py Normalizer.inverse() L188-207

        Formula: real = (normalized + 1) / 2 * (q99 - q01) + q01
        """
        if embodiment_name not in self.action_norm_stats:
            return action
        stats = self.action_norm_stats[embodiment_name]
        q01 = stats["q01"].to(device=action.device, dtype=action.dtype)
        q99 = stats["q99"].to(device=action.device, dtype=action.dtype)
        # action shape: (B, horizon, action_dim) or (B, horizon, max_action_dim)
        # q01/q99 shape: (actual_action_dim,) — only denorm actual dims
        actual_dim = q01.shape[0]
        action_real = action.clone()
        action_real[..., :actual_dim] = (
            (action[..., :actual_dim] + 1) / 2 * (q99 - q01) + q01
        )
        return action_real

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
