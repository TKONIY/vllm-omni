from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.outputs import UADModelRunnerItemOutput, UADStateUpdate
from vllm_omni.uad.request import UADPhaseUpdate, UADRequestState, UADToken
from vllm_omni.uad.state.base import UADModelStateMachine


@dataclass(frozen=True)
class _Resolution:
    height: int
    width: int

    @property
    def ratio(self) -> float:
        return self.height / self.width


class _ResolutionGroup:
    """Lightweight copy of HunyuanImage3's ratio bucket math."""

    def __init__(self, base_size: int, step: int | None = None, align: int = 1) -> None:
        if base_size % align != 0:
            raise ValueError(f"base_size {base_size} is not divisible by align {align}")
        self.base_size = base_size
        self.align = align
        self.step = step or base_size // 16
        if self.step > base_size // 2:
            raise ValueError(f"step must be <= base_size // 2, got {self.step}")
        self.data = self._calc_by_step()

    def __getitem__(self, index: int) -> _Resolution:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def _calc_by_step(self) -> list[_Resolution]:
        if self.align > self.step:
            raise ValueError(f"align {self.align} must be <= step {self.step}")

        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [_Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break
            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(
                _Resolution(
                    cur_height // self.align * self.align,
                    cur_width // self.align * self.align,
                )
            )

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break
            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(
                _Resolution(
                    cur_height // self.align * self.align,
                    cur_width // self.align * self.align,
                )
            )

        return sorted(resolutions, key=lambda resolution: resolution.ratio)

    def get_target_size(self, width: int, height: int) -> tuple[int, int]:
        ratio = height / width
        index = min(range(len(self.data)), key=lambda i: abs(self.data[i].ratio - ratio))
        resolution = self.data[index]
        return resolution.width, resolution.height

    def get_base_size_and_ratio_index(self, width: int, height: int) -> tuple[int, int]:
        ratio = height / width
        index = min(range(len(self.data)), key=lambda i: abs(self.data[i].ratio - ratio))
        return self.base_size, index


@dataclass(frozen=True)
class HunyuanImage3UADGenerationMetadata:
    """Image-generation metadata derived at HunyuanImage3's AR -> DiT boundary."""

    image_width: int
    image_height: int
    token_width: int
    token_height: int
    base_size: int
    ratio_index: int
    image_token_count: int
    image_context_token_count: int
    latent_shape: tuple[int, int, int]
    num_inference_steps: int
    guidance_scale: float
    seed: int | None = None


@dataclass(frozen=True)
class HunyuanImage3UADStateConfig:
    """HunyuanImage3 token rules needed by the UAD state machine.

    The generation-mode rules mirror
    `HunyuanImage3ForConditionalGeneration.sample()`:
    `</think> -> <recaption>`, `</recaption> -> <answer><boi><img_size_*>`,
    `<img_size_*> -> <img_ratio_*>`, and ratio token -> EOS.
    """

    img_token_id: int = 128006
    boi_token_id: int | None = None
    eoi_token_id: int | None = None
    cfg_token_id: int | None = None
    end_think_token_id: int | None = None
    recaption_token_id: int | None = None
    end_recaption_token_id: int | None = None
    answer_token_id: int | None = None
    end_answer_token_id: int | None = None
    image_size_token_id: int | None = None
    eos_token_id: int | None = None
    primary_ratio_token_start_id: int = 128100
    primary_ratio_token_end_id: int = 128132
    extra_ratio_token_slices: tuple[tuple[int, int, int], ...] = ()
    image_base_size: int = 1024
    vae_downsample_factor: tuple[int, int] = (8, 8)
    patch_size: int = 2
    latent_channels: int = 16
    default_num_inference_steps: int = 50
    default_guidance_scale: float = 1.0
    default_seed: int | None = None
    toy_image_context_token_count: int | None = 4
    toy_total_dit_steps: int = 2

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        image_base_size: int = 1024,
        vae_downsample_factor: tuple[int, int] = (8, 8),
        patch_size: int = 2,
        latent_channels: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        toy_image_context_token_count: int | None = None,
        toy_total_dit_steps: int | None = None,
    ) -> HunyuanImage3UADStateConfig:
        """Build the UAD state helper from the production tokenizer.

        This mirrors the token-id lookup in the original vLLM
        HunyuanImage3 model constructor, but only keeps the IDs needed by
        the UAD request state machine. It does not install logits processors,
        stop tokens, model weights, or diffusion pipeline metadata.
        """
        def required(token: str) -> int:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                raise ValueError(f"HunyuanImage3 tokenizer is missing required token {token!r}")
            return token_id

        def optional(token: str) -> int | None:
            return tokenizer.convert_tokens_to_ids(token)

        ratio_33 = optional("<img_ratio_33>")
        ratio_36 = optional("<img_ratio_36>")
        extra_ratio_token_slices: tuple[tuple[int, int, int], ...] = ()
        if ratio_33 is not None and ratio_36 is not None:
            extra_ratio_token_slices = ((ratio_33, ratio_36 + 1, 33),)

        return cls(
            img_token_id=required("<img>"),
            boi_token_id=optional("<boi>"),
            eoi_token_id=optional("<eoi>"),
            cfg_token_id=optional("<cfg>"),
            end_think_token_id=optional("</think>"),
            recaption_token_id=optional("<recaption>"),
            end_recaption_token_id=optional("</recaption>"),
            answer_token_id=optional("<answer>"),
            end_answer_token_id=optional("</answer>"),
            image_size_token_id=optional(f"<img_size_{image_base_size}>"),
            eos_token_id=tokenizer.eos_token_id,
            primary_ratio_token_start_id=required("<img_ratio_0>"),
            primary_ratio_token_end_id=required("<img_ratio_32>"),
            extra_ratio_token_slices=extra_ratio_token_slices,
            image_base_size=image_base_size,
            vae_downsample_factor=vae_downsample_factor,
            patch_size=patch_size,
            latent_channels=latent_channels,
            default_num_inference_steps=num_inference_steps,
            default_guidance_scale=guidance_scale,
            default_seed=seed,
            toy_image_context_token_count=toy_image_context_token_count,
            toy_total_dit_steps=toy_total_dit_steps or num_inference_steps,
        )

    def ratio_index(self, token_id: int) -> int | None:
        """Return the image-ratio index encoded by a sampled ratio token.

        The original sampler treats ratio tokens as an allowed vocabulary
        slice after `<img_size_*>`. UAD needs the same recognition step to
        decide when AR generation has reached the AR-to-DiT boundary and to
        record image-shape metadata for later runner work.
        """
        if self.primary_ratio_token_start_id <= token_id <= self.primary_ratio_token_end_id:
            return token_id - self.primary_ratio_token_start_id
        for start_id, end_id, index_offset in self.extra_ratio_token_slices:
            if start_id <= token_id < end_id:
                return index_offset + token_id - start_id
        return None

    def is_ratio_token(self, token_id: int) -> bool:
        """Check whether a token is one of HunyuanImage3's ratio tokens.

        This is the UAD-side predicate corresponding to the original
        model's `_all_ratio_ids` set. It is intentionally a pure predicate so
        the runner can use it both for phase switching and forced-EOS logic.
        """
        return self.ratio_index(token_id) is not None

    @property
    def stage_transitions(self) -> dict[int, list[int]]:
        """Return forced AR-stage transition sequences.

        The returned map matches the production sampler's generation-mode
        transitions: `</think>` forces `<recaption>`, and `</recaption>`
        forces `<answer><boi><img_size_*>`. UAD exposes the map as data; it
        does not directly modify logits here.
        """
        transitions: dict[int, list[int]] = {}
        if self.end_think_token_id is not None and self.recaption_token_id is not None:
            transitions[self.end_think_token_id] = [self.recaption_token_id]
        if (
            self.end_recaption_token_id is not None
            and self.answer_token_id is not None
            and self.boi_token_id is not None
            and self.image_size_token_id is not None
        ):
            transitions[self.end_recaption_token_id] = [
                self.answer_token_id,
                self.boi_token_id,
                self.image_size_token_id,
            ]
        return transitions

    def get_forced_token(self, decoded_tokens: list[int]) -> int | None:
        """Compute the next forced transition token from AR output history.

        This is the stateless part of the original `_get_forced_token`
        sampler helper. The production path uses the result to mask logits;
        UAD uses it as runner-side state-machine guidance before later
        replacing the toy sampler with real model sampling.
        """
        for index in range(len(decoded_tokens) - 1, -1, -1):
            trigger = decoded_tokens[index]
            forced_sequence = self.stage_transitions.get(trigger)
            if forced_sequence is None:
                continue

            emitted = decoded_tokens[index + 1 :]
            matched = 0
            for expected, actual in zip(forced_sequence, emitted):
                if actual != expected:
                    return None
                matched += 1

            if matched < len(forced_sequence):
                return forced_sequence[matched]
            return None
        return None

    def should_force_eos_after(self, token_id: int) -> bool:
        """Return whether the next AR token should be EOS after `token_id`.

        The production sampler forces EOS immediately after an `<img_ratio_*>`
        token. UAD keeps the same rule so text generation can terminate while
        the request moves into image-context/DiT execution.
        """
        return self.eos_token_id is not None and self.is_ratio_token(token_id)

    def should_restrict_to_ratio_after(self, token_id: int) -> bool:
        """Return whether the next sampled AR token must be an image ratio."""
        return self.image_size_token_id is not None and token_id == self.image_size_token_id

    def ratio_index_for_size(self, width: int, height: int) -> int:
        """Return the HunyuanImage3 ratio bucket for a requested image size."""
        _, ratio_index = _ResolutionGroup(base_size=self.image_base_size).get_base_size_and_ratio_index(
            width,
            height,
        )
        return ratio_index

    def is_engine_only_token(self, token_id: int) -> bool:
        """Identify control tokens that should not be streamed as user text.

        vLLM's normal text path samples token IDs, appends them to request
        state, and detokenizes/streams visible text. UAD keeps HunyuanImage3
        structural tokens in the engine ledger, but suppresses them from the
        materialized output ledger used for user-visible streaming.
        """
        control_token_ids = {
            self.img_token_id,
            self.boi_token_id,
            self.eoi_token_id,
            self.cfg_token_id,
            self.end_think_token_id,
            self.recaption_token_id,
            self.end_recaption_token_id,
            self.answer_token_id,
            self.end_answer_token_id,
            self.image_size_token_id,
        }
        return token_id in control_token_ids or self.is_ratio_token(token_id)

    def build_generation_metadata(
        self,
        ratio_index: int,
        *,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> HunyuanImage3UADGenerationMetadata:
        """Derive DiT metadata from a sampled `<img_ratio_*>` token.

        This mirrors HunyuanImage3ImageProcessor.build_image_info without
        loading model weights or image processors. `token_width` and
        `token_height` are the generated-image token grid used by the DiT
        final layer; `latent_shape` is the initial denoise latent shape.
        """
        resolution_group = _ResolutionGroup(base_size=self.image_base_size)
        if ratio_index < 0 or ratio_index >= len(resolution_group):
            raise ValueError(
                f"ratio_index {ratio_index} is out of range for base_size {self.image_base_size} "
                f"with {len(resolution_group)} buckets"
            )

        resolution = resolution_group[ratio_index]
        image_height = resolution.height
        image_width = resolution.width
        token_height = image_height // (self.vae_downsample_factor[0] * self.patch_size)
        token_width = image_width // (self.vae_downsample_factor[1] * self.patch_size)
        image_token_count = token_height * token_width

        if self.toy_image_context_token_count is None:
            image_context_token_count = image_token_count + 1  # timestep slot + generated image tokens
            if self.eoi_token_id is not None:
                image_context_token_count += 1
        else:
            image_context_token_count = self.toy_image_context_token_count
            if self.eoi_token_id is not None:
                image_context_token_count += 1

        latent_shape = (
            self.latent_channels,
            image_height // self.vae_downsample_factor[0],
            image_width // self.vae_downsample_factor[1],
        )
        default_steps = (
            self.toy_total_dit_steps
            if self.toy_image_context_token_count is not None
            else self.default_num_inference_steps
        )
        return HunyuanImage3UADGenerationMetadata(
            image_width=image_width,
            image_height=image_height,
            token_width=token_width,
            token_height=token_height,
            base_size=self.image_base_size,
            ratio_index=ratio_index,
            image_token_count=image_token_count,
            image_context_token_count=image_context_token_count,
            latent_shape=latent_shape,
            num_inference_steps=num_inference_steps or default_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else self.default_guidance_scale,
            seed=self.default_seed if seed is None else seed,
        )

    def build_generation_metadata_from_size(
        self,
        width: int,
        height: int,
        *,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> HunyuanImage3UADGenerationMetadata:
        """Derive DiT metadata from a requested output size."""
        return self.build_generation_metadata(
            self.ratio_index_for_size(width, height),
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def build_image_context_tokens(
        self,
        metadata: HunyuanImage3UADGenerationMetadata,
    ) -> list[UADToken]:
        """Create logical image-context tokens appended after a ratio token.

        In toy mode, this preserves the old small placeholder count. In real
        metadata mode (`toy_image_context_token_count=None`), the count matches
        HunyuanImage3's generated image sequence after `<img_ratio_*>`:
        timestep slot + generated image tokens + optional `<eoi>`.
        """
        context_token_count = metadata.image_context_token_count
        if self.eoi_token_id is not None:
            image_placeholder_count = context_token_count - 1
        else:
            image_placeholder_count = context_token_count
        tokens = [
            UADToken(modality="image", token_id=self.img_token_id)
            for _ in range(image_placeholder_count)
        ]
        if self.eoi_token_id is not None:
            tokens.append(UADToken(modality="image", token_id=self.eoi_token_id))
        return tokens

    def build_toy_image_context_tokens(self) -> list[UADToken]:
        """Compatibility wrapper for older toy tests."""
        metadata = self.build_generation_metadata(0)
        return self.build_image_context_tokens(metadata)


@dataclass
class HunyuanImage3UADStateMachine(UADModelStateMachine):
    """HunyuanImage3-specific UAD phase and output-ledger policy.

    `UADRunner` should not know that HunyuanImage3 uses `<img_ratio_*>` as
    its AR-to-DiT boundary. The scheduler calls this policy from
    `update_from_output()`, and this class decides whether a raw runner output
    becomes visible text, engine-only structure, or a phase switch into DiT.
    """

    config: HunyuanImage3UADStateConfig = field(default_factory=HunyuanImage3UADStateConfig)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        image_base_size: int = 1024,
        vae_downsample_factor: tuple[int, int] = (8, 8),
        patch_size: int = 2,
        latent_channels: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        toy_image_context_token_count: int | None = None,
        toy_total_dit_steps: int | None = None,
    ) -> HunyuanImage3UADStateMachine:
        """Build the HunyuanImage3 state machine from the production tokenizer."""
        return cls(
            config=HunyuanImage3UADStateConfig.from_tokenizer(
                tokenizer,
                image_base_size=image_base_size,
                vae_downsample_factor=vae_downsample_factor,
                patch_size=patch_size,
                latent_channels=latent_channels,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                toy_image_context_token_count=toy_image_context_token_count,
                toy_total_dit_steps=toy_total_dit_steps,
            )
        )

    def update_request_state(
        self,
        *,
        request: UADRequestState,
        runner_output: UADModelRunnerItemOutput,
    ) -> UADStateUpdate:
        """Apply HunyuanImage3 semantics to one raw runner output.

        This is called from scheduler `update_from_output()`, mirroring vLLM's
        request-state update path. The generic runner has already executed the
        item and does not know whether a token is a ratio/control token.
        """
        if runner_output.phase in ("ar_prefill", "ar_decode"):
            if runner_output.sampled_token is None:
                raise ValueError(f"AR runner output for {request.request_id} did not include sampled_token")
            return self._update_from_ar_token(
                request=request,
                sampled_token=runner_output.sampled_token,
            )
        if runner_output.phase == "dit_step":
            return self._update_from_dit_step(
                request=request,
            )
        raise NotImplementedError(f"unsupported HunyuanImage3 UAD phase: {runner_output.phase}")

    def _update_from_ar_token(
        self,
        *,
        request: UADRequestState,
        sampled_token: UADToken,
    ) -> UADStateUpdate:
        """Apply HunyuanImage3 AR-token semantics after runner sampling."""
        ratio_index = self.config.ratio_index(sampled_token.token_id)
        if ratio_index is not None:
            metadata = self.config.build_generation_metadata(
                ratio_index,
                seed=request.seed,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
            image_context_tokens = self.config.build_image_context_tokens(metadata)
            return UADStateUpdate(
                request_id=request.request_id,
                new_engine_tokens=[sampled_token] + image_context_tokens,
                new_materialized_tokens=[],
                phase_update=UADPhaseUpdate(
                    phase="dit_step",
                    dit_step_index=0,
                    total_dit_steps=metadata.num_inference_steps,
                    image_ratio_token_id=sampled_token.token_id,
                    image_ratio_index=ratio_index,
                    image_width=metadata.image_width,
                    image_height=metadata.image_height,
                    image_token_width=metadata.token_width,
                    image_token_height=metadata.token_height,
                    image_base_size=metadata.base_size,
                    image_context_token_count=metadata.image_context_token_count,
                    latent_shape=metadata.latent_shape,
                    seed=metadata.seed,
                    num_inference_steps=metadata.num_inference_steps,
                    guidance_scale=metadata.guidance_scale,
                    pending_image_context_commit=True,
                ),
                finished=False,
            )

        materialized_tokens = []
        if not self.config.is_engine_only_token(sampled_token.token_id):
            materialized_tokens.append(sampled_token)
        return UADStateUpdate(
            request_id=request.request_id,
            new_engine_tokens=[sampled_token],
            new_materialized_tokens=materialized_tokens,
            finished=False,
        )

    def _update_from_dit_step(
        self,
        *,
        request: UADRequestState,
    ) -> UADStateUpdate:
        """Advance HunyuanImage3's current toy DiT denoise-step state."""
        if request.total_dit_steps <= 0:
            raise ValueError(f"request {request.request_id} entered dit_step without total_dit_steps")

        next_step_index = min(request.dit_step_index + 1, request.total_dit_steps)
        is_final_step = next_step_index >= request.total_dit_steps
        return UADStateUpdate(
            request_id=request.request_id,
            phase_update=UADPhaseUpdate(
                phase="ar_decode" if is_final_step else "dit_step",
                dit_step_index=next_step_index,
                pending_image_context_commit=not is_final_step,
            ),
            finished=False,
        )
