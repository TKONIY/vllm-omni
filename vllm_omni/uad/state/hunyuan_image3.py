from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.outputs import UADModelOutput, UADRunnerOutput
from vllm_omni.uad.request import UADPhaseUpdate, UADRequestState, UADToken
from vllm_omni.uad.state.base import UADModelStateMachine


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
    toy_image_context_token_count: int = 4
    toy_total_dit_steps: int = 2

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        image_base_size: int = 1024,
        toy_image_context_token_count: int = 4,
        toy_total_dit_steps: int = 2,
    ) -> HunyuanImage3UADStateConfig:
        """Build the UAD state helper from the production tokenizer.

        This mirrors the token-id lookup in the original vLLM
        HunyuanImage3 model constructor, but only keeps the IDs needed by
        the UAD request state machine. It does not install logits processors,
        stop tokens, model weights, or diffusion pipeline metadata.
        """
        ratio_33 = tokenizer.convert_tokens_to_ids("<img_ratio_33>")
        ratio_36 = tokenizer.convert_tokens_to_ids("<img_ratio_36>")
        extra_ratio_token_slices: tuple[tuple[int, int, int], ...] = ()
        if ratio_33 is not None and ratio_36 is not None:
            extra_ratio_token_slices = ((ratio_33, ratio_36 + 1, 33),)

        return cls(
            img_token_id=tokenizer.convert_tokens_to_ids("<img>"),
            boi_token_id=tokenizer.convert_tokens_to_ids("<boi>"),
            eoi_token_id=tokenizer.convert_tokens_to_ids("<eoi>"),
            cfg_token_id=tokenizer.convert_tokens_to_ids("<cfg>"),
            end_think_token_id=tokenizer.convert_tokens_to_ids("</think>"),
            recaption_token_id=tokenizer.convert_tokens_to_ids("<recaption>"),
            end_recaption_token_id=tokenizer.convert_tokens_to_ids("</recaption>"),
            answer_token_id=tokenizer.convert_tokens_to_ids("<answer>"),
            end_answer_token_id=tokenizer.convert_tokens_to_ids("</answer>"),
            image_size_token_id=tokenizer.convert_tokens_to_ids(f"<img_size_{image_base_size}>"),
            eos_token_id=tokenizer.eos_token_id,
            primary_ratio_token_start_id=tokenizer.convert_tokens_to_ids("<img_ratio_0>"),
            primary_ratio_token_end_id=tokenizer.convert_tokens_to_ids("<img_ratio_32>"),
            extra_ratio_token_slices=extra_ratio_token_slices,
            toy_image_context_token_count=toy_image_context_token_count,
            toy_total_dit_steps=toy_total_dit_steps,
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

    def build_toy_image_context_tokens(self) -> list[UADToken]:
        """Create placeholder image tokens for the current toy DiT phase.

        This is not the production HunyuanImage3 image-token construction.
        The real path will derive image context from tokenizer/model metadata,
        latent shape, and cache allocation; Step 3 only appends deterministic
        placeholders so scheduler and runner state transitions can be tested.
        """
        tokens = [
            UADToken(modality="image", token_id=self.img_token_id)
            for _ in range(self.toy_image_context_token_count)
        ]
        if self.eoi_token_id is not None:
            tokens.append(UADToken(modality="image", token_id=self.eoi_token_id))
        return tokens


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
        toy_image_context_token_count: int = 4,
        toy_total_dit_steps: int = 2,
    ) -> HunyuanImage3UADStateMachine:
        """Build the HunyuanImage3 state machine from the production tokenizer."""
        return cls(
            config=HunyuanImage3UADStateConfig.from_tokenizer(
                tokenizer,
                image_base_size=image_base_size,
                toy_image_context_token_count=toy_image_context_token_count,
                toy_total_dit_steps=toy_total_dit_steps,
            )
        )

    def update_request_state(
        self,
        *,
        request: UADRequestState,
        runner_output: UADRunnerOutput,
    ) -> UADModelOutput:
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
    ) -> UADModelOutput:
        """Apply HunyuanImage3 AR-token semantics after runner sampling."""
        ratio_index = self.config.ratio_index(sampled_token.token_id)
        if ratio_index is not None:
            image_context_tokens = self.config.build_toy_image_context_tokens()
            return UADModelOutput(
                request_id=request.request_id,
                new_engine_tokens=[sampled_token] + image_context_tokens,
                new_materialized_tokens=[],
                phase_update=UADPhaseUpdate(
                    phase="dit_step",
                    dit_step_index=0,
                    total_dit_steps=self.config.toy_total_dit_steps,
                    image_ratio_token_id=sampled_token.token_id,
                    image_ratio_index=ratio_index,
                    image_context_token_count=len(image_context_tokens),
                    pending_image_context_commit=True,
                ),
                finished=False,
            )

        materialized_tokens = []
        if not self.config.is_engine_only_token(sampled_token.token_id):
            materialized_tokens.append(sampled_token)
        return UADModelOutput(
            request_id=request.request_id,
            new_engine_tokens=[sampled_token],
            new_materialized_tokens=materialized_tokens,
            finished=False,
        )

    def _update_from_dit_step(
        self,
        *,
        request: UADRequestState,
    ) -> UADModelOutput:
        """Advance HunyuanImage3's current toy DiT denoise-step state."""
        if request.total_dit_steps <= 0:
            raise ValueError(f"request {request.request_id} entered dit_step without total_dit_steps")

        next_step_index = min(request.dit_step_index + 1, request.total_dit_steps)
        is_final_step = next_step_index >= request.total_dit_steps
        return UADModelOutput(
            request_id=request.request_id,
            phase_update=UADPhaseUpdate(
                phase="ar_decode" if is_final_step else "dit_step",
                dit_step_index=next_step_index,
                pending_image_context_commit=not is_final_step,
            ),
            finished=False,
        )
