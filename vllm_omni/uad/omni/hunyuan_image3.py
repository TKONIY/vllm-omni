from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.uad.request import UADToken


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
        if self.primary_ratio_token_start_id <= token_id <= self.primary_ratio_token_end_id:
            return token_id - self.primary_ratio_token_start_id
        for start_id, end_id, index_offset in self.extra_ratio_token_slices:
            if start_id <= token_id < end_id:
                return index_offset + token_id - start_id
        return None

    def is_ratio_token(self, token_id: int) -> bool:
        return self.ratio_index(token_id) is not None

    @property
    def stage_transitions(self) -> dict[int, list[int]]:
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
        return self.eos_token_id is not None and self.is_ratio_token(token_id)

    def is_engine_only_token(self, token_id: int) -> bool:
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
        tokens = [
            UADToken(modality="image", token_id=self.img_token_id)
            for _ in range(self.toy_image_context_token_count)
        ]
        if self.eoi_token_id is not None:
            tokens.append(UADToken(modality="image", token_id=self.eoi_token_id))
        return tokens
