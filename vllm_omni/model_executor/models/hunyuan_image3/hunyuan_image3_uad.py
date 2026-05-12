from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HunyuanImage3UADToyOutput:
    request_id: str
    next_token_id: int


class HunyuanImage3UADForConditionalGeneration(nn.Module):
    """Toy UAD-native HunyuanImage3 entrypoint for Step 0.

    The class intentionally exposes a real model-executor-style callable while
    avoiding HunyuanImage3 weight loading. Later steps replace the deterministic
    token rule with the real AR and DiT module calls.
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        phase: str = "ar_decode",
    ) -> HunyuanImage3UADToyOutput:
        if input_ids.numel() == 0:
            raise ValueError("HunyuanImage3UAD toy forward requires at least one input token")
        self.forward_calls += 1
        last_token_id = int(input_ids.reshape(-1)[-1].item())
        next_token_id = (last_token_id + 1) % self.vocab_size
        return HunyuanImage3UADToyOutput(request_id=request_id, next_token_id=next_token_id)
