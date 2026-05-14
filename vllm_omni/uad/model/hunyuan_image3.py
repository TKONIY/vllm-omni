from __future__ import annotations

from torch import nn

from vllm_omni.uad.batch import UADBatchInputs, UADBatchItemOutput, UADBatchOutputs


class HunyuanImage3UADModel(nn.Module):
    """Toy UAD-native HunyuanImage3 batch model shell.

    This lives under `vllm_omni.uad` because the current implementation is a
    research-only UAD shell, not a production model-executor registration. It
    does not load HunyuanImage3 weights yet. The important Step 4 behavior is
    that AR token slots and DiT latent slots enter one batch contract before
    fake model outputs are scattered back to requests.
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self.last_batch_inputs: UADBatchInputs | None = None
        self.last_batch_outputs: UADBatchOutputs | None = None

    def forward(
        self,
        batch_inputs: UADBatchInputs,
    ) -> UADBatchOutputs:
        self.forward_calls += 1
        self.last_batch_inputs = batch_inputs

        item_outputs: list[UADBatchItemOutput] = []
        for item in batch_inputs.items:
            next_token_id: int | None = None
            if item.input_kind == "token_ids":
                if item.num_tokens <= 0:
                    raise ValueError("AR UAD item requires at least one token")
                item_token_ids = batch_inputs.input_token_ids[item.token_start : item.token_end]
                last_token_id = int(item_token_ids.reshape(-1)[-1].item())
                next_token_id = (last_token_id + 1) % self.vocab_size

            item_outputs.append(
                UADBatchItemOutput(
                    request_id=item.request_id,
                    phase=item.phase,
                    output_index=item.output_index,
                    num_scheduled_tokens=item.num_tokens,
                    next_token_id=next_token_id,
                )
            )

        batch_outputs = UADBatchOutputs(
            item_outputs=tuple(item_outputs),
            num_ffn_tokens=batch_inputs.num_ffn_tokens,
            num_ar_tokens=batch_inputs.num_ar_tokens,
            num_dit_tokens=batch_inputs.num_dit_tokens,
        )
        self.last_batch_outputs = batch_outputs
        return batch_outputs


HunyuanImage3UADForConditionalGeneration = HunyuanImage3UADModel
