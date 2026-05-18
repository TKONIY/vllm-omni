from __future__ import annotations

import torch
from torch import nn

from vllm_omni.uad.batch import UADBatchInputs, UADBatchItem, UADBatchItemOutput, UADBatchOutputs


class HunyuanImage3UADModel(nn.Module):
    """UAD-native HunyuanImage3 model shell.

    This lives under `vllm_omni.uad` because the current implementation is a
    research-only UAD shell, not a production model-executor registration.
    When `ar_model` is provided, AR items reuse only that model's forward
    structure here; runner owns logits projection and sampling, matching vLLM.
    Without `ar_model`, the old toy `+1` sampler remains available for
    scheduler/control-plane tests.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        ar_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ar_model = ar_model
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
            hidden_states: torch.Tensor | None = None
            if item.is_ar:
                next_token_id, hidden_states = self._run_ar_item(batch_inputs, item)

            item_outputs.append(
                UADBatchItemOutput(
                    request_id=item.request_id,
                    phase=item.phase,
                    output_index=item.output_index,
                    num_scheduled_tokens=item.num_tokens,
                    next_token_id=next_token_id,
                    hidden_states=hidden_states,
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

    def _run_ar_item(
        self,
        batch_inputs: UADBatchInputs,
        item: UADBatchItem,
    ) -> tuple[int | None, torch.Tensor | None]:
        if item.num_tokens <= 0:
            raise ValueError("AR UAD item requires at least one token")
        item_token_ids = batch_inputs.input_token_ids[item.token_start : item.token_end]
        if self.ar_model is None:
            last_token_id = int(item_token_ids.reshape(-1)[-1].item())
            return (last_token_id + 1) % self.vocab_size, None
        return None, self._run_backend_ar_forward(batch_inputs, item)

    def _run_backend_ar_forward(
        self,
        batch_inputs: UADBatchInputs,
        item: UADBatchItem,
    ) -> torch.Tensor:
        """Execute one AR forward through an existing HunyuanImage3-style model.

        The wrapper intentionally does not copy HunyuanImage3 internals. It
        feeds UAD-packed token ids and positions into the backend's public
        `forward` method only. Runner-side code handles logits and sampling.
        """
        assert self.ar_model is not None

        device = self._backend_device(batch_inputs.input_token_ids.device)
        input_ids = batch_inputs.input_token_ids[item.token_start : item.token_end].to(device=device)
        positions = batch_inputs.token_positions[item.token_start : item.token_end].to(device=device)

        hidden_states = self.ar_model(
            input_ids=input_ids,
            positions=positions,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"AR backend returned unsupported hidden state type: {type(hidden_states)!r}")
        return hidden_states

    def _backend_device(self, fallback: torch.device) -> torch.device:
        assert self.ar_model is not None
        parameters = getattr(self.ar_model, "parameters", None)
        if not callable(parameters):
            return fallback
        for parameter in parameters():
            return parameter.device
        return fallback


HunyuanImage3UADForConditionalGeneration = HunyuanImage3UADModel
