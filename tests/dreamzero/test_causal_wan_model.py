# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CausalWanModel (small config, CPU only)."""

import torch


def _make_tiny_model():
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel

    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,
        text_len=16,
        in_dim=4,
        dim=64,
        ffn_dim=128,
        freq_dim=32,
        text_dim=64,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        qk_norm=True,
        cross_attn_norm=True,
        num_frame_per_block=1,
        action_dim=8,
        num_action_per_block=4,
        num_state_per_block=1,
        max_num_embodiments=4,
        hidden_size=32,
        concat_first_frame_latent=False,
    )
    model.eval()
    return model


def _make_empty_kv(num_layers, batch_size, num_heads, head_dim):
    return [torch.zeros(2, batch_size, 0, num_heads, head_dim) for _ in range(num_layers)]


def test_causal_wan_model_init():
    model = _make_tiny_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CausalWanModel tiny: {n_params:,} params, {len(model.blocks)} blocks")
    assert len(model.blocks) == 2
    print("CausalWanModel init: OK")


def test_prefill_no_action():
    """First prefill step (current_start_frame=0): no action, just encode first frame."""
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]

    x = torch.randn(B, 4, 1, 4, 4)  # [B, C, T=1, H, W]
    timestep = torch.tensor([[0]])
    context = torch.randn(B, 16, 64)

    with torch.no_grad():
        video_pred, action_pred, updated_kv = model(
            x=x,
            timestep=timestep,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            context=context,
            seq_len=4,
            y=None,
            clip_feature=None,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=0,
        )

    assert video_pred is not None
    assert len(updated_kv) == 2
    new_seq = updated_kv[0].shape[2]
    assert new_seq > 0, f"KV cache should grow, got {new_seq}"
    print(f"Prefill (no action): video_pred={video_pred.shape}, KV 0→{new_seq}")
    print("Prefill no action: OK")


def test_inference_with_action():
    """Denoising step (current_start_frame=1): with action/state tokens."""
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16

    # First: prefill to populate KV cache
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]

    x_prefill = torch.randn(B, 4, 1, 4, 4)
    with torch.no_grad():
        _, _, updated_kv = model(
            x=x_prefill,
            timestep=torch.tensor([[0]]),
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            context=torch.randn(B, 16, 64),
            seq_len=4,
            y=None,
            clip_feature=None,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=0,
        )

    # Update KV cache
    for i, kv in enumerate(updated_kv):
        kv_cache[i] = kv.clone()

    # Now: inference step with action at current_start_frame=1
    x = torch.randn(B, 4, 1, 4, 4)
    action = torch.randn(B, 4, 8)
    timestep_action = torch.tensor([[500] * 4])
    state = torch.randn(B, 1, 64)
    embodiment_id = torch.tensor([0])

    with torch.no_grad():
        video_pred, action_pred, updated_kv2 = model(
            x=x,
            timestep=torch.tensor([[500]]),
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            context=torch.randn(B, 16, 64),
            seq_len=4,
            y=None,
            clip_feature=None,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=1,
        )

    assert video_pred is not None
    assert action_pred is not None
    print(f"Inference with action: video={video_pred.shape}, action={action_pred.shape}")
    print("Inference with action: OK")


def test_kv_cache_grows_across_steps():
    """KV cache should grow incrementally across AR steps."""
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]

    context = torch.randn(B, 16, 64)
    sizes = []

    for step in range(3):
        x = torch.randn(B, 4, 1, 4, 4)
        action = torch.randn(B, 4, 8) if step > 0 else None
        timestep_action = torch.tensor([[500] * 4]) if step > 0 else None
        state = torch.randn(B, 1, 64) if step > 0 else None
        embodiment_id = torch.tensor([0]) if step > 0 else None

        with torch.no_grad():
            _, _, updated_kv = model(
                x=x,
                timestep=torch.tensor([[0 if step == 0 else 500]]),
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=context,
                seq_len=4,
                y=None,
                clip_feature=None,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=step,
            )

        for i, kv in enumerate(updated_kv):
            kv_cache[i] = kv.clone()
        sizes.append(kv_cache[0].shape[2])

    print(f"KV cache sizes across steps: {sizes}")
    assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)), f"KV cache should grow monotonically: {sizes}"
    print("KV cache growth: OK")


if __name__ == "__main__":
    test_causal_wan_model_init()
    test_prefill_no_action()
    test_inference_with_action()
    test_kv_cache_grows_across_steps()
    print("\nAll CausalWanModel tests passed!")
