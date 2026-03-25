# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VideoActionScheduler composite scheduler."""

import torch


def test_video_action_scheduler_step():
    """VideoActionScheduler.step() accepts tuples and returns tuple pair."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import VideoActionScheduler
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    video_sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    action_sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    video_sched.set_timesteps(4, device="cpu", shift=5.0)
    action_sched.set_timesteps(4, device="cpu", shift=5.0)

    composite = VideoActionScheduler(video_sched, action_sched)

    # Mock inputs
    video_pred = torch.randn(1, 16, 1, 10, 20)
    action_pred = torch.randn(1, 24, 8)
    video_latents = torch.randn(1, 16, 1, 10, 20)
    action_latents = torch.randn(1, 24, 8)
    t_video = video_sched.timesteps[0]
    t_action = action_sched.timesteps[0]

    result = composite.step(
        (video_pred, action_pred),
        (t_video, t_action),
        (video_latents, action_latents),
        return_dict=False,
    )

    # Result should be ((video_out, action_out),)
    assert isinstance(result, tuple)
    assert len(result) == 1
    video_out, action_out = result[0]
    assert video_out.shape == video_latents.shape
    assert action_out.shape == action_latents.shape


def test_video_action_scheduler_generator_passthrough():
    """Generator is passed to both sub-schedulers."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import VideoActionScheduler
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    video_sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    action_sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    video_sched.set_timesteps(4, device="cpu", shift=5.0)
    action_sched.set_timesteps(4, device="cpu", shift=5.0)

    composite = VideoActionScheduler(video_sched, action_sched)
    gen = torch.Generator().manual_seed(42)

    video_pred = torch.randn(1, 16, 1, 10, 20)
    action_pred = torch.randn(1, 24, 8)
    video_latents = torch.randn(1, 16, 1, 10, 20)
    action_latents = torch.randn(1, 24, 8)

    # Should not raise
    result = composite.step(
        (video_pred, action_pred),
        (video_sched.timesteps[0], action_sched.timesteps[0]),
        (video_latents, action_latents),
        generator=gen,
    )
    assert result[0][0].shape == video_latents.shape


if __name__ == "__main__":
    test_video_action_scheduler_step()
    test_video_action_scheduler_generator_passthrough()
    print("All VideoActionScheduler tests passed!")
