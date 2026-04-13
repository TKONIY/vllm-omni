"""Test DreamZero pipeline forward pass with real checkpoint weights."""

import json
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

CHECKPOINT_DIR = "/home/yangshen/code/dreamzero/checkpoints/dreamzero"


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_DIR),
    reason="DreamZero checkpoint not available",
)
def test_pipeline_forward_single_step(default_vllm_config):
    """Test a single forward pass: transform → pipeline.forward() → action output."""
    from dataclasses import dataclass, field

    from vllm_omni.diffusion.distributed.parallel_state import init_dit_group
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    try:
        init_dit_group(dit_parallel_size=1, backend="nccl")
    except (AssertionError, RuntimeError):
        pass

    # Init CFG group (needed by predict_noise_maybe_with_cfg)
    import vllm_omni.diffusion.distributed.parallel_state as dps

    if dps._CFG is None:
        from vllm.distributed.parallel_state import get_world_group

        from vllm_omni.diffusion.distributed.parallel_state import init_model_parallel_group

        dps._CFG = init_model_parallel_group(
            group_ranks=[[0]],
            local_rank=get_world_group().local_rank,
            backend="nccl",
            parallel_mode="classifier_free_guidance",
        )

    @dataclass
    class MockODConfig:
        model: str = CHECKPOINT_DIR
        model_config: dict = field(default_factory=dict)
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=lambda: {"tokenizer": "google/umt5-xxl"})
        dtype: object = torch.bfloat16

    # ---- Init + load weights ----
    print("\n=== Init pipeline ===")
    pipeline = DreamZeroPipeline(od_config=MockODConfig())

    with open(os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard_keys = defaultdict(list)
    for key, shard_file in index["weight_map"].items():
        shard_keys[shard_file].append(key)

    def weight_iter():
        for shard_file, keys in sorted(shard_keys.items()):
            with safe_open(os.path.join(CHECKPOINT_DIR, shard_file), framework="pt", device="cpu") as f:
                for key in keys:
                    yield key, f.get_tensor(key)

    pipeline.load_weights(weight_iter())
    pipeline = pipeline.to("cuda", dtype=torch.bfloat16)
    pipeline.eval()
    print("Pipeline loaded and moved to GPU")

    # ---- Build a minimal observation via transform ----
    from vllm_omni.entrypoints.openai.realtime.robot.transform.roboarena import RoboArenaTransform

    transform = RoboArenaTransform()

    obs = {
        "observation/exterior_image_0_left": np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/exterior_image_1_left": np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick up the red block",
        "session_id": "test-001",
    }
    unified_obs = transform.transform_input(obs)
    print(f"Transform output: images={unified_obs['images'].shape}, prompt={unified_obs['prompt'][:50]}...")

    # ---- Build OmniDiffusionRequest ----
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    extra_args = {
        "reset": True,
        "session_id": "test-001",
        "unified_obs": unified_obs,
    }
    sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
    request = OmniDiffusionRequest(
        prompts=["pick up the red block"],
        sampling_params=sampling_params,
        request_ids=["test-001"],
    )

    # ---- Forward pass ----
    print("\n=== Forward pass ===")
    t0 = time.time()
    with torch.no_grad():
        result = pipeline.forward(request)
    dt = time.time() - t0
    print(f"Forward pass: {dt:.2f}s")

    # ---- Verify output ----
    actions = result.output["actions"]
    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"Actions sample: {actions[0, :5]}")

    assert actions.ndim == 2, f"Expected 2D, got {actions.ndim}D"
    assert actions.shape[0] == 24, f"Expected horizon=24, got {actions.shape[0]}"
    assert not np.isnan(actions).any(), "Actions contain NaN"
    assert not np.isinf(actions).any(), "Actions contain Inf"

    # Transform output
    final_actions = transform.transform_output(
        SimpleNamespace(multimodal_output={"actions": result.output["actions"]}),
    )
    print(f"Final actions: shape={final_actions.shape}")
    assert final_actions.shape == (24, 8), f"Expected (24, 8), got {final_actions.shape}"

    print("\nFORWARD PASS TEST PASSED")
