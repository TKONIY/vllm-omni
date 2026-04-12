#!/usr/bin/env python3
"""Minimal standalone WebSocket server for DreamZero e2e testing.

Starts a lightweight server that DreamZero's test_client_AR.py can connect to.
Uses DreamZeroPipeline directly, bypassing DiffusionEngine.

Usage:
    # Terminal 1: start server
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python tests/dreamzero/test_e2e_server.py --port 8000

    # Terminal 2: run DreamZero client
    cd /home/yangshen/code/dreamzero
    PYTHONPATH=. python test_client_AR.py --host localhost --port 8000 --use-zero-images
"""

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import websockets.asyncio.server
from openpi_client import msgpack_numpy
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "/home/yangshen/code/dreamzero/checkpoints/dreamzero"


def init_distributed():
    """Minimal distributed init for ColumnParallelLinear etc.
    Mirrors tests/dreamzero/conftest.py."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    # Use MASTER_PORT from env (caller sets it to avoid conflicts)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="env://")
    try:
        initialize_model_parallel(1, 1)
    except (AssertionError, RuntimeError):
        pass

    from vllm_omni.diffusion.distributed.parallel_state import init_dit_group
    try:
        init_dit_group(dit_parallel_size=1, backend="nccl")
    except (AssertionError, RuntimeError):
        pass

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


def load_pipeline():
    """Load DreamZeroPipeline with weights."""
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    @dataclass
    class MockODConfig:
        model: str = CHECKPOINT_DIR
        model_config: dict = field(default_factory=dict)
        model_class_name: str = "DreamZeroPipeline"
        model_paths: dict = field(default_factory=lambda: {"tokenizer": "google/umt5-xxl"})
        dtype: object = torch.bfloat16

    logger.info("Instantiating DreamZeroPipeline...")
    t0 = time.time()
    pipeline = DreamZeroPipeline(od_config=MockODConfig())
    logger.info(f"Init: {time.time()-t0:.1f}s")

    logger.info("Loading weights...")
    t0 = time.time()
    with open(os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard_keys = defaultdict(list)
    for key, shard_file in index["weight_map"].items():
        shard_keys[shard_file].append(key)

    def weight_iter():
        for shard_file, keys in sorted(shard_keys.items()):
            with safe_open(os.path.join(CHECKPOINT_DIR, shard_file),
                           framework="pt", device="cpu") as f:
                for key in keys:
                    yield key, f.get_tensor(key)

    loaded = pipeline.load_weights(weight_iter())
    logger.info(f"Loaded {len(loaded)} params in {time.time()-t0:.1f}s")

    pipeline = pipeline.to("cuda", dtype=torch.bfloat16)
    pipeline.eval()
    logger.info("Pipeline ready on GPU")
    return pipeline


class DreamZeroServer:
    """Minimal WebSocket server compatible with DreamZero test_client_AR.py."""

    def __init__(self, pipeline, port: int):
        self.pipeline = pipeline
        self.port = port

        from vllm_omni.entrypoints.openai.realtime.robot.transform.roboarena import RoboArenaTransform
        self.transform = RoboArenaTransform()
        self._call_count = 0
        self._session_id = None

    async def handler(self, websocket):
        """Handle a single client connection. Matches DreamZero policy_server.py."""
        logger.info("Client connected")

        # Send metadata
        metadata = {
            "image_resolution": (180, 320),
            "n_external_cameras": 2,
            "needs_wrist_camera": True,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_position",
        }
        await websocket.send(msgpack_numpy.packb(metadata))

        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue

                obs = msgpack_numpy.unpackb(message)
                endpoint = obs.pop("endpoint", "infer")

                if endpoint == "reset":
                    self.pipeline.state.reset()
                    self._call_count = 0
                    await websocket.send("reset successful")
                    logger.info("Reset")
                    continue

                # Session tracking
                session_id = obs.get("session_id")
                if session_id != self._session_id:
                    if self._session_id is not None:
                        self.pipeline.state.reset()
                        self._call_count = 0
                    self._session_id = session_id

                self._call_count += 1

                # Transform
                unified_obs = self.transform.transform_input(obs)

                # Build request
                from vllm_omni.diffusion.request import OmniDiffusionRequest
                from vllm_omni.inputs.data import OmniDiffusionSamplingParams

                extra_args = {
                    "reset": self._call_count <= 1,
                    "session_id": self._session_id or "default",
                    "unified_obs": unified_obs,
                }
                sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
                request = OmniDiffusionRequest(
                    prompts=[unified_obs["prompt"]],
                    sampling_params=sampling_params,
                    request_ids=[f"robot-{self._session_id or 'default'}"],
                )

                # Inference
                t0 = time.time()
                with torch.no_grad():
                    result = self.pipeline.forward(request)
                dt = time.time() - t0

                # Extract actions
                actions = self.transform.transform_output(
                    SimpleNamespace(multimodal_output={"actions": result.output["actions"]}),
                )
                logger.info(f"Infer #{self._call_count}: {dt:.2f}s, actions={actions.shape}")

                await websocket.send(msgpack_numpy.packb(actions))

        except websockets.exceptions.ConnectionClosed:
            pass
        logger.info("Client disconnected")

    async def serve(self):
        logger.info(f"Starting DreamZero server on port {self.port}")
        async with websockets.asyncio.server.serve(
            self.handler, "0.0.0.0", self.port,
            max_size=100 * 1024 * 1024,  # 100MB
        ):
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # VllmConfig context must wrap the entire process lifetime
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        init_distributed()
        pipeline = load_pipeline()
        server = DreamZeroServer(pipeline, args.port)
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
