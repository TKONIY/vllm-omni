# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Direct precision characterization for native vLLM ``RowParallelLinear``.

This test isolates the tensor-parallel linear itself from any model-specific
logic. It verifies two behaviors:

- ``TP=2 + fp32`` native ``RowParallelLinear`` matches the equivalent
  unsharded ``nn.functional.linear`` reference.
- ``TP=2 + bf16`` native ``RowParallelLinear`` does *not* exactly match the
  equivalent unsharded bf16 reference, showing the TP numeric drift exists in
  the layer path itself rather than only inside DreamZero.

Run:
    PYTHONPATH=. .venv/bin/python -m pytest \
        tests/diffusion/layers/test_row_parallel_linear_precision.py -v -s
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs required",
)

TP_SIZE = 2
IN_FEATURES = 64
OUT_FEATURES = 64
BATCH_SHAPE = (2, 3)
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5
BF16_DRIFT_MIN = 1e-2


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _run_worker(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype_name: str,
    expect_drift: bool,
) -> None:
    dtype = _dtype_from_name(dtype_name)
    device = torch.device(f"cuda:{local_rank}")

    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(device)
    current_omni_platform.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = VllmConfig(device_config=DeviceConfig(device="cuda"))
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=world_size,
            rank=local_rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )

        try:
            layer = (
                RowParallelLinear(
                    IN_FEATURES,
                    OUT_FEATURES,
                    bias=True,
                    input_is_parallel=False,
                    return_bias=False,
                    params_dtype=dtype,
                )
                .to(device=device, dtype=dtype)
                .eval()
            )

            if local_rank == 0:
                full_weight = torch.randn(
                    OUT_FEATURES,
                    IN_FEATURES,
                    device=device,
                    dtype=dtype,
                )
                full_bias = torch.randn(
                    OUT_FEATURES,
                    device=device,
                    dtype=dtype,
                )
                x = torch.randn(
                    *BATCH_SHAPE,
                    IN_FEATURES,
                    device=device,
                    dtype=dtype,
                )
            else:
                full_weight = torch.empty(
                    OUT_FEATURES,
                    IN_FEATURES,
                    device=device,
                    dtype=dtype,
                )
                full_bias = torch.empty(
                    OUT_FEATURES,
                    device=device,
                    dtype=dtype,
                )
                x = torch.empty(
                    *BATCH_SHAPE,
                    IN_FEATURES,
                    device=device,
                    dtype=dtype,
                )

            torch.distributed.broadcast(full_weight, src=0)
            torch.distributed.broadcast(full_bias, src=0)
            torch.distributed.broadcast(x, src=0)

            shard_size = layer.input_size_per_partition
            shard = full_weight[
                :,
                local_rank * shard_size : (local_rank + 1) * shard_size,
            ].contiguous()
            layer.weight.data.copy_(shard)
            assert layer.bias is not None
            layer.bias.data.copy_(full_bias)

            actual = layer(x)
            expected = F.linear(x, full_weight, full_bias)
            max_diff = (actual.float() - expected.float()).abs().max()
            torch.distributed.all_reduce(max_diff, op=torch.distributed.ReduceOp.MAX)
            diff = max_diff.item()
            if local_rank == 0:
                print(f"RowParallelLinear precision check: dtype={dtype_name}, tp={world_size}, max_diff={diff:.6e}")

            if expect_drift:
                assert diff >= BF16_DRIFT_MIN, (
                    f"Expected native bf16 TP RowParallelLinear drift >= {BF16_DRIFT_MIN:.3e}, got {diff:.3e}"
                )
            else:
                assert torch.allclose(actual.float(), expected.float(), atol=FP32_ATOL, rtol=FP32_RTOL), (
                    f"Expected fp32 TP RowParallelLinear to match full linear, got max_diff={diff:.3e}"
                )
        finally:
            cleanup_dist_env_and_memory()


def _run_case(dtype_name: str, *, expect_drift: bool) -> None:
    world_size = TP_SIZE
    master_port = _find_free_port()
    mp.spawn(
        _run_worker,
        args=(world_size, master_port, dtype_name, expect_drift),
        nprocs=world_size,
        join=True,
    )


def test_row_parallel_linear_tp2_fp32_matches_full_linear() -> None:
    _run_case("float32", expect_drift=False)


def test_row_parallel_linear_tp2_bf16_drifts_from_full_linear() -> None:
    _run_case("bfloat16", expect_drift=True)
