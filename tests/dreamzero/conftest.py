# SPDX-License-Identifier: Apache-2.0
import os
import socket

import pytest
import torch


@pytest.fixture(autouse=True)
def default_vllm_config(default_vllm_config):
    """Extend parent default_vllm_config: also init TP group.

    Parent fixture (tests/conftest.py) sets VllmConfig context.
    This fixture adds torch.distributed + model parallel init
    needed by ColumnParallelLinear / QKVParallelLinear etc.
    """
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")

    try:
        initialize_model_parallel(1, 1)
    except (AssertionError, RuntimeError):
        pass
    yield
