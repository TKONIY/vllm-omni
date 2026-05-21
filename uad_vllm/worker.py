from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.v1.worker.worker_base import WorkerBase

from uad_vllm.runner import UADGPUModelRunner


class UADGPUWorker(WorkerBase):
    """Placeholder worker interface for a future UAD executor backend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.model_runner: UADGPUModelRunner | None = None

    def init_device(self) -> None:
        # TODO: mirror Omni GPU worker device init and install UADGPUModelRunner.
        self.model_runner = None
