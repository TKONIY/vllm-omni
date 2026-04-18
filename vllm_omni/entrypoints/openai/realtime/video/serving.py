# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import iscoroutinefunction, signature
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from .protocol import RealtimeVideoGenerationRequest

logger = init_logger(__name__)


@dataclass
class RealtimeVideoTurnResult:
    """Normalized serving result for one websocket generation turn."""

    raw_output: Any
    rendered_prompt: str
    text_layers: dict[str, str] = field(default_factory=dict)
    request_id: str = ""
    chunk_index: int = 0


class RealtimeVideoServing:
    """Generic serving facade for realtime interactive video backends."""

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name

    def _resolve_stage_configs(self) -> Any:
        stage_configs = getattr(self.engine_client, "stage_configs", None)
        if stage_configs is not None:
            return stage_configs
        inner_engine = getattr(self.engine_client, "engine", None)
        return getattr(inner_engine, "stage_configs", None)

    def _resolve_diffusion_stage_ids(self) -> list[int]:
        stage_configs = self._resolve_stage_configs()
        if not stage_configs:
            return []
        return [
            index for index, stage_config in enumerate(stage_configs)
            if get_stage_type(stage_config) == "diffusion"
        ]

    def _resolve_collective_rpc(self) -> Any:
        inner_engine = getattr(self.engine_client, "engine", None)
        collective_rpc = getattr(inner_engine, "collective_rpc", None)
        if callable(collective_rpc) and not iscoroutinefunction(collective_rpc):
            return collective_rpc

        collective_rpc = getattr(self.engine_client, "collective_rpc", None)
        if callable(collective_rpc) and not iscoroutinefunction(collective_rpc):
            return collective_rpc
        return None

    @staticmethod
    def _supports_stage_ids(collective_rpc: Any) -> bool:
        try:
            return "stage_ids" in signature(collective_rpc).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _extract_chunk_index(output: Any) -> int:
        custom_output = getattr(output, "custom_output", {}) or {}
        realtime_video = custom_output.get("realtime_video", {}) or {}
        generated_chunks = realtime_video.get("generated_chunks")
        if isinstance(generated_chunks, int) and generated_chunks > 0:
            return generated_chunks - 1
        return 0

    def reset(self, session_id: str) -> None:
        logger.info("Realtime video session reset requested: %s", session_id)
        collective_rpc = self._resolve_collective_rpc()
        if collective_rpc is None:
            return

        kwargs: dict[str, Any] = {
            "method": "reset_realtime_video_session",
            "args": (session_id,),
        }
        if self._supports_stage_ids(collective_rpc):
            diffusion_stage_ids = self._resolve_diffusion_stage_ids()
            if not diffusion_stage_ids:
                return
            kwargs["stage_ids"] = diffusion_stage_ids

        try:
            collective_rpc(**kwargs)
        except Exception:
            logger.warning(
                "Failed to reset realtime video session %s via diffusion RPC.",
                session_id,
                exc_info=True,
            )

    def build_engine_request(self, request: RealtimeVideoGenerationRequest) -> OmniDiffusionRequest:
        prompt = OmniTextPrompt(prompt=request.rendered_prompt)
        if request.image is not None:
            prompt["multi_modal_data"] = {"image": request.image}

        sampling = OmniDiffusionSamplingParams(
            height=request.height,
            width=request.width,
            fps=request.fps,
            num_frames=request.num_frames,
            seed=request.seed,
            extra_args={
                "realtime_video": {
                    "session_id": request.session_id,
                    "text_layers": request.text_layers,
                    "rendered_prompt": request.rendered_prompt,
                    "control": request.control,
                    "chunk_size": request.chunk_size,
                    "shift": request.shift,
                    "max_attention_size": request.max_attention_size,
                    "reset": request.reset,
                }
            },
        )
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling,
            request_ids=[request.request_id],
        )

    async def generate(self, request: RealtimeVideoGenerationRequest) -> RealtimeVideoTurnResult:
        engine_request = self.build_engine_request(request)
        result = None
        async for output in self.engine_client.generate(
            prompt=engine_request.prompts[0],
            request_id=engine_request.request_ids[0],
            sampling_params_list=[engine_request.sampling_params],
        ):
            result = output

        if result is None:
            raise RuntimeError("Realtime video request produced no output.")

        return RealtimeVideoTurnResult(
            raw_output=result,
            rendered_prompt=request.rendered_prompt,
            text_layers=request.text_layers,
            request_id=request.request_id,
            chunk_index=self._extract_chunk_index(result),
        )
