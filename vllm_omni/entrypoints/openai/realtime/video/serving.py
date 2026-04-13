# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
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

    def reset(self, session_id: str) -> None:
        logger.info("Realtime video session reset requested: %s", session_id)

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
        )
