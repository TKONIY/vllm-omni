# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.lingbot_world_fast import (
    LingbotWorldFastSessionConfig,
    LingbotWorldFastSessionState,
    normalize_lingbot_control_chunk,
)

from .protocol import RealtimeVideoGenerationRequest
from .serving import RealtimeVideoServing, RealtimeVideoTurnResult

logger = init_logger(__name__)


class LingbotWorldFastRealtimeServing(RealtimeVideoServing):
    """Lingbot-World-Fast specific request normalization for realtime video."""

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        super().__init__(engine_client=engine_client, model_name=model_name)
        self.sessions: dict[str, LingbotWorldFastSessionState] = {}

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        super().reset(session_id)

    def _get_or_create_session_state(
        self,
        request: RealtimeVideoGenerationRequest,
        normalized_control: list,
    ) -> LingbotWorldFastSessionState:
        config = LingbotWorldFastSessionConfig(
            session_id=request.session_id,
            rendered_prompt=request.rendered_prompt,
            text_layers=dict(request.text_layers),
            width=request.width,
            height=request.height,
            fps=request.fps,
            chunk_size=request.chunk_size,
            seed=request.seed,
            shift=request.shift,
            max_attention_size=request.max_attention_size,
        )
        session_state = self.sessions.get(request.session_id)
        if session_state is None or session_state.config.signature != config.signature:
            session_state = LingbotWorldFastSessionState(config=config)
            self.sessions[request.session_id] = session_state
        elif request.reset:
            session_state.reset_runtime()

        for chunk in normalized_control:
            session_state.append_control(chunk)
        return session_state

    @staticmethod
    def _count_generated_frames(output: Any) -> int:
        custom_output = getattr(output, "custom_output", {}) or {}
        video_chunk = custom_output.get("video_chunk")
        if hasattr(video_chunk, "shape") and len(video_chunk.shape) >= 1:
            return int(video_chunk.shape[0])

        images = getattr(output, "images", None) or []
        if images and hasattr(images[0], "shape") and len(images[0].shape) >= 1:
            return int(images[0].shape[0])

        multimodal_output = getattr(output, "multimodal_output", {}) or {}
        video = multimodal_output.get("video")
        if hasattr(video, "shape") and len(video.shape) >= 1:
            return int(video.shape[0])
        return 0

    def build_engine_request(self, request: RealtimeVideoGenerationRequest):
        normalized_control = [normalize_lingbot_control_chunk(item) for item in request.control]
        session_state = self._get_or_create_session_state(request, normalized_control)
        engine_request = super().build_engine_request(request)
        realtime_video = engine_request.sampling_params.extra_args["realtime_video"]
        realtime_video["backend"] = "lingbot_world_fast"
        realtime_video["control"] = [
            {
                "poses": chunk.poses,
                "intrinsics": chunk.intrinsics,
                "wasd_action": chunk.wasd_action,
                "ijkl_action": chunk.ijkl_action,
                "control_type": chunk.control_type,
            }
            for chunk in normalized_control
        ]
        realtime_video["control_type"] = (
            normalized_control[-1].control_type if normalized_control else "cam"
        )
        realtime_video["session_state"] = {
            "current_chunk_index": session_state.current_chunk_index,
            "generated_frame_count": session_state.generated_frame_count,
            "total_control_frames": session_state.total_control_frames,
            "prompt_changed": session_state.prompt_changed,
            "image_changed": session_state.image_changed,
        }
        logger.info(
            "Lingbot realtime request %s normalized %d control chunks (%s).",
            request.request_id,
            len(normalized_control),
            realtime_video["control_type"],
        )
        return engine_request

    async def generate(self, request: RealtimeVideoGenerationRequest) -> RealtimeVideoTurnResult:
        result = await super().generate(request)
        session_state = self.sessions.get(request.session_id)
        if session_state is not None:
            session_state.mark_chunk_generated(
                produced_frames=self._count_generated_frames(result.raw_output),
            )
            result.chunk_index = session_state.current_chunk_index - 1
        return result
