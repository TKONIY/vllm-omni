# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via /v1/realtime/robot/openpi.

Mirrors OpenAIServingRealtime: holds engine reference, provides inference
abstraction. Engine-agnostic — works with DiffusionEngine, LLM EngineClient,
or any object implementing step(request).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer (mirrors OpenAIServingRealtime).

    Holds the engine reference, builds requests from observations,
    and extracts actions from results. Engine-agnostic: works with
    DiffusionEngine or LLM EngineClient.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name

    def reset(self, obs: dict) -> None:
        """Handle reset. Override for engine-specific cleanup."""
        pass

    async def infer(self, obs: dict, session_state: Any = None) -> np.ndarray:
        """Run inference: observation dict -> action ndarray."""
        request = self._build_request(obs, session_state)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.engine_client.step, request)
        return self._extract_actions(result)

    def _build_request(self, obs: dict, session_state: Any = None) -> Any:
        """Convert OpenPI observation dict to engine request.

        Override for different engine types. Default builds OmniDiffusionRequest.
        """
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        assert isinstance(self.engine_client, DiffusionEngine), (
            f"Default _build_request requires DiffusionEngine, "
            f"got {type(self.engine_client).__name__}. "
            f"Override _build_request() for other engine types."
        )

        prompt = obs.get("prompt", "")
        session_id = obs.get("session_id", "default")
        needs_reset = session_state.needs_reset if session_state else True

        extra_args = {
            "reset": needs_reset,
            "session_id": session_id,
            "images": {
                "exterior_image_1_left": obs.get("observation/exterior_image_0_left"),
                "exterior_image_2_left": obs.get("observation/exterior_image_1_left"),
                "wrist_image_left": obs.get("observation/wrist_image_left"),
            },
            "state": {
                "joint_position": obs.get("observation/joint_position"),
                "gripper_position": obs.get("observation/gripper_position"),
            },
        }

        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[f"robot-{session_id}"],
        )

    def _extract_actions(self, result: Any) -> np.ndarray:
        """Extract action ndarray from engine result."""
        if hasattr(result, "__iter__"):
            result = list(result)
            if result:
                result = result[0]

        if hasattr(result, "custom_output"):
            actions = result.custom_output.get("actions")
            if actions is not None:
                if hasattr(actions, "numpy"):
                    return actions.numpy()
                return np.asarray(actions)

        logger.warning("No actions in result, returning zeros")
        return np.zeros((24, 8), dtype=np.float32)
