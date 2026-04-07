# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DROID dataset transform.

DROID uses 1-indexed exterior cameras, 3 views total (OXE_DROID embodiment).
Stitching layout (same as RoboArena — both are OXE_DROID):
    ┌─────────────────────────┐
    │   wrist (2x width)      │  ← pixel-repeat along width
    ├────────────┬────────────┤
    │  left ext  │ right ext  │
    └────────────┴────────────┘

Source: dreamzero_cotrain.py _prepare_video() L326-349
"""

from __future__ import annotations

import numpy as np

from vllm_omni.entrypoints.openai.realtime.robot.transform.base import (
    RobotPolicyTransform,
    register_transform,
)

# OXE_DROID text template
# Source: dreamzero_cotrain.py collate() ~L104-112
_DROID_TEMPLATE = (
    "A multi-view video shows that a robot {instruction} "
    "The video is split into three views: "
    "The top view shows the camera view from the robot's wrist, "
    "the bottom-left view shows the camera view from the left exterior camera, "
    "and the bottom-right view shows the camera view from the right exterior camera. "
    "During training, one of the two bottom exterior views may be a black screen "
    "(dropped view). The robot {instruction}"
)


class DroidTransform(RobotPolicyTransform):
    """Transform for DROID dataset (OXE_DROID embodiment).

    DROID observation keys (1-indexed exterior cameras):
        observation/exterior_image_1_left  → left exterior
        observation/exterior_image_2_left  → right exterior
        observation/wrist_image_left       → wrist
    """

    IMAGE_KEY_MAP = {
        "observation/exterior_image_1_left": "images/exterior_0",
        "observation/exterior_image_2_left": "images/exterior_1",
        "observation/wrist_image_left": "images/wrist",
    }
    EMBODIMENT_NAME = "oxe_droid"
    ACTION_DIM = 8  # 7 joint + 1 gripper

    def _stitch_views(self, images: dict[str, np.ndarray]) -> np.ndarray:
        """OXE_DROID 2x2 stitching: wrist top (2x wide), exteriors bottom.
        Source: dreamzero_cotrain.py _prepare_video() L326-349
        """
        left_ext = images.get("images/exterior_0")
        right_ext = images.get("images/exterior_1")
        wrist = images.get("images/wrist")

        # Ensure 4D: (T, H, W, C)
        def ensure_4d(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return arr if arr.ndim == 4 else arr[np.newaxis]

        left_ext = ensure_4d(left_ext)
        right_ext = ensure_4d(right_ext)
        wrist = ensure_4d(wrist)

        # Determine shape from first available view
        ref = next((v for v in [wrist, left_ext, right_ext] if v is not None), None)
        if ref is None:
            return np.zeros((1, 360, 640, 3), dtype=np.uint8)

        t, h, w, c = ref.shape
        out = np.zeros((t, 2 * h, 2 * w, c), dtype=ref.dtype)  # (T, 2H, 2W, C)

        # Top row: wrist repeated 2x along width                  # L341
        if wrist is not None:
            wrist_wide = np.repeat(wrist, 2, axis=2)              # (T, H, 2W, C)
            out[:, :h, :] = wrist_wide

        # Bottom row: left exterior | right exterior               # L346-348
        if left_ext is not None:
            out[:, h:, :w] = left_ext
        if right_ext is not None:
            out[:, h:, w:] = right_ext

        return out

    def _language_template(self, prompt: str) -> str:
        """OXE_DROID language template.
        Source: dreamzero_cotrain.py collate() ~L104-112
        """
        instruction = prompt.lower() if prompt else "perform a task."
        return _DROID_TEMPLATE.format(instruction=instruction)

    def _extract_raw_state(self, obs: dict) -> np.ndarray:
        """OXE_DROID state: 7 joint + 1 gripper = 8 dims.
        Source: dreamzero_cotrain.py _prepare_state() L436-467
        """
        parts = []
        if "observation/joint_position" in obs:
            parts.append(np.asarray(obs["observation/joint_position"], dtype=np.float64).flatten())
        if "observation/gripper_position" in obs:
            parts.append(np.asarray(obs["observation/gripper_position"], dtype=np.float64).flatten())
        if parts:
            return np.concatenate(parts)
        return np.zeros(8, dtype=np.float64)


register_transform("droid", DroidTransform())
