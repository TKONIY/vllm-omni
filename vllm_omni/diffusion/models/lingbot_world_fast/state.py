# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_float32_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array, but got shape {array.shape}.")
    return array


@dataclass
class LingbotWorldFastControlChunk:
    """Normalized control chunk for one realtime turn."""

    poses: np.ndarray
    intrinsics: np.ndarray
    wasd_action: np.ndarray | None = None
    ijkl_action: np.ndarray | None = None
    control_type: str = "cam"

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


def normalize_lingbot_control_chunk(payload: dict[str, Any]) -> LingbotWorldFastControlChunk:
    poses = _to_float32_array(payload.get("poses"), ndim=3, name="poses")
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must have shape [F, 4, 4], but got {poses.shape}.")

    intrinsics = _to_float32_array(payload.get("intrinsics"), ndim=2, name="intrinsics")
    if intrinsics.shape[-1] != 4:
        raise ValueError(f"intrinsics must have shape [F, 4] or [1, 4], but got {intrinsics.shape}.")
    if intrinsics.shape[0] == 1:
        intrinsics = np.repeat(intrinsics, poses.shape[0], axis=0)
    if intrinsics.shape[0] != poses.shape[0]:
        raise ValueError(
            "intrinsics frame count must match poses frame count "
            f"({intrinsics.shape[0]} != {poses.shape[0]})."
        )

    wasd_action = None
    raw_wasd_action = payload.get("wasd_action", payload.get("action"))
    if raw_wasd_action is not None:
        wasd_action = _to_float32_array(raw_wasd_action, ndim=2, name="wasd_action")
        if wasd_action.shape[0] != poses.shape[0]:
            raise ValueError(
                "wasd_action frame count must match poses frame count "
                f"({wasd_action.shape[0]} != {poses.shape[0]})."
            )

    ijkl_action = None
    raw_ijkl_action = payload.get("ijkl_action")
    if raw_ijkl_action is not None:
        ijkl_action = _to_float32_array(raw_ijkl_action, ndim=2, name="ijkl_action")
        if ijkl_action.shape[0] != poses.shape[0]:
            raise ValueError(
                "ijkl_action frame count must match poses frame count "
                f"({ijkl_action.shape[0]} != {poses.shape[0]})."
            )

    control_type = "act" if wasd_action is not None else "cam"
    return LingbotWorldFastControlChunk(
        poses=poses,
        intrinsics=intrinsics,
        wasd_action=wasd_action,
        ijkl_action=ijkl_action,
        control_type=control_type,
    )
