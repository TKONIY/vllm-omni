# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base transform interface for robot policy serving.

Transforms convert dataset-specific observation formats into a unified
model_inputs format. They are stateless — all state lives in the pipeline's
State object.

Flow: raw obs (dataset format) → Transform → unified obs → State.accumulate → model
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RobotPolicyTransform:
    """Base class for dataset-specific observation transforms.

    Subclass and override transform_input/transform_output for each
    dataset (DROID, RoboArena, LIBERO, etc.).
    """

    def transform_input(self, obs: dict) -> dict:
        """Map dataset-specific keys to unified model_inputs format.

        Unified format:
            images/exterior_0: ndarray (H,W,3) or (T,H,W,3)
            images/exterior_1: ndarray (H,W,3) or (T,H,W,3)  [optional]
            images/wrist: ndarray (H,W,3) or (T,H,W,3)       [optional]
            state/joint_position: ndarray (N,)
            state/gripper_position: ndarray (N,)
            prompt: str

        Default: identity (pass through as-is).
        """
        return obs

    def transform_output(self, result: Any) -> np.ndarray:
        """Convert model output to action ndarray (N, action_dim).

        Default: extract from DiffusionOutput.custom_output["actions"].
        """
        if hasattr(result, "custom_output"):
            actions = result.custom_output.get("actions")
            if actions is not None:
                return np.asarray(actions, dtype=np.float32)
        return np.zeros((24, 8), dtype=np.float32)


# Transform registry — keyed by embodiment/dataset name
TRANSFORMS: dict[str, RobotPolicyTransform] = {}


def register_transform(name: str, transform: RobotPolicyTransform) -> None:
    TRANSFORMS[name] = transform


def get_transform(name: str) -> RobotPolicyTransform:
    if name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform '{name}'. Available: {list(TRANSFORMS.keys())}"
        )
    return TRANSFORMS[name]
