# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DROID dataset transform.

DROID observation keys (from OpenPI droid_policy.py):
    observation/exterior_image_1_left  → images/exterior_0    (note: 1-indexed in DROID)
    observation/wrist_image_left       → images/wrist
    observation/joint_position         → state/joint_position
    observation/gripper_position       → state/gripper_position
    prompt                             → prompt
"""

from __future__ import annotations

import numpy as np

from vllm_omni.entrypoints.openai.realtime.robot.transform.base import (
    RobotPolicyTransform,
    register_transform,
)


class DroidTransform(RobotPolicyTransform):

    def transform_input(self, obs: dict) -> dict:
        unified: dict = {}

        # Images: DROID uses 1-indexed exterior camera
        if "observation/exterior_image_1_left" in obs:
            unified["images/exterior_0"] = np.asarray(obs["observation/exterior_image_1_left"])
        if "observation/wrist_image_left" in obs:
            unified["images/wrist"] = np.asarray(obs["observation/wrist_image_left"])

        # State
        if "observation/joint_position" in obs:
            unified["state/joint_position"] = np.asarray(obs["observation/joint_position"], dtype=np.float64)
        if "observation/gripper_position" in obs:
            unified["state/gripper_position"] = np.asarray(obs["observation/gripper_position"], dtype=np.float64)

        # Prompt
        if "prompt" in obs:
            unified["prompt"] = obs["prompt"]

        # Pass through session_id if present
        if "session_id" in obs:
            unified["session_id"] = obs["session_id"]

        return unified


register_transform("droid", DroidTransform())
