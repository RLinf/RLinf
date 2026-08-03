# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action helpers for dual-Franka policies."""

from __future__ import annotations

import numpy as np

from rlinf.utils.rot6d import quat_xyzw_to_rot6d

DUAL_FRANKA_TCP_ACTION_DIM_PER_ARM = 10
DUAL_FRANKA_TCP_ACTION_DIM = 2 * DUAL_FRANKA_TCP_ACTION_DIM_PER_ARM


def hold_right_arm_action(
    action: np.ndarray,
    right_tcp_pose: np.ndarray,
    right_gripper_open: bool,
) -> np.ndarray:
    """Replace the right-arm command with its current TCP/gripper state.

    Args:
        action: Dual-Franka TCP action with layout
            ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
        right_tcp_pose: Current right TCP pose ``[xyz, quat_xyzw]``.
        right_gripper_open: Whether the right gripper is currently open.

    Returns:
        A copy of ``action`` whose left-arm slice is unchanged and whose
        right-arm slice commands the current state.

    Raises:
        ValueError: If an input does not have the expected shape.
    """
    action_array = np.asarray(action)
    if action_array.shape != (DUAL_FRANKA_TCP_ACTION_DIM,):
        raise ValueError(
            "hold_right_arm_action expects a 20D Dual-Franka TCP action, "
            f"got shape {action_array.shape}."
        )

    tcp_pose = np.asarray(right_tcp_pose)
    if tcp_pose.shape != (7,):
        raise ValueError(
            "hold_right_arm_action expects right_tcp_pose with shape (7,), "
            f"got {tcp_pose.shape}."
        )

    held_action = action_array.copy()
    right_start = DUAL_FRANKA_TCP_ACTION_DIM_PER_ARM
    held_action[right_start : right_start + 3] = tcp_pose[:3]
    held_action[right_start + 3 : right_start + 9] = quat_xyzw_to_rot6d(tcp_pose[3:])
    held_action[right_start + 9] = 1.0 if right_gripper_open else -1.0
    return held_action
