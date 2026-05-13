# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-arm Franka env with 16-D joint-space actions (absolute or delta).

Layout: [left_j1..j7, left_grip, right_j1..j7, right_grip].
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from ..dual_franka_env import DualFrankaEnv, DualFrankaRobotConfig
from ..franky_controller import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER

ACTION_DIM_PER_ARM = 8  # 7 joints + 1 gripper trigger


@dataclass
class DualFrankaJointRobotConfig(DualFrankaRobotConfig):
    """Config for :class:`DualFrankaJointEnv`."""

    joint_position_limits_lower: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_LOWER.copy()
    )
    joint_position_limits_upper: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_UPPER.copy()
    )

    joint_action_mode: str = "absolute"  # "absolute" | "delta"
    joint_action_scale: float = 0.1  # rad per unit action (delta mode)

    # When True, a 1 kHz daemon (DualGelloJointIntervention) owns the controllers.
    teleop_direct_stream: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.joint_position_limits_lower = np.array(self.joint_position_limits_lower)
        self.joint_position_limits_upper = np.array(self.joint_position_limits_upper)


class DualFrankaJointEnv(DualFrankaEnv):
    """Dual-arm Franka env with 16-D joint-space actions."""

    CONFIG_CLS: type[DualFrankaJointRobotConfig] = DualFrankaJointRobotConfig

    PER_ARM_ACTION_DIM = ACTION_DIM_PER_ARM
    GRIPPER_IDX_IN_ARM = 7  # gripper slot after the 7 arm joints

    def _init_action_obs_spaces(self):
        self._cartesian_safety_boxes()

        if self.config.joint_action_mode == "absolute":
            arm_low = np.concatenate(
                [self.config.joint_position_limits_lower, np.array([-1.0])]
            )
            arm_high = np.concatenate(
                [self.config.joint_position_limits_upper, np.array([1.0])]
            )
            act_low = np.concatenate([arm_low, arm_low]).astype(np.float32)
            act_high = np.concatenate([arm_high, arm_high]).astype(np.float32)
        elif self.config.joint_action_mode == "delta":
            act_low = -np.ones(2 * ACTION_DIM_PER_ARM, dtype=np.float32)
            act_high = np.ones(2 * ACTION_DIM_PER_ARM, dtype=np.float32)
        else:
            raise ValueError(
                f"DualFrankaJointEnv.joint_action_mode must be 'absolute' or "
                f"'delta', got {self.config.joint_action_mode!r}"
            )
        self.action_space = gym.spaces.Box(act_low, act_high)

        self.observation_space = self._build_observation_space(
            joint_position_dim=2 * 7
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        del dt
        if self.config.teleop_direct_stream:
            return

        target_joints = []
        for arm in range(2):
            joint_action = actions[arm, :7]
            if self.config.joint_action_mode == "absolute":
                tj = joint_action.copy()
            else:  # delta
                tj = (
                    states[arm].arm_joint_position
                    + joint_action * self.config.joint_action_scale
                )
            tj = self._clip_joints_to_limits(tj)
            target_joints.append(tj)

        left_f = ctrls[0].move_joints(target_joints[0].astype(np.float32))
        right_f = ctrls[1].move_joints(target_joints[1].astype(np.float32))
        left_f.wait()
        right_f.wait()

    def _pace_between_action_and_state_read(self) -> bool:
        return not self.config.teleop_direct_stream

    def _clip_joints_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(
            q,
            self.config.joint_position_limits_lower,
            self.config.joint_position_limits_upper,
        )

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()

        state = {
            "tcp_pose": np.concatenate(
                [self._left_state.tcp_pose, self._right_state.tcp_pose]
            ),
            "tcp_vel": np.concatenate(
                [self._left_state.tcp_vel, self._right_state.tcp_vel]
            ),
            "joint_position": np.concatenate(
                [
                    self._left_state.arm_joint_position,
                    self._right_state.arm_joint_position,
                ]
            ),
            "joint_velocity": np.concatenate(
                [
                    self._left_state.arm_joint_velocity,
                    self._right_state.arm_joint_velocity,
                ]
            ),
            "gripper_position": np.array(
                [
                    self._left_state.gripper_position,
                    self._right_state.gripper_position,
                ],
                dtype=np.float32,
            ),
            "tcp_force": np.concatenate(
                [self._left_state.tcp_force, self._right_state.tcp_force]
            ),
            "tcp_torque": np.concatenate(
                [self._left_state.tcp_torque, self._right_state.tcp_torque]
            ),
        }
        return copy.deepcopy({"state": state, "frames": frames})
