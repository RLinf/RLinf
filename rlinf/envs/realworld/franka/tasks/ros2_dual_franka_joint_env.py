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

"""Main dual-Franka joint environment with a ROS 2 controller backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.franka.ros2_controller import (
    FR3_JOINT_LIMITS_LOWER,
    FR3_JOINT_LIMITS_UPPER,
    Ros2ControllerConfig,
    Ros2DualFrankaBackend,
    Ros2FrankaControllerProxy,
)

from .dual_franka_joint_env import DualFrankaJointEnv, DualFrankaJointRobotConfig


@dataclass
class Ros2DualFrankaJointRobotConfig(DualFrankaJointRobotConfig):
    joint_position_limits_lower: np.ndarray = field(
        default_factory=lambda: FR3_JOINT_LIMITS_LOWER.copy()
    )
    joint_position_limits_upper: np.ndarray = field(
        default_factory=lambda: FR3_JOINT_LIMITS_UPPER.copy()
    )
    ros_discovery_timeout: float = 2.0
    ros_wait_timeout: float = 30.0
    ros_state_max_age: float = 0.5
    controlled_motion_tolerance: float = 0.03
    controlled_motion_timeout: float = 180.0
    controlled_motion_stable_time: float = 0.5


class Ros2DualFrankaJointEnv(DualFrankaJointEnv):
    """Reuse the main env and wrappers; replace only the arm transport."""

    CONFIG_CLS = Ros2DualFrankaJointRobotConfig

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            self.close()
            raise

    def _init_action_obs_spaces(self) -> None:
        super()._init_action_obs_spaces()
        frames_space = self.observation_space["frames"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "proprio": gym.spaces.Box(
                            -np.inf, np.inf, shape=(16,), dtype=np.float32
                        )
                    }
                ),
                "frames": frames_space,
            }
        )

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self.observation_space.sample()
        proprio = np.concatenate(
            [
                self._left_state.arm_joint_position,
                np.asarray([self._left_state.gripper_position]),
                self._right_state.arm_joint_position,
                np.asarray([self._right_state.gripper_position]),
            ]
        ).astype(np.float32)
        return {
            "state": {"proprio": proprio},
            "frames": self._get_camera_frames(),
        }

    def _calc_step_reward(self, is_gripper_effective: list[bool]) -> float:
        """Return the placeholder reward used by manual data collection."""
        return 0.0

    def _setup_hardware(self) -> None:
        self._resolve_hw_overrides()
        required = {
            "left_robot_ip": self.config.left_robot_ip,
            "right_robot_ip": self.config.right_robot_ip,
            "left_gripper_connection": self.config.left_gripper_connection,
            "right_gripper_connection": self.config.right_gripper_connection,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing ROS 2 dual-Franka hardware fields: {missing}")

        self._ros2_backend = Ros2DualFrankaBackend(
            Ros2ControllerConfig(
                left_robot_ip=self.config.left_robot_ip,
                right_robot_ip=self.config.right_robot_ip,
                left_gripper_type=self.config.left_gripper_type
                or self._DEFAULT_GRIPPER_TYPE,
                right_gripper_type=self.config.right_gripper_type
                or self._DEFAULT_GRIPPER_TYPE,
                left_gripper_connection=self.config.left_gripper_connection,
                right_gripper_connection=self.config.right_gripper_connection,
                ros_discovery_timeout=self.config.ros_discovery_timeout,
                ros_wait_timeout=self.config.ros_wait_timeout,
                ros_state_max_age=self.config.ros_state_max_age,
                controlled_motion_tolerance=self.config.controlled_motion_tolerance,
                controlled_motion_timeout=self.config.controlled_motion_timeout,
                controlled_motion_stable_time=(
                    self.config.controlled_motion_stable_time
                ),
            )
        )
        self._left_ctrl = Ros2FrankaControllerProxy(self._ros2_backend, "left")
        self._right_ctrl = Ros2FrankaControllerProxy(self._ros2_backend, "right")

    def close(self) -> None:
        try:
            super().close()
        finally:
            if hasattr(self, "_ros2_backend"):
                self._ros2_backend.close()
