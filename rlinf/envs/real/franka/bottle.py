# Copyright 2025 The RLinf Authors.
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

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from .base import FrankaEnv, FrankaRobotConfig, compliance


@dataclass
class BottleConfig(FrankaRobotConfig):
    task_description: str = "screw the bottle cap onto the bottle"
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.01
    clip_x_range: float = 0.01
    clip_y_range: float = 0.01
    clip_z_range_low: float = 0.001
    clip_z_range_high: float = 0.02
    random_rz_range: float = np.pi / 6
    clip_rz_range: float = np.pi / 6
    enable_random_reset: bool = True
    enable_gripper_penalty: bool = False
    step_frequency: float = 5.0

    def __post_init__(self) -> None:
        self.compliance_param = compliance(
            rotational_clip_neg_z=0.5,
            rotational_clip_z=0.5,
            translational_clip_neg_x=0.001,
            translational_clip_neg_y=0.001,
            translational_clip_neg_z=0.001,
            translational_clip_x=0.001,
            translational_clip_y=0.001,
            translational_clip_z=0.001,
        )
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = self.target_ee_pose + np.array(
            [0.0, 0.0, self.clip_z_range_high, 0.0, 0.0, 0.0]
        )
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array([0.01, 0.5, 1])
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - self.clip_x_range,
                self.target_ee_pose[1] - self.clip_y_range,
                self.target_ee_pose[2] - self.clip_z_range_low,
                self.target_ee_pose[3] - 0.01,
                self.target_ee_pose[4] - 0.01,
                self.target_ee_pose[5] - self.clip_rz_range,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + self.clip_x_range,
                self.target_ee_pose[1] + self.clip_y_range,
                self.target_ee_pose[2] + self.clip_z_range_high,
                self.target_ee_pose[3] + 0.01,
                self.target_ee_pose[4] + 0.01,
                self.target_ee_pose[5] + self.clip_rz_range,
            ]
        )


class BottleEnv(FrankaEnv):
    CONFIG_CLS = BottleConfig

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Lift clear of the object before moving to the base rest pose."""
        self._end_effector_action(np.array([1.0]))
        self._franka_state = self._read_robot()
        self._move_action(self._franka_state.tcp_pose)

        self._franka_state = self._read_robot()
        # Lift clear of the slot before returning to rest.
        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.03
        time.sleep(5)
        self._interpolate_move(reset_pose, timeout=1)
        time.sleep(2)
        reset_pose[2] += 0.02
        self._interpolate_move(reset_pose, timeout=1)

        super().go_to_rest(joint_reset)
