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

from dataclasses import dataclass

import numpy as np

from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig


@dataclass
class ButtonEnvConfig(Turtle2RobotConfig):
    random_xy_range: float = 0.05
    random_z_range_low: float = -0.005
    random_z_range_high: float = 0.1
    random_rz_range: float = np.pi / 9
    enable_random_reset: bool = True
    add_gripper_penalty: bool = False

    def __post_init__(self):
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = self.target_ee_pose + np.array(
            [
                [0.0, 0.0, self.random_z_range_high, 0.0, 0.0, 0.0],
                [0.0, 0.0, self.random_z_range_high, 0.0, 0.0, 0.0],
            ]
        )
        self.reward_threshold = np.array([0.015, 0.015, 0.01, 0.15, 0.15, 0.15])
        self.action_scale = np.array([0.01, 0.05, 0.0])  # remain the gripper close

        self.ee_pose_limit_min = self.target_ee_pose.copy()
        self.ee_pose_limit_min[:, 0] -= self.random_xy_range
        self.ee_pose_limit_min[:, 1] -= self.random_xy_range
        self.ee_pose_limit_min[:, 2] -= self.random_z_range_low
        self.ee_pose_limit_min[:, 3] -= self.random_rz_range
        self.ee_pose_limit_min[:, 4] -= self.random_rz_range
        self.ee_pose_limit_min[:, 5] -= self.random_rz_range

        self.ee_pose_limit_max = self.target_ee_pose.copy()
        self.ee_pose_limit_max[:, 0] += self.random_xy_range
        self.ee_pose_limit_max[:, 1] += self.random_xy_range
        self.ee_pose_limit_max[:, 2] += self.random_z_range_high
        self.ee_pose_limit_max[:, 3] += self.random_rz_range
        self.ee_pose_limit_max[:, 4] += self.random_rz_range
        self.ee_pose_limit_max[:, 5] += self.random_rz_range


class ButtonEnv(Turtle2Env):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        # Update config according to current env
        config = ButtonEnvConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "Press the button with the end-effector."
