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
from dataclasses import dataclass, field

import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class PegInsertionConfig(FrankaRobotConfig):
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.05
    random_z_range_low: float = 0.0
    random_z_range_high: float = 0.1
    random_rz_range: float = np.pi / 6
    enable_random_reset: bool = True
    add_gripper_penalty: bool = False

    def __post_init__(self):
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.003,
            "translational_clip_y": 0.003,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.003,
            "translational_clip_neg_y": 0.003,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.02,
            "rotational_clip_y": 0.02,
            "rotational_clip_z": 0.02,
            "rotational_clip_neg_x": 0.02,
            "rotational_clip_neg_y": 0.02,
            "rotational_clip_neg_z": 0.02,
            "rotational_Ki": 0,
        }
        self.precision_param = {
            "translational_stiffness": 3000,
            "translational_damping": 89,
            "rotational_stiffness": 300,
            "rotational_damping": 9,
            "translational_Ki": 0.1,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.05,
            "rotational_clip_y": 0.05,
            "rotational_clip_z": 0.05,
            "rotational_clip_neg_x": 0.05,
            "rotational_clip_neg_y": 0.05,
            "rotational_clip_neg_z": 0.05,
            "rotational_Ki": 0.1,
        }
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = self.target_ee_pose + np.array(
            [0.0, 0.0, self.random_z_range_high, 0.0, 0.0, 0.0]
        )
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array([0.02, 0.1, 1])
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - self.random_xy_range,
                self.target_ee_pose[1] - self.random_xy_range,
                self.target_ee_pose[2] - self.random_z_range_low,
                self.target_ee_pose[3] - 0.01,
                self.target_ee_pose[4] - 0.01,
                self.target_ee_pose[5] - self.random_rz_range,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + self.random_xy_range,
                self.target_ee_pose[1] + self.random_xy_range,
                self.target_ee_pose[2] + self.random_z_range_high,
                self.target_ee_pose[3] + 0.01,
                self.target_ee_pose[4] + 0.01,
                self.target_ee_pose[5] + self.random_rz_range,
            ]
        )


class PegInsertionEnv(FrankaEnv):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        # Update config according to current env
        config = PegInsertionConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "peg and insertion"

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._gripper_action(-1)
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        self._franka_state = self._controller.get_state().wait()[0]
        # Move up to clear the slot
        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)

        super().go_to_rest(joint_reset)
