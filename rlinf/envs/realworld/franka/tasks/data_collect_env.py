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

from dataclasses import dataclass, field

import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class DataCollectConfig(FrankaRobotConfig):
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.05
    random_z_range_low: float = 0.1
    random_z_range_high: float = 0.1
    random_rz_range: float = np.pi / 6
    enable_random_reset: bool = False

    def __post_init__(self):
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.02,
            "translational_clip_y": 0.02,
            "translational_clip_z": 0.02,
            "translational_clip_neg_x": 0.02,
            "translational_clip_neg_y": 0.02,
            "translational_clip_neg_z": 0.02,
            "rotational_clip_x": 0.04,
            "rotational_clip_y": 0.04,
            "rotational_clip_z": 0.04,
            "rotational_clip_neg_x": 0.04,
            "rotational_clip_neg_y": 0.04,
            "rotational_clip_neg_z": 0.04,
            "rotational_Ki": 0,
        }
        self.precision_param = {
            "translational_stiffness": 3000,
            "translational_damping": 89,
            "rotational_stiffness": 300,
            "rotational_damping": 9,
            "translational_Ki": 0.1,
            "translational_clip_x": 0.02,
            "translational_clip_y": 0.02,
            "translational_clip_z": 0.02,
            "translational_clip_neg_x": 0.02,
            "translational_clip_neg_y": 0.02,
            "translational_clip_neg_z": 0.02,
            "rotational_clip_x": 0.1,
            "rotational_clip_y": 0.1,
            "rotational_clip_z": 0.1,
            "rotational_clip_neg_x": 0.1,
            "rotational_clip_neg_y": 0.1,
            "rotational_clip_neg_z": 0.1,
            "rotational_Ki": 0.1,
        }
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = self.target_ee_pose + np.array(
            [0.0, 0.0, self.random_z_range_high, 0.0, 0.0, 0.0]
        )
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array([1, 1, 1])

        # Allow explicit ee_pose_limit_{min,max} from YAML override_cfg.
        # If both are all-zero (the FrankaRobotConfig default), auto-compute
        # from range parameters instead.
        explicit_min = np.array(self.ee_pose_limit_min)
        explicit_max = np.array(self.ee_pose_limit_max)
        if np.any(explicit_min != 0) or np.any(explicit_max != 0):
            self.ee_pose_limit_min = explicit_min
            self.ee_pose_limit_max = explicit_max
        else:
            pos_weight = 10.0
            ori_weight = 10.0
            self.ee_pose_limit_min = np.array(
                [
                    self.target_ee_pose[0] - pos_weight * self.random_xy_range,
                    self.target_ee_pose[1] - pos_weight * self.random_xy_range,
                    self.target_ee_pose[2] - pos_weight * self.random_z_range_low,
                    self.target_ee_pose[3] - ori_weight * np.pi / 6,
                    self.target_ee_pose[4] - ori_weight * np.pi / 6,
                    self.target_ee_pose[5] - ori_weight * self.random_rz_range,
                ]
            )
            self.ee_pose_limit_max = np.array(
                [
                    self.target_ee_pose[0] + pos_weight * self.random_xy_range,
                    self.target_ee_pose[1] + pos_weight * self.random_xy_range,
                    self.target_ee_pose[2] + pos_weight * self.random_z_range_high,
                    self.target_ee_pose[3] + ori_weight * np.pi / 6,
                    self.target_ee_pose[4] + ori_weight * np.pi / 6,
                    self.target_ee_pose[5] + ori_weight * self.random_rz_range,
                ]
            )


class DataCollectEnv(FrankaEnv):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        # Update config according to current env
        self.config = DataCollectConfig(**override_cfg)
        super().__init__(self.config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "pick up the duck and put it into the container"

    def go_to_rest(self, joint_reset=False):
        super().go_to_rest(joint_reset)

    def get_tcp_pose(self):
        self._franka_state = self._controller.get_state().wait()[0]
        tcp_pose = self._franka_state.tcp_pose
        return tcp_pose

    def get_action_scale(self):
        return self.config.action_scale
