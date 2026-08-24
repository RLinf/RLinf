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

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


def _default_compliance_param() -> dict[str, float]:
    return {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }


def _default_precision_param() -> dict[str, float]:
    return {
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


@dataclass
class PhysicalAgentFrankaConfig(FrankaRobotConfig):
    """Single Franka robot contract shared by collection and deployment."""

    task_description: str = "grasp the target object"
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.65875577, 0.04438992, 0.07458871, 2.94857731, -0.0676843, -0.18130717]
        )
    )
    reset_ee_pose: np.ndarray | None = None
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0
    clip_x_range: float = 0.5
    clip_y_range: float = 0.5
    clip_z_range_low: float = 0.04
    clip_z_range_high: float = 0.2
    clip_roll_pitch_range: float = 0.01
    clip_rz_range: float = 1.58
    derive_safety_box_from_target: bool = True
    enable_random_reset: bool = False
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.1, 1.0]))
    compliance_param: dict[str, float] = field(default_factory=dict)
    precision_param: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.camera_names is not None:
            self.camera_names = {
                str(serial): str(camera_name)
                for serial, camera_name in self.camera_names.items()
            }
        if self.camera_crop_regions is not None:
            self.camera_crop_regions = {
                str(serial): crop_region
                for serial, crop_region in self.camera_crop_regions.items()
            }

        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64)
        if self.reset_ee_pose is None:
            self.reset_ee_pose = self.target_ee_pose + np.array(
                [0.0, 0.0, self.clip_z_range_high, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
        else:
            self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64)
        self.reward_threshold = np.asarray(self.reward_threshold, dtype=np.float64)
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        self.ee_pose_limit_min = np.asarray(self.ee_pose_limit_min, dtype=np.float64)
        self.ee_pose_limit_max = np.asarray(self.ee_pose_limit_max, dtype=np.float64)
        self.hand_target_state = np.asarray(self.hand_target_state, dtype=np.float64)
        self.hand_reset_state = np.asarray(self.hand_reset_state, dtype=np.float64)

        compliance = _default_compliance_param()
        compliance.update(dict(self.compliance_param or {}))
        self.compliance_param = compliance
        precision = _default_precision_param()
        precision.update(dict(self.precision_param or {}))
        self.precision_param = precision

        if self.derive_safety_box_from_target:
            self.ee_pose_limit_min = np.array(
                [
                    self.target_ee_pose[0] - self.clip_x_range,
                    self.target_ee_pose[1] - self.clip_y_range,
                    self.target_ee_pose[2] - self.clip_z_range_low,
                    self.target_ee_pose[3] - self.clip_roll_pitch_range,
                    self.target_ee_pose[4] - self.clip_roll_pitch_range,
                    self.target_ee_pose[5] - self.clip_rz_range,
                ],
                dtype=np.float64,
            )
            self.ee_pose_limit_max = np.array(
                [
                    self.target_ee_pose[0] + self.clip_x_range,
                    self.target_ee_pose[1] + self.clip_y_range,
                    self.target_ee_pose[2] + self.clip_z_range_high,
                    self.target_ee_pose[3] + self.clip_roll_pitch_range,
                    self.target_ee_pose[4] + self.clip_roll_pitch_range,
                    self.target_ee_pose[5] + self.clip_rz_range,
                ],
                dtype=np.float64,
            )


class PhysicalAgentFrankaEnv(FrankaEnv):
    """FrankaEnv variant used as the RPent real-robot contract."""

    CONFIG_CLS = PhysicalAgentFrankaConfig

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Lift away from the workspace before moving to the reset pose."""
        self._end_effector_action(np.array([-1.0]))
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        self._franka_state = self._controller.get_state().wait()[0]

        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)
        super().go_to_rest(joint_reset)
