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
import queue
import time
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class BinEnvConfig(FrankaRobotConfig):
    random_x_range: float = 0.10
    random_y_range: float = 0.15
    random_z_range_high: float = 0.1
    random_z_range_low: float = 0.001
    random_rz_range: float = np.pi / 6

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )

    def __post_init__(self):
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.004,
            "translational_clip_y": 0.004,
            "translational_clip_z": 0.004,
            "translational_clip_neg_x": 0.004,
            "translational_clip_neg_y": 0.004,
            "translational_clip_neg_z": 0.004,
            "rotational_clip_x": 0.04,
            "rotational_clip_y": 0.04,
            "rotational_clip_z": 0.02,
            "rotational_clip_neg_x": 0.04,
            "rotational_clip_neg_y": 0.04,
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
        self.action_scale = np.array([0.03, 0.1, 1])
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - 0.01,
                self.target_ee_pose[1] - self.random_y_range,
                self.target_ee_pose[2] - self.random_z_range_low,
                self.target_ee_pose[3] - 0.01,
                self.target_ee_pose[4] - 0.01,
                self.target_ee_pose[5] - self.random_rz_range,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + 0.2,
                self.target_ee_pose[1] + self.random_y_range,
                self.target_ee_pose[2] + self.random_z_range_high,
                self.target_ee_pose[3] + 0.01,
                self.target_ee_pose[4] + 0.01,
                self.target_ee_pose[5] + self.random_rz_range,
            ]
        )


class FrankaBinRelocationEnv(FrankaEnv):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = BinEnvConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)
        self.task_id = 0  # 0 for forward task, 1 for backward task
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self.config.target_ee_pose[:3] - np.array([0.07, 0.03, 0.001]),
            self.config.target_ee_pose[:3] + np.array([0.07, 0.03, 0.04]),
            dtype=np.float64,
        )

    @property
    def task_description(self):
        return "bin relocation"

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

    def _clip_position_to_safety_box(self, pose):
        pose = super()._clip_position_to_safety_box(pose)
        # Clip xyz to inner box
        if self.inner_safety_box.contains(pose[:3]):
            pose[:3] = self.intersect_line_bbox(
                self._franka_state.tcp_pose[:3],
                pose[:3],
                self.inner_safety_box.low,
                self.inner_safety_box.high,
            )
        return pose

    def _crop_frame(self, name, image):
        """Crop realsense images to be a square."""
        return image[:, 80:560, :]

    def _get_camera_frames(self):
        images = {}
        display_images = {}
        for camera in self._cameras:
            try:
                rgb = camera.get_frame()
                cropped_rgb = self._crop_frame(camera.name, rgb)
                resized = cv2.resize(
                    cropped_rgb,
                    self.observation_space["frames"][camera.name].shape[:2][::-1],
                )
                images[camera.name] = resized[..., ::-1]
                display_images[camera.name] = resized
                if camera.name == "front":
                    display_images[camera.name + "_full"] = cv2.resize(
                        cropped_rgb, (480, 480)
                    )
                elif camera.name == "wrist_1":
                    display_images[camera.name + "_full"] = cropped_rgb
            except queue.Empty:
                time.sleep(5)
                camera.close()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_images)
        return images

    def task_graph(self, obs=None):
        if obs is None:
            return (self.task_id + 1) % 2

    def set_task_id(self, task_id):
        self.task_id = task_id

    def reset(self, joint_reset=False, **kwargs):
        if self.task_id == 0:
            self._reset_pose[1] = self.config.target_ee_pose[1] + 0.1
        elif self.task_id == 1:
            self._reset_pose[1] = self.config.target_ee_pose[1] - 0.1
        else:
            raise ValueError(f"Task id {self.task_id} should be 0 or 1")

        return super().reset(joint_reset)

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._gripper_action(1)
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        time.sleep(0.5)
        self._franka_state = self._controller.get_state().wait()[0]

        # Move up to clear the slot
        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)

        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
