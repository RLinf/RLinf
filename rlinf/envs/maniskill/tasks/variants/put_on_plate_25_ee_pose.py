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

from itertools import product

import numpy as np
import torch
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


@register_env(
    "PutOnPlateInScene25EEPose-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25EEPose(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    def _generate_init_pose(self):
        super()._generate_init_pose()

        robot_qpos = []

        robot_qpos.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        p0 = [-0.15, -0.08, 0.0, 0.08, 0.15]
        p1 = [-0.15, -0.08, 0.0, 0.08, 0.15]
        p2 = [-0.15, -0.08, 0.0, 0.08, 0.15]
        p3 = [-0.15, -0.08, 0.0, 0.08, 0.15]
        p4 = [-0.15, -0.08, 0.0, 0.08, 0.15]
        p5 = [-0.6, -0.3, 0.0, 0.3, 0.6]
        for p0, p1, p2, p3, p4, p5 in product(p0, p1, p2, p3, p4, p5):
            p012sum = p1 + p2
            if abs(p012sum) > 0.25:
                continue
            robot_qpos.append(np.array([p0, p1, p2, p3, p4, p5, 0.0, 0.0]))

        robot_qpos = np.stack(robot_qpos)
        self.robot_qpos = robot_qpos

        print(f"robot_qpos: {robot_qpos.shape}")

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            le = 1
            le_offset = 0
        elif self.obj_set == "test":
            le = len(self.robot_qpos) - 1
            le_offset = 1
        elif self.obj_set == "all":
            le = len(self.robot_qpos)
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")
        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        lp_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)

        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset

    @property
    def total_num_trials(self):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = (
            self.basic_obj_infos
        )
        ltt = lc * le * lp * lo * l1 * l2
        return ltt

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = (
            self.basic_obj_infos
        )
        self._reset_episode_idx(env_idx, self.total_num_trials, options)

        self.select_carrot_ids = (
            self.episode_id // (le * lp * lo * l1 * l2) + lc_offset
        )  # [b]
        self.select_extra_ids = (
            self.episode_id // (lp * lo * l1 * l2)
        ) % le + le_offset  # [b]
        self.select_plate_ids = (self.episode_id // (lo * l1 * l2)) % lp + lp_offset
        self.select_overlay_ids = (self.episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (self.episode_id // l2) % l1
        self.select_quat_ids = self.episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack(
            [self.overlay_images_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_images = torch.tensor(
            overlay_images, device=self.device
        )  # [b, H, W, 3]
        overlay_textures = np.stack(
            [self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_textures = torch.tensor(
            overlay_textures, device=self.device
        )  # [b, H, W, 3]
        overlay_mix = np.array(
            [self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0],
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = (
                torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = (
                torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        # self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            pass
            # self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)

        robot_qpos = torch.tensor(self.robot_qpos, device=self.device)
        initial_qpos = torch.tensor(self.initial_qpos, device=self.device).reshape(
            1, -1
        )  # [1, 8]
        qpos = robot_qpos[self.select_extra_ids] + initial_qpos  # [b, 8]
        self.agent.reset(init_qpos=qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(carrot_actor)]
        )  # [b, 4]
        self.plate_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(plate_actor)]
        )  # [b, 4]
        corner_signs = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            device=self.device,
        )

        # carrot
        carrot_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_carrot]
        )  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.carrot_q_after_settle
        )  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(
            c_bbox_corners, c_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        c_rotated_bbox_size = (
            c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_plate]
        )  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.plate_q_after_settle
        )  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(
            p_bbox_corners, p_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        p_rotated_bbox_size = (
            p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self._reset_stats(env_idx)
