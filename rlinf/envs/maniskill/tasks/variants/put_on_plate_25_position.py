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

import numpy as np
import torch
from mani_skill.utils.registration import register_env
from transforms3d.euler import euler2quat

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


@register_env(
    "PutOnPlateInScene25Position-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25Position(PutOnPlateInScene25MainV3):
    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            l1 = self.xyz_configs_len1
            l1_offset = 0
            l2 = self.quat_configs_len1
            l2_offset = 0
        elif self.obj_set == "test":
            l1 = self.xyz_configs_len2
            l1_offset = self.xyz_configs_len1
            l2 = self.quat_configs_len2
            l2_offset = self.quat_configs_len1
        elif self.obj_set == "all":
            l1 = self.xyz_configs_len1 + self.xyz_configs_len2
            l1_offset = 0
            l2 = self.quat_configs_len1 + self.quat_configs_len2
            l2_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        lp_offset = 0
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l1_offset, l2, l2_offset

    @property
    def total_num_trials(self):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l1_offset, l2, l2_offset = (
            self.basic_obj_infos
        )
        ltt = lc * lp * lo * l1 * l2
        return ltt

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l1_offset, l2, l2_offset = (
            self.basic_obj_infos
        )
        self._reset_episode_idx(env_idx, self.total_num_trials, options)

        self.select_carrot_ids = (
            self.episode_id // (lp * lo * l1 * l2) + lc_offset
        )  # [b]
        self.select_plate_ids = (self.episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (self.episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (self.episode_id // l2) % l1 + l1_offset
        self.select_quat_ids = self.episode_id % l2 + l2_offset

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        # 1
        grid_pos = (
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.2],
                    [0.0, 0.4],
                    [0.0, 0.6],
                    [0.0, 0.8],
                    [0.0, 1.0],
                    [0.2, 0.0],
                    [0.2, 0.2],
                    [0.2, 0.4],
                    [0.2, 0.6],
                    [0.2, 0.8],
                    [0.2, 1.0],
                    [0.4, 0.0],
                    [0.4, 0.2],
                    [0.4, 0.4],
                    [0.4, 0.6],
                    [0.4, 0.8],
                    [0.4, 1.0],
                    [0.6, 0.0],
                    [0.6, 0.2],
                    [0.6, 0.4],
                    [0.6, 0.6],
                    [0.6, 0.8],
                    [0.6, 1.0],
                    [0.8, 0.0],
                    [0.8, 0.2],
                    [0.8, 0.4],
                    [0.8, 0.6],
                    [0.8, 0.8],
                    [0.8, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.2],
                    [1.0, 0.4],
                    [1.0, 0.6],
                    [1.0, 0.8],
                    [1.0, 1.0],
                ]
            )
            * 2
            - 1
        )  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs1 = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs1.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs1 = np.stack(xyz_configs1)

        quat_configs1 = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        # 2
        grid_pos = (
            np.array(
                [
                    [-0.2, -0.2],
                    [-0.2, 0.0],
                    [-0.2, 0.2],
                    [-0.2, 0.4],
                    [-0.2, 0.6],
                    [-0.2, 0.8],
                    [-0.2, 1.0],
                    [-0.2, 1.2],
                    [0.0, -0.2],
                    [0.0, 1.2],
                    [0.2, -0.2],
                    [0.2, 1.2],
                    [0.4, -0.2],
                    [0.4, 1.2],
                    [0.6, -0.2],
                    [0.6, 1.2],
                    [0.8, -0.2],
                    [0.8, 1.2],
                    [1.0, -0.2],
                    [1.0, 1.2],
                    [1.2, -0.2],
                    [1.2, 0.0],
                    [1.2, 0.2],
                    [1.2, 0.4],
                    [1.2, 0.6],
                    [1.2, 0.8],
                    [1.2, 1.0],
                    [1.2, 1.2],
                ]
            )
            * 2
            - 1
        )  # [28, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs2 = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs2.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs2 = np.stack(xyz_configs2)

        quat_configs2 = np.stack(
            [
                np.array([euler2quat(0, 0, -np.pi / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, -np.pi * 3 / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, -np.pi * 5 / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, -np.pi * 7 / 8), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs_len1 = xyz_configs1.shape[0]
        self.xyz_configs_len2 = xyz_configs2.shape[0]
        self.quat_configs_len1 = quat_configs1.shape[0]
        self.quat_configs_len2 = quat_configs2.shape[0]

        self.xyz_configs = np.concatenate([xyz_configs1, xyz_configs2], axis=0)
        self.quat_configs = np.concatenate([quat_configs1, quat_configs2], axis=0)

        assert (
            self.xyz_configs.shape[0] == self.xyz_configs_len1 + self.xyz_configs_len2
        )
        assert (
            self.quat_configs.shape[0]
            == self.quat_configs_len1 + self.quat_configs_len2
        )
