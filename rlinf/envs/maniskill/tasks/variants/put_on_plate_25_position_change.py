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
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


@register_env(
    "PutOnPlateInScene25PositionChangeTo-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25PositionChange(PutOnPlateInScene25MainV3):
    can_change_position: bool
    change_position_timestep = 5

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

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

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                for k, grid_pos_3 in enumerate(grid_pos):
                    if (
                        np.linalg.norm(grid_pos_1 - grid_pos_2) > 0.070
                        and np.linalg.norm(grid_pos_3 - grid_pos_2) > 0.070
                        and np.linalg.norm(grid_pos_1 - grid_pos_3) > 0.15
                    ):
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 0.95),  # carrot
                                    np.append(grid_pos_2, 0.92),  # plate
                                    np.append(grid_pos_3, 1.0),  # extra carrot
                                ]
                            )
                        )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")

        self.can_change_position = False

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            self.can_change_position = False
        elif self.obj_set == "test":
            self.can_change_position = True
        elif self.obj_set == "all":
            self.can_change_position = True
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
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2

    def evaluate(self, success_require_src_completely_on_target=True):
        if (
            self.can_change_position
            and self.elapsed_steps[0].item() == self.change_position_timestep
        ):
            b = self.num_envs

            xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
            quat_configs = torch.tensor(self.quat_configs, device=self.device)

            for idx, name in enumerate(self.model_db_carrot):
                p_reset = (
                    torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device)
                    .reshape(1, -1)
                    .repeat(b, 1)
                )  # [b, 3]
                is_select = self.select_carrot_ids == idx  # [b]
                p_select = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
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

            # self._settle(0.5)

        return super().evaluate(success_require_src_completely_on_target)
