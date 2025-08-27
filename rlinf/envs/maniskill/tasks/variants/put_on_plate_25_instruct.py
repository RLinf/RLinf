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

import torch
from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


# Language
@register_env(
    "PutOnPlateInScene25Instruct-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25Instruct(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            le = 1
            le_offset = 0
        elif self.obj_set == "test":
            le = 16
            le_offset = 1
        elif self.obj_set == "all":
            le = 17
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
        self.select_plate_ids = (self.episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (self.episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (self.episode_id // l2) % l1
        self.select_quat_ids = self.episode_id % l2

    def get_language_instruction(self):
        templates = [
            "put $C$ on $P$",
            "Place the $C$ on the $P$",
            "set $C$ on $P$",
            "move the $C$ to the $P$",
            "Take the $C$ and put it on the $P$",
            "pick up $C$ and set it down on $P$",
            "please put the $C$ on the $P$",
            "Put $C$ onto $P$.",
            "place the $C$ onto the $P$ surface",
            "Make sure $C$ is on $P$.",
            "on the $P$, put the $C$",
            "put the $C$ where the $P$ is",
            "Move the $C$ from the table to the $P$",
            "Move $C$ so it’s on $P$.",
            "Can you put $C$ on $P$?",
            "$C$ on the $P$, please.",
            "the $C$ should be placed on the $P$.",  # test
            "Lay the $C$ down on the $P$.",
            "could you place $C$ over $P$",
            "position the $C$ atop the $P$",
            "Arrange for the $C$ to be resting on $P$.",
        ]
        assert len(templates) == 21
        temp_idx = self.select_extra_ids

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        instruct = []
        for idx in range(self.num_envs):
            carrot_name = self.model_db_carrot[select_carrot[idx]]["name"]
            plate_name = self.model_db_plate[select_plate[idx]]["name"]

            temp = templates[temp_idx[idx]]
            temp = temp.replace("$C$", carrot_name)
            temp = temp.replace("$P$", plate_name)
            instruct.append(temp)

            # instruct.append(f"put {carrot_name} on {plate_name}")

        return instruct
