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

from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


@register_env(
    "PutOnPlateInScene25MainCarrot-v3",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25MainCarrotV3(PutOnPlateInScene25MainV3):
    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            lc = 16
            lc_offset = 0
        elif self.obj_set == "test":
            lc = 9
            lc_offset = 16
        elif self.obj_set == "all":
            lc = 25
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")

        lo = 1
        lo_offset = 0
        lp_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2


@register_env(
    "PutOnPlateInScene25Carrot-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25Carrot(PutOnPlateInScene25MainV3):
    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            lc = 16
            lc_offset = 0
        elif self.obj_set == "test":
            lc = 9
            lc_offset = 16
        elif self.obj_set == "all":
            lc = 25
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")

        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        lp_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2
