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

import cv2
from mani_skill.utils import io_utils
from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    CARROT_DATASET_DIR,
    PutOnPlateInScene25MainV3,
)


@register_env(
    "PutOnPlateInScene25Single-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25Single(PutOnPlateInScene25MainV3):
    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        only_carrot_name = list(self.model_db_carrot.keys())[0]
        self.model_db_carrot = {
            k: v for k, v in self.model_db_carrot.items() if k == only_carrot_name
        }
        assert len(self.model_db_carrot) == 1

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {
            k: v for k, v in self.model_db_plate.items() if k == only_plate_name
        }
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )
        only_table_name = list(model_db_table.keys())[0]
        model_db_table = {
            k: v for k, v in model_db_table.items() if k == only_table_name
        }
        assert len(model_db_table) == 1

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(
                cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB),
                (self.overlay_images_hw[1], self.overlay_images_hw[0]),
            )
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(
                cv2.cvtColor(
                    cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB
                ),
                (self.overlay_texture_hw[1], self.overlay_texture_hw[0]),
            )
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"]
            for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 1
        assert len(self.overlay_textures_numpy) == 1
        assert len(self.overlay_mix_numpy) == 1

    @property
    def basic_obj_infos(self):
        lo = 1
        lo_offset = 0
        if self.obj_set == "train":
            lc = 1
            lc_offset = 0
        elif self.obj_set == "test":
            lc = 1
            lc_offset = 0
        elif self.obj_set == "all":
            lc = 1
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")
        lp_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2
