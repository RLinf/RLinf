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
    "PutOnPlateInScene25Plate-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25Plate(PutOnPlateInScene25MainV3):
    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        assert len(self.model_db_plate) == 17

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

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
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            lp = 1
            lp_offset = 0
        elif self.obj_set == "test":
            lp = 16
            lp_offset = 1
        elif self.obj_set == "all":
            lp = 17
            lp_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")
        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2
