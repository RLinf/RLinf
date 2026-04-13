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

import os

from .eva_vit import EVAVisionTowerLavis


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    image_processor = getattr(
        vision_tower_cfg,
        "image_processor",
        getattr(
            vision_tower_cfg,
            "image_processor",
            "./model_zoo/OpenAI/clip-vit-large-patch14",
        ),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)

    if not is_absolute_path_exists:
        raise ValueError(f"Not find vision tower: {vision_tower}")

    if "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(
            vision_tower, image_processor, args=vision_tower_cfg, **kwargs
        )
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
