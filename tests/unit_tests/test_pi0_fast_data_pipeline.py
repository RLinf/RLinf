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

import torch

from rlinf.models.embodiment.pi0_fast.data_pipeline import (
    build_lerobot_batch_from_env_obs,
)


def test_build_lerobot_batch_maps_libero_obs():
    env_obs = {
        "main_images": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["pick up the object", "open the drawer"],
    }

    batch = build_lerobot_batch_from_env_obs(env_obs, image_size=224)

    assert batch["observation.images.image"].shape == (2, 3, 224, 224)
    assert batch["observation.images.image2"].shape == (2, 3, 224, 224)
    assert batch["observation.state"].shape == (2, 8)
    assert batch["task"] == ["pick up the object", "open the drawer"]
