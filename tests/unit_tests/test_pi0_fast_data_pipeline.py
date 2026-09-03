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

import pytest
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


def _uint8_obs(main_images):
    return {
        "main_images": main_images,
        "states": torch.zeros(main_images.shape[0], 8),
        "task_descriptions": ["pick up the object"] * main_images.shape[0],
    }


def test_uint8_images_are_rescaled_to_unit_range():
    images = torch.full((1, 4, 4, 3), 255, dtype=torch.uint8)

    batch = build_lerobot_batch_from_env_obs(_uint8_obs(images))

    assert torch.allclose(batch["observation.images.image"], torch.ones(1, 3, 4, 4))


def test_near_black_uint8_images_are_still_rescaled():
    # A frame whose brightest pixel is 1 must become 1/255, not stay at 1.0.
    # Scaling by observed maximum would silently skip this frame.
    images = torch.zeros(1, 4, 4, 3, dtype=torch.uint8)
    images[0, 0, 0, 0] = 1

    batch = build_lerobot_batch_from_env_obs(_uint8_obs(images))

    assert torch.allclose(
        batch["observation.images.image"].max(), torch.tensor(1.0 / 255.0)
    )


def test_float_images_are_left_untouched():
    images = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)

    batch = build_lerobot_batch_from_env_obs(_uint8_obs(images))

    assert torch.allclose(
        batch["observation.images.image"], torch.full((1, 3, 4, 4), 0.5)
    )


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, float("inf"), float("nan")],
)
def test_float_images_must_be_finite_and_normalized(value):
    images = torch.full((1, 4, 4, 3), value, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        build_lerobot_batch_from_env_obs(_uint8_obs(images))
