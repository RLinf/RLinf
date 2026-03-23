# Copyright 2026 Ying-Chun Lee
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

"""Unit tests for OpenPI's YAM follower runtime helpers."""

import json

import torch

from rlinf.models.embodiment.openpi import (
    _load_hf_visual_feature_order,
    _normalize_openpi_state_dict_keys,
)
from rlinf.models.embodiment.openpi.openpi_action_model import (
    _build_pi05_yam_follower_image_obs,
)


def test_build_pi05_yam_follower_image_obs_respects_checkpoint_camera_order():
    top = torch.full((1, 2, 2, 3), 10, dtype=torch.uint8)
    left = torch.full((1, 2, 2, 3), 20, dtype=torch.uint8)
    right = torch.full((1, 2, 2, 3), 30, dtype=torch.uint8)
    env_obs = {
        "main_images": top,
        "wrist_images": torch.stack([left, right], dim=1),
    }

    remapped = _build_pi05_yam_follower_image_obs(
        env_obs, camera_order=("left", "top", "right")
    )

    assert torch.equal(remapped["observation/image"], left)
    assert torch.equal(remapped["observation/wrist_image"], top)
    assert torch.equal(remapped["observation/extra_view_image"], right)


def test_build_pi05_yam_follower_image_obs_uses_extra_view_for_right_camera():
    top = torch.full((1, 2, 2, 3), 10, dtype=torch.uint8)
    left = torch.full((1, 2, 2, 3), 20, dtype=torch.uint8)
    right = torch.full((1, 2, 2, 3), 30, dtype=torch.uint8)
    env_obs = {
        "main_images": top,
        "wrist_images": left[:, None, ...],
        "extra_view_images": right,
    }

    remapped = _build_pi05_yam_follower_image_obs(
        env_obs, camera_order=("top", "right", "left")
    )

    assert torch.equal(remapped["observation/image"], top)
    assert torch.equal(remapped["observation/wrist_image"], right)
    assert torch.equal(remapped["observation/extra_view_image"], left)


def test_load_hf_visual_feature_order_reads_input_feature_order(tmp_path):
    config = {
        "input_features": {
            "observation.images.left": {},
            "observation.images.top": {},
            "observation.images.right": {},
            "observation.state": {},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert _load_hf_visual_feature_order(str(tmp_path)) == ("left", "top", "right")


def test_normalize_openpi_state_dict_keys_strips_model_and_orig_mod_prefixes():
    normalized = _normalize_openpi_state_dict_keys(
        {
            "model._orig_mod.layer.weight": torch.tensor([1.0]),
            "model._orig_mod.layer.bias": torch.tensor([0.0]),
        }
    )

    assert sorted(normalized) == ["layer.bias", "layer.weight"]
