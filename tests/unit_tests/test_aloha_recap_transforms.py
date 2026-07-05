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

import numpy as np
import pytest


def test_value_dataset_builds_aloha_transform() -> None:
    pytest.importorskip("openpi")

    from rlinf.data.datasets.recap.value_dataset import ValueDataset

    transform = ValueDataset._build_transform(
        robot_type="aloha",
        model_type="pi05",
        action_dim=14,
        default_prompt="Assemble a sandwich.",
    )
    sample = {
        "observation.images.cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.ones((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
        "observation.state": np.zeros((14,), dtype=np.float32),
        "action": np.zeros((10, 14), dtype=np.float32),
        "prompt": "Assemble a sandwich.",
    }

    transformed = transform(sample)

    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape[-1] == 14
    assert transformed["prompt"] == "Assemble a sandwich."


def test_checkpoint_utils_builds_aloha_input_transforms() -> None:
    pytest.importorskip("openpi")

    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        build_input_transforms,
    )

    transforms = build_input_transforms(
        env_type="aloha",
        model_type="pi05",
        action_dim=14,
        default_prompt="Assemble a sandwich.",
        norm_stats=None,
        use_quantile_norm=True,
    )

    names = [type(transform).__name__ for transform in transforms]
    assert names == ["InjectDefaultPrompt", "AlohaInputs", "PadStatesAndActions"]
