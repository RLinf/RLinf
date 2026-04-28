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

pytest.importorskip("openpi")
pytest.importorskip("jax")
pytest.importorskip("openpi.models.pi0_config")
pytest.importorskip("openpi.models_pytorch.pi0_pytorch")

from rlinf.models.embodiment.openpi.openpi_action_model import (  # noqa: E402
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies.x2robot_policy import (  # noqa: E402
    X2RobotInputs,
    X2RobotOutputs,
)


def test_openpi_x2robot_rollout_obs_transform_smoke():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    object.__setattr__(
        model, "config", OpenPi0Config(config_name="pi0_turtle2_x2robot_s2s")
    )

    batch = 2
    face = np.full((batch, 8, 8, 3), 10, dtype=np.uint8)
    left = np.full((batch, 8, 8, 3), 20, dtype=np.uint8)
    right = np.full((batch, 8, 8, 3), 30, dtype=np.uint8)
    env_obs = {
        "main_images": face,
        "extra_view_images": np.stack([left, right], axis=1),
        "states": np.arange(batch * 14, dtype=np.float32).reshape(batch, 14),
        "task_descriptions": np.asarray(["fold the towel", "fold the towel"]),
    }

    processed = model.obs_processor(env_obs)
    assert set(processed["images"]) == {
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    }
    np.testing.assert_array_equal(processed["images"]["left_wrist_view"], left)
    np.testing.assert_array_equal(processed["images"]["face_view"], face)
    np.testing.assert_array_equal(processed["images"]["right_wrist_view"], right)
    np.testing.assert_array_equal(processed["state"], env_obs["states"])
    np.testing.assert_array_equal(processed["prompt"], env_obs["task_descriptions"])

    inputs = X2RobotInputs(action_dim=14)(processed)
    assert inputs["state"].shape == (batch, 14)
    assert inputs["image"]["base_0_rgb"].shape == (batch, 8, 8, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (batch, 8, 8, 3)
    assert inputs["image"]["right_wrist_0_rgb"].shape == (batch, 8, 8, 3)
    assert inputs["image_mask"]["base_0_rgb"]
    assert inputs["image_mask"]["left_wrist_0_rgb"]
    assert inputs["image_mask"]["right_wrist_0_rgb"]

    padded_actions = np.zeros((30, 20), dtype=np.float32)
    outputs = X2RobotOutputs(action_dim=14)({"actions": padded_actions})
    assert outputs["actions"].shape == (30, 14)
