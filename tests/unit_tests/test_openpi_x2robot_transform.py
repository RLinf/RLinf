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

import importlib

import numpy as np
import pytest


def _openpi_symbols():
    pytest.importorskip("openpi")
    pytest.importorskip("jax")
    pytest.importorskip("openpi.models.pi0_config")
    pytest.importorskip("openpi.models_pytorch.pi0_pytorch")

    action_model = importlib.import_module(
        "rlinf.models.embodiment.openpi.openpi_action_model"
    )
    x2robot_policy = importlib.import_module(
        "rlinf.models.embodiment.openpi.policies.x2robot_policy"
    )

    return (
        action_model.OpenPi0Config,
        action_model.OpenPi0ForRLActionPrediction,
        x2robot_policy.X2RobotInputs,
        x2robot_policy.X2RobotOutputs,
    )


def _new_x2robot_openpi_model(openpi_symbols):
    OpenPi0Config, OpenPi0ForRLActionPrediction, _, _ = openpi_symbols
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    object.__setattr__(
        model, "config", OpenPi0Config(config_name="pi0_turtle2_x2robot_s2s")
    )
    return model


def _assert_x2robot_rollout_transform(env_obs, face, left, right):
    openpi_symbols = _openpi_symbols()
    _, _, X2RobotInputs, X2RobotOutputs = openpi_symbols
    model = _new_x2robot_openpi_model(openpi_symbols)
    processed = model.obs_processor(env_obs)
    assert set(processed["images"]) == {
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    }
    np.testing.assert_array_equal(
        np.asarray(processed["images"]["left_wrist_view"]), np.asarray(left)
    )
    np.testing.assert_array_equal(
        np.asarray(processed["images"]["face_view"]), np.asarray(face)
    )
    np.testing.assert_array_equal(
        np.asarray(processed["images"]["right_wrist_view"]), np.asarray(right)
    )
    np.testing.assert_array_equal(
        np.asarray(processed["state"]), np.asarray(env_obs["states"])
    )
    np.testing.assert_array_equal(processed["prompt"], env_obs["task_descriptions"])

    inputs = X2RobotInputs(action_dim=14)(processed)
    assert inputs["state"].shape == (face.shape[0], 14)
    assert inputs["image"]["base_0_rgb"].shape == face.shape
    assert inputs["image"]["left_wrist_0_rgb"].shape == left.shape
    assert inputs["image"]["right_wrist_0_rgb"].shape == right.shape
    assert inputs["image_mask"]["base_0_rgb"]
    assert inputs["image_mask"]["left_wrist_0_rgb"]
    assert inputs["image_mask"]["right_wrist_0_rgb"]

    padded_actions = np.zeros((30, 20), dtype=np.float32)
    outputs = X2RobotOutputs(action_dim=14)({"actions": padded_actions})
    assert outputs["actions"].shape == (30, 14)


def test_openpi_x2robot_rollout_obs_transform_smoke():
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

    _assert_x2robot_rollout_transform(env_obs, face, left, right)


def test_openpi_x2robot_rollout_obs_transform_accepts_cpu_torch_tensors():
    torch = pytest.importorskip("torch")
    batch = 2
    face = torch.full((batch, 8, 8, 3), 10, dtype=torch.uint8)
    left = torch.full((batch, 8, 8, 3), 20, dtype=torch.uint8)
    right = torch.full((batch, 8, 8, 3), 30, dtype=torch.uint8)
    env_obs = {
        "main_images": face,
        "extra_view_images": torch.stack([left, right], dim=1),
        "states": torch.arange(batch * 14, dtype=torch.float32).reshape(batch, 14),
        "task_descriptions": np.asarray(["fold the towel", "fold the towel"]),
    }

    _assert_x2robot_rollout_transform(env_obs, face, left, right)
