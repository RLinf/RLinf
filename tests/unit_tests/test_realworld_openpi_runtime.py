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

"""Unit tests for realworld task registration and OpenPI runtime adapters."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


def _import_data_collect_config():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")

    from rlinf.envs.realworld.franka.tasks.data_collect_env import DataCollectConfig

    return DataCollectConfig


def test_data_collect_env_is_registered():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    import rlinf.envs.realworld.franka.tasks  # noqa: F401

    spec = gym.spec("DataCollectEnv-v1")
    assert spec.id == "DataCollectEnv-v1"
    assert spec.entry_point == "rlinf.envs.realworld.franka.tasks:DataCollectEnv"


def test_data_collect_config_computes_reset_pose_and_limits():
    DataCollectConfig = _import_data_collect_config()
    config = DataCollectConfig(
        target_ee_pose=[0.5, -0.1, 0.2, -3.14, 0.0, 0.1],
        random_xy_range=0.02,
        random_z_range_low=0.03,
        random_z_range_high=0.04,
        random_rz_range=0.5,
    )

    assert np.allclose(config.reset_ee_pose, [0.5, -0.1, 0.24, -3.14, 0.0, 0.1])
    assert np.allclose(config.action_scale, [1.0, 1.0, 1.0])
    assert np.allclose(
        config.ee_pose_limit_min,
        [0.3, -0.3, -0.1, -3.14 - 10 * np.pi / 6, -10 * np.pi / 6, -4.9],
    )
    assert np.allclose(
        config.ee_pose_limit_max,
        [0.7, 0.1, 0.6, -3.14 + 10 * np.pi / 6, 10 * np.pi / 6, 5.1],
    )


def test_data_collect_config_preserves_explicit_limits():
    DataCollectConfig = _import_data_collect_config()
    config = DataCollectConfig(
        ee_pose_limit_min=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ee_pose_limit_max=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    assert np.allclose(config.ee_pose_limit_min, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    assert np.allclose(config.ee_pose_limit_max, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def openpi_runtime():
    pytest.importorskip("openpi")
    import torch
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.policies.realworld_policy import (
        RealworldInputs,
        RealworldOutputs,
    )

    return SimpleNamespace(
        torch=torch,
        model_enum=openpi_model,
        get_openpi_config=get_openpi_config,
        RealworldInputs=RealworldInputs,
        RealworldOutputs=RealworldOutputs,
    )


def test_get_openpi_config_returns_realworld_runtime_config(openpi_runtime):
    config = openpi_runtime.get_openpi_config("pi0_realworld_pnp")

    assert config.name == "pi0_realworld_pnp"
    assert config.data.repo_id == "realworld_pnp"
    assert config.data.extra_delta_transform is True
    assert config.data.extra_image_keys == ("extra_image_0", "extra_image_1")
    assert config.data.state_indices == (4, 5, 6, 7, 8, 9, 0)
    assert config.data.pi0_slot_keys == (
        "observation/extra_image_0",
        "observation/image",
        "observation/extra_image_1",
    )


def test_get_openpi_config_unknown_name_suggests_realworld_config(openpi_runtime):
    with pytest.raises(ValueError, match="pi0_realworld_pnp"):
        openpi_runtime.get_openpi_config("pi0_realworld")


def test_get_openpi_config_overrides_do_not_mutate_registry(openpi_runtime):
    overridden = openpi_runtime.get_openpi_config(
        "pi0_realworld_pnp",
        model_path="/tmp/mock_model",
        data_kwargs={
            "repo_id": "custom_realworld",
            "state_indices": (0, 1, 2, 3, 4, 5, 6),
        },
        batch_size=8,
    )
    fresh = openpi_runtime.get_openpi_config("pi0_realworld_pnp")

    assert overridden.pytorch_weight_path == "/tmp/mock_model"
    assert overridden.batch_size == 8
    assert overridden.data.repo_id == "custom_realworld"
    assert overridden.data.state_indices == (0, 1, 2, 3, 4, 5, 6)
    assert fresh.pytorch_weight_path == "checkpoints/torch/pi0_base"
    assert fresh.batch_size != 8
    assert fresh.data.repo_id == "realworld_pnp"
    assert fresh.data.state_indices == (4, 5, 6, 7, 8, 9, 0)


def test_realworld_inputs_select_state_and_unpack_extra_views(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((4, 4, 3), 33, dtype=np.uint8)
    stacked_extra = np.stack([extra0, extra1], axis=0)
    state19 = np.arange(19, dtype=np.float32)

    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_view_images": stacked_extra,
            "observation/state": state19,
            "prompt": "pick up the duck and put it into the container",
        }
    )

    assert openpi_runtime.torch.equal(
        outputs["state"],
        openpi_runtime.torch.tensor(
            [4, 5, 6, 7, 8, 9, 0], dtype=openpi_runtime.torch.float32
        ),
    )
    assert np.array_equal(outputs["image"]["base_0_rgb"], extra0)
    assert np.array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    assert np.array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)
    assert outputs["prompt"] == "pick up the duck and put it into the container"


def test_realworld_inputs_accepts_legacy_singular_extra_view_key(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((4, 4, 3), 33, dtype=np.uint8)

    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_view_image": np.stack([extra0, extra1], axis=0),
            "observation/state": np.arange(19, dtype=np.float32),
            "prompt": "demo",
        }
    )

    assert np.array_equal(outputs["image"]["base_0_rgb"], extra0)
    assert np.array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    assert np.array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)


def test_realworld_inputs_accepts_direct_extra_image_keys(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((4, 4, 3), 33, dtype=np.uint8)

    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_image_0": extra0,
            "observation/extra_image_1": extra1,
            "observation/state": np.arange(19, dtype=np.float32),
        }
    )

    assert np.array_equal(outputs["image"]["base_0_rgb"], extra0)
    assert np.array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    assert np.array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)


def test_realworld_inputs_unpacks_batched_extra_views(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    main = np.full((2, 4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((2, 4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((2, 4, 4, 3), 33, dtype=np.uint8)

    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_view_images": np.stack([extra0, extra1], axis=1),
            "observation/state": np.arange(38, dtype=np.float32).reshape(2, 19),
        }
    )

    assert np.array_equal(outputs["image"]["base_0_rgb"], extra0)
    assert np.array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    assert np.array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)


def test_realworld_inputs_supports_sparse_slot_mapping_and_bytes_prompt(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=10,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
        state_indices=None,
        pi0_slot_keys=(None, "observation/image", None),
    )

    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    outputs = transform(
        {
            "observation/image": main,
            "observation/state": np.arange(7, dtype=np.float32),
            "prompt": b"demo",
        }
    )

    assert outputs["state"].shape[-1] == 10
    assert openpi_runtime.torch.equal(
        outputs["state"][..., :7],
        openpi_runtime.torch.tensor(
            [0, 1, 2, 3, 4, 5, 6], dtype=openpi_runtime.torch.float32
        ),
    )
    assert np.array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    assert np.array_equal(outputs["image"]["base_0_rgb"], np.zeros_like(main))
    assert np.array_equal(outputs["image"]["right_wrist_0_rgb"], np.zeros_like(main))
    assert outputs["image_mask"]["base_0_rgb"] == np.False_
    assert outputs["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert outputs["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert outputs["prompt"] == "demo"


def test_realworld_inputs_raises_when_no_images_are_provided(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    with pytest.raises(ValueError, match="At least one image must be provided"):
        transform({"observation/state": np.arange(19, dtype=np.float32)})


def test_realworld_inputs_rejects_bad_actions_shape(openpi_runtime):
    transform = openpi_runtime.RealworldInputs(
        action_dim=7,
        model_type=openpi_runtime.model_enum.ModelType.PI0,
    )

    with pytest.raises(AssertionError, match="Expected actions shape"):
        transform(
            {
                "observation/image": np.full((4, 4, 3), 11, dtype=np.uint8),
                "observation/state": np.arange(19, dtype=np.float32),
                "actions": np.arange(7, dtype=np.float32),
            }
        )


def test_realworld_outputs_returns_first_seven_action_dims(openpi_runtime):
    transform = openpi_runtime.RealworldOutputs()

    outputs = transform({"actions": np.arange(20, dtype=np.float32).reshape(2, 10)})

    assert outputs["actions"].shape == (2, 7)
    assert np.array_equal(
        outputs["actions"],
        np.array(
            [[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16]],
            dtype=np.float32,
        ),
    )
