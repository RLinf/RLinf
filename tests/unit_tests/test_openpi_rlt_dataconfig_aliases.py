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

import dataclasses
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("torch")
pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.rlt_joint_policy import (
    RLTJointInputs,
    RLTJointOutputs,
)
from rlinf.models.embodiment.rlt_stage2.proprio import (
    resolve_proprio_dim,
)

ROOT = Path(__file__).resolve().parents[2]
MANISKILL_STAGE2_CONFIG = (
    ROOT / "examples/embodiment/config/rlt_stage2_maniskill_joint.yaml"
)


def _load_yaml_config(path: Path):
    return OmegaConf.load(path)


def _normalize_config(config):
    cfg_dict = dataclasses.asdict(config)
    cfg_dict.pop("name", None)
    return cfg_dict


def _uint8_image(shape):
    return (np.arange(np.prod(shape)).reshape(shape) % 256).astype(np.uint8)


def _canonicalize_raw_sample(cfg, raw_sample):
    canonical = {
        "observation/image": raw_sample[cfg.data.image_key],
        "observation/wrist_image": raw_sample[cfg.data.wrist_image_key],
        "observation/state": raw_sample[cfg.data.state_key],
        "actions": raw_sample[cfg.data.action_key],
    }
    if "prompt" in raw_sample:
        canonical["prompt"] = raw_sample["prompt"]
    if "task" in raw_sample:
        canonical["task"] = raw_sample["task"]
    return canonical


def _assert_rlt_train_config_contract(
    cfg,
    *,
    repo_id,
    image_key,
    wrist_image_key,
):
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is True
    assert cfg.data.repo_id == repo_id
    assert cfg.data.image_key == image_key
    assert cfg.data.wrist_image_key == wrist_image_key
    assert cfg.data.state_key == "state"
    assert cfg.data.action_key == "actions"
    assert cfg.data.extra_delta_transform is False


def _assert_rlt_policy_transform_contract(
    cfg,
    raw_sample,
    *,
    expected_state_dim,
    expected_base_image_shape,
    expected_wrist_image_shape,
):
    inputs_transform = RLTJointInputs(
        model_type=cfg.model.model_type,
        use_wrist_image=True,
    )
    outputs_transform = RLTJointOutputs(output_action_dim=8)

    transformed = inputs_transform(_canonicalize_raw_sample(cfg, raw_sample))
    actions = outputs_transform({"actions": raw_sample[cfg.data.action_key]})

    assert transformed["state"].shape == (expected_state_dim,)
    assert transformed["actions"].shape == raw_sample[cfg.data.action_key].shape
    assert actions["actions"].shape == (cfg.model.action_horizon, 8)
    np.testing.assert_array_equal(
        actions["actions"],
        raw_sample[cfg.data.action_key][:, :8],
    )

    assert transformed["image"]["base_0_rgb"].shape == expected_base_image_shape
    assert transformed["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(
        transformed["image"]["base_0_rgb"],
        raw_sample[cfg.data.image_key],
    )

    assert transformed["image"]["left_wrist_0_rgb"].shape == expected_wrist_image_shape
    assert transformed["image"]["left_wrist_0_rgb"].dtype == np.uint8
    expected_wrist = raw_sample[cfg.data.wrist_image_key]
    if np.issubdtype(expected_wrist.dtype, np.floating):
        expected_wrist = (255 * expected_wrist).astype(np.uint8)
    if expected_wrist.ndim == 3 and expected_wrist.shape[0] == 3:
        expected_wrist = np.transpose(expected_wrist, (1, 2, 0))
    np.testing.assert_array_equal(
        transformed["image"]["left_wrist_0_rgb"],
        expected_wrist,
    )

    assert transformed["image"]["right_wrist_0_rgb"].shape == expected_base_image_shape
    np.testing.assert_array_equal(
        transformed["image"]["right_wrist_0_rgb"],
        np.zeros(expected_base_image_shape, dtype=np.uint8),
    )
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert transformed["prompt"] == raw_sample["prompt"]


def _assert_stage2_dimension_contract(
    cfg,
    *,
    config_name,
    action_dim,
    action_horizon,
    num_images,
    proprio_dim,
    proprio_mode=None,
):
    assert cfg.actor.model.rlt_stage2.config_name == config_name
    assert cfg.actor.model.num_action_chunks == action_horizon
    assert cfg.actor.model.action_dim == action_dim
    assert cfg.actor.model.rlt_stage2.num_images_in_input == num_images
    assert cfg.actor.model.rlt_stage2.proprio_dim == proprio_dim
    assert cfg.env.eval.action_exec_chunks == action_horizon

    if proprio_mode is not None:
        assert cfg.actor.model.rlt_stage2.proprio_mode == proprio_mode
    assert (
        resolve_proprio_dim(
            cfg.actor.model.rlt_stage2,
            default_dim=cfg.actor.model.rlt_stage2.proprio_dim,
        )
        == proprio_dim
    )


def test_rlt_maniskill_joint_dataconfig_contract():
    canonical = get_openpi_config("pi05_rlt_joint")
    legacy = get_openpi_config("pi05_rlt_maniskill_joint")

    assert _normalize_config(canonical) == _normalize_config(legacy)
    _assert_rlt_train_config_contract(
        canonical,
        repo_id="rlt_maniskill_joint",
        image_key="image",
        wrist_image_key="wrist_image",
    )

    raw_sample = {
        "image": _uint8_image((384, 384, 3)),
        "wrist_image": _uint8_image((3, 128, 128)),
        "state": np.linspace(-1.0, 1.0, 9, dtype=np.float32),
        "actions": np.arange(10 * 10, dtype=np.float32).reshape(10, 10),
        "prompt": "insert the peg in the hole",
    }
    _assert_rlt_policy_transform_contract(
        canonical,
        raw_sample,
        expected_state_dim=9,
        expected_base_image_shape=(384, 384, 3),
        expected_wrist_image_shape=(128, 128, 3),
    )


def test_rlt_maniskill_stage2_yaml_dimension_contract():
    cfg = _load_yaml_config(MANISKILL_STAGE2_CONFIG)

    assert cfg.env.train.wrap_obs_mode == "rlt_openpi_joint"
    assert cfg.env.train.init_params.sensor_configs.width == 384
    assert cfg.env.train.init_params.sensor_configs.height == 384
    _assert_stage2_dimension_contract(
        cfg,
        config_name="pi05_rlt_joint",
        action_dim=8,
        action_horizon=10,
        num_images=2,
        proprio_dim=9,
    )
