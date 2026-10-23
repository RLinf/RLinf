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

"""Unit tests for the embodied env step-reward registry and built-ins."""

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.rewards import (
    ENV_REWARD_REGISTRY,
    get_env_reward_fn,
    list_env_rewards,
    register_env_reward,
)


@pytest.fixture
def raw_reward():
    return torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)


@pytest.fixture
def info():
    return {
        "success": torch.tensor([False, True, True]),
        "is_src_obj_grasped": torch.tensor([True, True, False]),
        "gripper_obj_dist": torch.tensor([0.0, 0.2, 1.0]),
    }


def test_builtins_registered():
    for name in ("raw", "only_success", "weighted_components"):
        assert name in list_env_rewards()
        assert callable(get_env_reward_fn(name))


def test_raw_passthrough(raw_reward, info):
    fn = get_env_reward_fn("raw")
    reward = fn(raw_reward=raw_reward, info=info, cfg=None)
    assert reward.dtype == torch.float32
    assert torch.allclose(reward, raw_reward)


def test_only_success(raw_reward, info):
    fn = get_env_reward_fn("only_success")
    reward = fn(raw_reward=raw_reward, info=info, cfg=None)
    assert torch.allclose(reward, torch.tensor([0.0, 1.0, 1.0]))


def test_weighted_components_scalar_weights(raw_reward, info):
    # The layered structure from RLinf issue #1422: contact 0.3 + task
    # completion 1.0 as plain weights.
    cfg = OmegaConf.create(
        {
            "reward_components": {
                "is_src_obj_grasped": 0.3,
                "success": 1.0,
            }
        }
    )
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    expected = torch.tensor([0.3, 1.3, 1.0])
    assert torch.allclose(reward, expected)


def test_weighted_components_dense_transform(raw_reward, info):
    cfg = OmegaConf.create(
        {
            "reward_components": {
                "gripper_obj_dist": {
                    "weight": 0.5,
                    "transform": "one_minus_tanh",
                    "scale": 5.0,
                }
            }
        }
    )
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    expected = 0.5 * (1.0 - torch.tanh(5.0 * info["gripper_obj_dist"]))
    assert torch.allclose(reward, expected)


def test_weighted_components_neg_and_neg_exp(raw_reward, info):
    cfg = OmegaConf.create(
        {
            "reward_components": {
                "gripper_obj_dist": {"weight": 1.0, "transform": "neg"},
            }
        }
    )
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    assert torch.allclose(reward, -info["gripper_obj_dist"])

    cfg = OmegaConf.create(
        {
            "reward_components": {
                "gripper_obj_dist": {
                    "weight": 1.0,
                    "transform": "neg_exp",
                    "scale": 2.0,
                },
            }
        }
    )
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    assert torch.allclose(reward, torch.exp(-2.0 * info["gripper_obj_dist"]))


def test_weighted_components_hierarchical_gating(raw_reward, info):
    # Task completion only counts when the grasp stage holds, mirroring the
    # hardcoded (success & is_src_obj_grasped) hierarchy of the default mode.
    cfg = OmegaConf.create(
        {
            "reward_components": {
                "success": {
                    "weight": 1.0,
                    "requires": ["is_src_obj_grasped"],
                },
            }
        }
    )
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    # env 1 has success & grasp; env 2 has success but no grasp.
    assert torch.allclose(reward, torch.tensor([0.0, 1.0, 0.0]))


def test_weighted_components_raw_component(raw_reward, info):
    cfg = OmegaConf.create({"reward_components": {"raw": 0.1, "success": 1.0}})
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    expected = 0.1 * raw_reward + info["success"].float()
    assert torch.allclose(reward, expected)


def test_weighted_components_maniskill_default_equivalence(raw_reward, info):
    # The YAML equivalent of ManiskillEnv's hardcoded "default" reward.
    info = dict(info, consecutive_grasp=torch.tensor([False, True, False]))
    cfg = OmegaConf.create(
        {
            "reward_components": {
                "is_src_obj_grasped": 0.1,
                "consecutive_grasp": 0.1,
                "success": {"weight": 1.0, "requires": ["is_src_obj_grasped"]},
            }
        }
    )
    fn = get_env_reward_fn("weighted_components")
    reward = fn(raw_reward=raw_reward, info=info, cfg=cfg)
    expected = (
        info["is_src_obj_grasped"] * 0.1
        + info["consecutive_grasp"] * 0.1
        + (info["success"] & info["is_src_obj_grasped"]) * 1.0
    ).float()
    assert torch.allclose(reward, expected)


def test_weighted_components_errors(raw_reward, info):
    fn = get_env_reward_fn("weighted_components")
    with pytest.raises(ValueError, match="reward_components"):
        fn(raw_reward=raw_reward, info=info, cfg=OmegaConf.create({}))
    with pytest.raises(KeyError, match="not found in env info"):
        fn(
            raw_reward=raw_reward,
            info=info,
            cfg=OmegaConf.create({"reward_components": {"missing_key": 1.0}}),
        )
    with pytest.raises(ValueError, match="Unknown transform"):
        fn(
            raw_reward=raw_reward,
            info=info,
            cfg=OmegaConf.create(
                {
                    "reward_components": {
                        "success": {"weight": 1.0, "transform": "bogus"}
                    }
                }
            ),
        )


def test_register_custom_reward(raw_reward, info):
    @register_env_reward("test_custom_reward")
    def custom_fn(*, raw_reward, info, cfg=None):
        return raw_reward * 2.0

    try:
        fn = get_env_reward_fn("TEST_CUSTOM_REWARD")  # case-insensitive
        assert torch.allclose(fn(raw_reward=raw_reward, info=info), raw_reward * 2.0)
        with pytest.raises(AssertionError, match="already registered"):
            register_env_reward("test_custom_reward")(custom_fn)
    finally:
        ENV_REWARD_REGISTRY.pop("test_custom_reward", None)


def test_unknown_reward_name():
    with pytest.raises(ValueError, match="not registered"):
        get_env_reward_fn("does_not_exist")


def test_reward_fn_module_import(tmp_path, monkeypatch, raw_reward, info):
    module_file = tmp_path / "my_custom_rewards.py"
    module_file.write_text(
        "from rlinf.envs.rewards import register_env_reward\n"
        "\n"
        "@register_env_reward('test_module_reward')\n"
        "def module_reward(*, raw_reward, info, cfg=None):\n"
        "    return raw_reward + 1.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        fn = get_env_reward_fn("test_module_reward", module="my_custom_rewards")
        assert torch.allclose(fn(raw_reward=raw_reward, info=info), raw_reward + 1.0)
    finally:
        ENV_REWARD_REGISTRY.pop("test_module_reward", None)
