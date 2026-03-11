# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
import types
from enum import Enum

import gymnasium as gymnasium
import numpy as np
from omegaconf import OmegaConf


class _DummyTaskEnv:
    def __init__(self, env_name: str, split: str | None = None, **kwargs):
        del kwargs
        self.env_name = env_name
        self.split = split
        self.step_count = 0

    def _obs(self):
        image_value = 5 if self.split == "pretrain" else 9
        return {
            "robot0_agentview_left_image": np.full(
                (4, 4, 3), image_value + self.step_count, dtype=np.uint8
            ),
            "robot0_eye_in_hand_image": np.full(
                (4, 4, 3), image_value + self.step_count + 1, dtype=np.uint8
            ),
            "robot0_base_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "robot0_base_to_eef_quat": np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float32
            ),
            "robot0_base_to_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_gripper_qvel": np.array([0.4], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.5], dtype=np.float32),
        }

    def reset(self, **kwargs):
        del kwargs
        self.step_count = 0
        return self._obs()

    def step(self, action):
        del action
        self.step_count += 1
        success = self.step_count >= 1
        return self._obs(), float(success), success, {"success": success}

    def close(self):
        return None


class _DummySubprocEnv:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]

    def reset(self, id=None):
        indices = list(id) if id is not None else list(range(len(self.envs)))
        return [self.envs[idx].reset() for idx in indices]

    def step(self, actions):
        results = [
            env.step(action) for env, action in zip(self.envs, np.asarray(actions))
        ]
        obs, rewards, dones, infos = zip(*results, strict=True)
        return (
            list(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        for env in self.envs:
            env.close()


def _load_robocasa365_module(monkeypatch):
    soups = {
        ("atomic_seen", "pretrain", "human"): [
            "KitchenOpenDrawer",
            "KitchenCompositeServeMeal",
        ],
        ("atomic_seen", "target", "human"): [
            "KitchenCloseDrawer",
            "KitchenCompositeServeMeal",
        ],
    }
    metadata = {
        "KitchenOpenDrawer": {
            "prompt": "open drawer from registry",
            "task_mode": "atomic",
            "task_soup": "atomic_seen",
        },
        "KitchenCloseDrawer": {
            "ep_meta": {"language": "close drawer from nested metadata"},
            "task_mode": "atomic",
            "task_soup": "atomic_seen",
        },
        "KitchenCompositeServeMeal": {
            "description": "serve a meal",
            "task_mode": "composite",
            "task_soup": "atomic_seen",
        },
    }

    fake_robocasa = types.ModuleType("robocasa")
    fake_utils = types.ModuleType("robocasa.utils")
    fake_dataset_registry = types.ModuleType("robocasa.utils.dataset_registry")
    fake_dataset_registry.get_ds_soup = (
        lambda task_soup, split, source: soups[(task_soup, split, source)]
    )
    fake_dataset_registry.get_ds_meta = (
        lambda task_name, source=None: metadata[task_name]
    )
    fake_utils.dataset_registry = fake_dataset_registry
    fake_robocasa.utils = fake_utils

    fake_robosuite = types.ModuleType("robosuite")
    fake_controllers = types.ModuleType("robosuite.controllers")
    fake_controllers.load_composite_controller_config = (
        lambda controller=None, robot=None: {"controller": controller, "robot": robot}
    )

    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "robocasa.utils.dataset_registry", fake_dataset_registry
    )
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)
    monkeypatch.setitem(sys.modules, "gym", gymnasium)

    sys.modules.pop("rlinf.envs.robocasa365", None)
    sys.modules.pop("rlinf.envs.robocasa365.robocasa365_env", None)
    module = importlib.import_module("rlinf.envs.robocasa365.robocasa365_env")
    monkeypatch.setattr(module, "RobocasaSubprocEnv", _DummySubprocEnv)
    monkeypatch.setattr(
        module.gym,
        "make",
        lambda env_name, **kwargs: _DummyTaskEnv(env_name=env_name, **kwargs),
    )
    return module


def _load_action_utils_module(monkeypatch):
    class _FakeSupportedModel(Enum):
        OPENVLA = "openvla"
        OPENVLA_OFT = "openvla_oft"
        OPENPI = "openpi"

    fake_config = types.ModuleType("rlinf.config")
    fake_config.SupportedModel = _FakeSupportedModel

    monkeypatch.setitem(sys.modules, "rlinf.config", fake_config)
    sys.modules.pop("rlinf.envs.action_utils", None)
    return importlib.import_module("rlinf.envs.action_utils")


def _make_cfg(split: str, task_filter=None):
    return OmegaConf.create(
        {
            "seed": 0,
            "group_size": 1,
            "ignore_terminations": False,
            "auto_reset": False,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "video_cfg": {"save_video": False},
            "camera_names": ["robot0_agentview_left", "robot0_eye_in_hand"],
            "init_params": {"camera_heights": 4, "camera_widths": 4},
            "robot_name": "PandaMobile",
            "max_episode_steps": 3,
            "task_source": "dataset_registry",
            "dataset_source": "human",
            "split": split,
            "task_soup": "atomic_seen",
            "task_mode": "atomic",
            "task_filter": task_filter or [],
            "observation": {},
            "action_space": {
                "env_action_dim": 12,
                "openpi_valid_action_slice": [5, 12],
                "disable_base_control": True,
                "base_mode_index": 11,
            },
        }
    )


def test_robocasa365_env_keeps_registry_selection_separate(monkeypatch):
    module = _load_robocasa365_module(monkeypatch)
    cfg = _make_cfg(split="pretrain")

    env = module.Robocasa365Env(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )

    assert len(env.task_specs) == 1
    assert env.task_specs[0]["task_name"] == "KitchenOpenDrawer"
    assert env.task_specs[0]["task_description"] == "open drawer from registry"

    obs, _ = env.reset()
    env.is_start = False

    assert obs["main_images"].shape == (1, 4, 4, 3)
    assert obs["wrist_images"].shape == (1, 4, 4, 3)
    assert obs["states"].shape == (1, 14)
    assert obs["task_descriptions"] == ["open drawer from registry"]
    assert obs["task_metadata"][0]["split"] == "pretrain"
    assert obs["task_metadata"][0]["benchmark_selection"] == "human/pretrain/atomic_seen"

    step_obs, reward, terminations, truncations, _ = env.step(
        np.zeros((1, 12), dtype=np.float32),
        auto_reset=False,
    )

    assert step_obs["task_metadata"][0]["task_mode"] == "atomic"
    assert reward.tolist() == [1.0]
    assert terminations.tolist() == [True]
    assert truncations.tolist() == [False]

    env.close()


def test_robocasa365_env_supports_target_split_and_nested_prompts(monkeypatch):
    module = _load_robocasa365_module(monkeypatch)
    cfg = _make_cfg(
        split="target",
        task_filter={"include": ["CloseDrawer"], "exclude": ["Composite"]},
    )

    env = module.Robocasa365Env(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )

    assert len(env.task_specs) == 1
    assert env.task_specs[0]["task_name"] == "KitchenCloseDrawer"
    assert env.task_specs[0]["task_description"] == "close drawer from nested metadata"
    assert env.task_specs[0]["benchmark_selection"] == "human/target/atomic_seen"

    env.close()


def test_prepare_actions_for_robocasa365_uses_action_space_overrides(monkeypatch):
    action_utils = _load_action_utils_module(monkeypatch)
    raw_chunk_actions = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
    env_cfg = OmegaConf.create(
        {
            "action_space": {
                "env_action_dim": 6,
                "openpi_valid_action_slice": [2, 8],
                "disable_base_control": True,
                "base_mode_index": 5,
            }
        }
    )

    chunk_actions = action_utils.prepare_actions(
        raw_chunk_actions=raw_chunk_actions,
        env_type="robocasa365",
        model_type="openpi",
        num_action_chunks=2,
        action_dim=12,
        env_cfg=env_cfg,
    )

    expected = raw_chunk_actions[..., 2:8].copy()
    expected[..., 5] = 0.0
    np.testing.assert_allclose(chunk_actions, expected)
