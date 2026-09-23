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

"""CPU contract tests for simulation environment adapters."""

import importlib
import os
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv


class StubVectorEnv:
    """Model the external bridge with deterministic observations and rewards."""

    def __init__(self, task_config: dict, n_envs: int, env_seeds: list[int]):
        assert task_config == {"task_name": "test_task"}
        assert len(env_seeds) == n_envs
        self.n_envs = n_envs
        self.steps = np.zeros(n_envs, dtype=np.int64)
        self.closed = []

    def reset(self, env_idx=None, env_seeds=None):
        assert len(env_seeds) == self.n_envs
        self.steps[slice(None) if env_idx is None else env_idx] = 0

    def get_obs(self):
        return [
            {
                "full_image": np.full((8, 8, 3), step, dtype=np.uint8),
                "left_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "right_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "state": np.full(14, step, dtype=np.float32),
                "instruction": "complete the task",
            }
            for step in self.steps
        ]

    def step(self, actions):
        assert actions.shape[0] == self.n_envs
        self.steps += actions.shape[1]
        success = self.steps >= 3
        return (
            self.get_obs(),
            np.full(self.n_envs, actions.shape[1], dtype=np.float32),
            success,
            np.zeros(self.n_envs, dtype=bool),
            [{"success": bool(value)} for value in success],
        )

    def check_seeds(self, seeds):
        return [seed >= 0 for seed in seeds]

    def close(self, clear_cache):
        self.closed.append(clear_cache)


@pytest.fixture
def make_env(monkeypatch):
    bridge = ModuleType("robodojo_runtime.bridge")
    bridge.VectorEnv = StubVectorEnv
    monkeypatch.setitem(sys.modules, "robodojo_runtime", ModuleType("robodojo_runtime"))
    monkeypatch.setitem(sys.modules, "robodojo_runtime.bridge", bridge)
    monkeypatch.setenv("ROBODOJO_ASSETS_ROOT", "inherited-assets")
    environments = []

    def build(record_metrics=True, **overrides):
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "group_size": 1,
                "auto_reset": False,
                "ignore_terminations": False,
                "use_rel_reward": False,
                "use_custom_reward": False,
                "use_fixed_reset_state_ids": False,
                "task_config": {"task_name": "test_task"},
                "assets_path": "/unused/stub-assets",
                "max_episode_steps": 5,
                "reward_coef": 2.0,
                **overrides,
            }
        )
        env = RoboDojoEnv(cfg, 2, 0, 1, None, record_metrics)
        environments.append(env)
        return env

    yield build
    for env in environments:
        env.close()


def test_robodojo_registration_and_lazy_import():
    assert get_env_cls("robodojo") is RoboDojoEnv
    assert (
        importlib.import_module("rlinf.envs.sim.robodojo.robodojo_env").RoboDojoEnv
        is RoboDojoEnv
    )
    # A fresh interpreter blocks every simulator import, even if installed.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['robodojo_runtime'] = None; "
            "sys.modules['isaacsim'] = None; sys.modules['isaaclab'] = None; "
            "from rlinf.envs import get_env_cls; "
            "from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv; "
            "assert get_env_cls('robodojo') is RoboDojoEnv",
        ],
        check=True,
    )


@pytest.mark.parametrize("assets_path", [None, "/data/scenes"])
def test_robodojo_assets_root_preserves_environment_default(make_env, assets_path):
    make_env(assets_path=assets_path)
    assert os.environ["ROBODOJO_ASSETS_ROOT"] == (assets_path or "inherited-assets")


@pytest.mark.parametrize("record_metrics", [True, False])
def test_robodojo_reset_step_chunk_contract(make_env, record_metrics):
    env = make_env(record_metrics=record_metrics)
    assert env.device == torch.device("cpu")
    assert env.is_start
    obs, infos = env.reset()
    assert not env.is_start
    assert infos == {}
    assert obs["main_images"].shape == (2, 8, 8, 3)
    assert obs["wrist_images"].shape == (2, 2, 8, 8, 3)
    assert obs["states"].shape == (2, 14)
    assert obs["task_descriptions"] == ["complete the task"] * 2
    assert env.elapsed_steps.tolist() == [0, 0]

    obs, reward, terminated, truncated, infos = env.step(
        {"actions": torch.zeros(2, 14)}
    )
    assert reward.tolist() == [1.0, 1.0]
    assert terminated.tolist() == truncated.tolist() == [False, False]
    assert env.elapsed_steps.tolist() == [1, 1]
    assert obs["states"][:, 0].tolist() == [1.0, 1.0]
    assert ("episode" in infos) == record_metrics

    observations, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((2, 2, 14), dtype=np.float32)
    )
    assert len(observations) == len(infos) == 1
    assert rewards.tolist() == [[0.0, 2.0], [0.0, 2.0]]
    assert terminations.tolist() == [[False, True], [False, True]]
    assert not truncations.any()
    assert env.elapsed_steps.tolist() == [3, 3]
    if record_metrics:
        assert infos[0]["episode"]["return"].tolist() == [3.0, 3.0]
        assert infos[0]["episode"]["success_once"].all()

    obs, _ = env.reset(env_idx=[0])
    assert env.elapsed_steps.tolist() == [0, 3]
    assert obs["states"][:, 0].tolist() == [0.0, 3.0]
    _, _, _, truncated, _ = env.step(np.zeros((2, 2, 14)))
    assert truncated.tolist() == [False, True]


def test_robodojo_auto_reset_preserves_final_state(make_env):
    env = make_env(auto_reset=True, is_eval=True)
    env.reset()
    env.step(np.zeros((2, 1, 14)))
    env.reset(env_idx=0)
    observations, _, terminated, _, infos = env.chunk_step(np.zeros((2, 2, 14)))
    assert terminated.tolist() == [[False, False], [False, True]]
    assert env.elapsed_steps.tolist() == [2, 0]
    assert observations[0]["states"][:, 0].tolist() == [2.0, 0.0]
    assert infos[0]["_final_observation"].tolist() == [False, True]
    assert infos[0]["final_observation"]["states"][:, 0].tolist() == [2.0, 3.0]
    assert infos[0]["final_info"]["episode"]["episode_len"].tolist() == [2, 3]


def test_robodojo_custom_reward_and_ignored_termination(make_env):
    env = make_env(
        use_custom_reward=True, use_rel_reward=True, ignore_terminations=True
    )
    env.reset()
    _, reward, terminated, _, infos = env.step(np.zeros((2, 3, 14)))
    assert reward.tolist() == [2.0, 2.0]
    assert not terminated.any()
    assert infos["episode"]["success_at_end"].all()
    _, reward, _, _, _ = env.step(np.zeros((2, 14)))
    assert reward.tolist() == [0.0, 0.0]
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros((2, 14)))
    assert reward.tolist() == [0.0, 0.0]


def test_robodojo_seeds_and_close(make_env):
    env = make_env(group_size=2, use_fixed_reset_state_ids=True)
    seeds = env.reset_state_ids.clone()
    assert seeds[0] == seeds[1]
    env.update_reset_state_ids()
    assert torch.equal(seeds, env.reset_state_ids)
    assert env.check_seeds([1, -1]) == [True, False]
    env.close(clear_cache=False)
    env.close()
    assert env.venv.closed == [False]


def test_robodojo_action_passthrough():
    from rlinf.envs.action_utils import prepare_actions

    actions = np.zeros((2, 3, 14))
    assert prepare_actions(actions, "robodojo", "openpi", 1, 14) is actions
