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

"""RL rollout contracts for RoboDojo and RoboLab without running Isaac Sim."""

from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv
from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv


@pytest.mark.parametrize(
    ("recipe", "env_type", "chunk_size"),
    [
        (
            "robodojo_put_bottles_into_dustbin_ppo_openpi_pi05",
            "robodojo",
            50,
        ),
        ("robolab_banana_in_bowl_ppo_openpi_pi05", "robolab", 15),
    ],
)
def test_training_recipe_composes(recipe, env_type, chunk_size, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    config_dir = root / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=recipe)
    OmegaConf.resolve(cfg)

    assert cfg.env.train.env_type == env_type
    assert cfg.env.train.auto_reset
    assert not cfg.env.train.enable_offload
    assert cfg.actor.model.num_action_chunks == chunk_size
    samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (cfg.env.train.max_steps_per_rollout_epoch // chunk_size)
    )
    assert samples % cfg.actor.global_batch_size == 0


def _observation(num_envs: int) -> dict:
    return {
        "main_images": np.zeros((num_envs, 8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((num_envs, 8, 8, 3), dtype=np.uint8),
        "states": np.zeros((num_envs, 8), dtype=np.float32),
        "task_descriptions": ["put the banana in the bowl"] * num_envs,
        "active_mask": np.ones(num_envs, dtype=bool),
    }


def test_robolab_chunk_exposes_terminal_data_and_execution_mask(monkeypatch):
    env = object.__new__(RoboLabEnv)
    env.num_envs = 2
    env._closed = True
    terminal_obs = _observation(2)
    reset_obs = _observation(2)
    reset_obs["states"][0, 0] = 1

    result = {
        "observation": reset_obs,
        "final_observation": terminal_obs,
        "final_mask": np.array([True, False]),
        "executed_steps": np.array([2, 3]),
        "rewards": np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
        "terminations": np.array(
            [[False, True, True], [False, False, False]], dtype=bool
        ),
        "truncations": np.zeros((2, 3), dtype=bool),
        "episode": {
            "success_once": np.array([1, 0], dtype=np.float32),
            "return": np.array([1, 0], dtype=np.float32),
            "episode_len": np.array([2, 3], dtype=np.float32),
            "reward": np.array([0.5, 0], dtype=np.float32),
        },
    }
    class RemoteChunkStep:
        def remote(self, actions):
            return result

    env._process = SimpleNamespace(chunk_step=RemoteChunkStep())
    monkeypatch.setattr(
        "rlinf.envs.sim.robolab.robolab_env.ray.get", lambda value: value
    )

    observations, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((2, 3, 8), dtype=np.float32)
    )

    assert len(observations) == len(infos) == 1
    assert rewards.shape == terminations.shape == truncations.shape == (2, 3)
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert infos[0]["_final_observation"].tolist() == [True, False]
    assert infos[0]["final_info"]["episode"]["return"].tolist() == [1, 0]
    assert observations[0]["states"][0, 0].item() == 1
    assert infos[0]["final_observation"]["states"][0, 0].item() == 0


def test_robodojo_chunk_exposes_execution_mask_and_auto_reset():
    env = object.__new__(RoboDojoEnv)
    env.num_envs = 2
    env.auto_reset = True
    env.cfg = SimpleNamespace(get=lambda key, default=None: False)
    obs = {"states": torch.zeros(2, 14)}
    info = {"executed_steps": np.array([2, 3])}

    env.step = MethodType(
        lambda self, actions, auto_reset=False: (
            obs,
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            info,
        ),
        env,
    )

    def auto_reset(self, dones, final_obs, final_info):
        return {"states": torch.ones(2, 14)}, {
            "final_observation": final_obs,
            "final_info": final_info,
            "_final_observation": dones,
            "_final_info": dones,
        }

    env._auto_reset = MethodType(auto_reset, env)
    observations, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((2, 3, 14), dtype=np.float32)
    )

    assert rewards.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert terminations.tolist() == [
        [False, False, True],
        [False, False, False],
    ]
    assert not truncations.any()
    assert observations[0]["states"].eq(1).all()
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
