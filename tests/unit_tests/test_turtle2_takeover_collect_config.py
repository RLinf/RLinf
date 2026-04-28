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

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg
from rlinf.envs.wrappers import CollectEpisode
from rlinf.workers.env.env_worker import EnvWorker


def _compose_config(config_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            return compose(config_name=config_name)
    finally:
        GlobalHydra.instance().clear()


def test_turtle2_takeover_collect_openpi_config_composes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    cfg = _compose_config("realworld_turtle2_dagger_takeover_collect_openpi")

    assert cfg.runner.only_eval is True
    assert cfg.rollout.collect_transitions is False
    assert cfg.algorithm.adv_type == "raw"
    assert cfg.algorithm.loss_type == "actor"
    assert cfg.env.eval.use_master_takeover is True
    assert cfg.env.eval.action_mode == "absolute_pose"
    assert cfg.env.eval.master_takeover.control_mode == "pose"
    assert cfg.env.eval.master_takeover.max_pose_age_s == 0.25
    assert cfg.env.eval.override_cfg.pose_control_backend == "direct"
    assert cfg.env.eval.override_cfg.direct_publish_hz == 100.0
    assert cfg.env.eval.data_collection.enabled is True
    assert cfg.env.eval.data_collection.export_format == "pickle"
    assert cfg.env.eval.data_collection.only_success is False
    assert cfg.env.eval.data_collection.record_executed_action is True
    assert cfg.env.eval.data_collection.robot_type == "turtle2"
    assert cfg.actor.model.openpi.config_name == "pi0_turtle2_x2robot"
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 30


def test_turtle2_takeover_config_rejects_single_arm(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    cfg = _compose_config("realworld_turtle2_dagger_takeover_collect_openpi")
    cfg.env.eval.override_cfg.use_arm_ids = [1]

    with pytest.raises(ValueError, match=r"override_cfg.use_arm_ids=\[0, 1\]"):
        validate_embodied_cfg(cfg)


def test_turtle2_default_backend_stays_smooth(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    cfg = _compose_config("env/realworld_turtle2_deploy")

    assert "pose_control_backend" not in cfg.env.override_cfg


class OneStepEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)}
        )

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros((1, 1), dtype=np.float32)}, {}

    def step(self, action):
        return {"state": np.ones((1, 1), dtype=np.float32)}, 0.0, True, False, {}


def test_data_collection_wraps_env_when_collect_transitions_false(tmp_path):
    worker = EnvWorker.__new__(EnvWorker)
    worker.stage_num = 1
    worker._rank = 0
    worker._world_size = 1
    worker._worker_info = None

    env_cfg = OmegaConf.create(
        {
            "video_cfg": {"save_video": False},
            "data_collection": {
                "enabled": True,
                "save_dir": str(tmp_path),
                "export_format": "pickle",
                "only_success": False,
                "only_intervened": False,
                "record_executed_action": True,
                "robot_type": "turtle2",
                "fps": 60,
                "finalize_interval": 1,
            },
        }
    )

    envs = worker._setup_env_and_wrappers(OneStepEnv, env_cfg, 1)

    assert len(envs) == 1
    assert isinstance(envs[0], CollectEpisode)
    assert envs[0].record_executed_action is True
