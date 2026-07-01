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

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import rlinf.config as config_module
from rlinf.config import validate_embodied_cfg


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _make_async_sac_qwen_cfg():
    return OmegaConf.create(
        {
            "runner": {
                "only_eval": False,
                "val_check_interval": -1,
                "overlap_env_bootstrap": False,
            },
            "algorithm": {"loss_type": "embodied_sac"},
            "reward": {
                "use_reward_model": True,
                "reward_mode": "history_buffer",
                "pending_step_window": 1,
                "aggregate_request_count": 1,
                "use_output_step": 0,
            },
            "rollout": {"pipeline_stage_num": 1},
            "actor": {
                "model": {
                    "model_type": "mlp_policy",
                    "num_action_chunks": 1,
                    "policy_setup": "panda-ee-dpos",
                    "action_dim": 4,
                }
            },
            "env": {
                "train": {
                    "env_type": "maniskill",
                    "total_num_envs": 8,
                    "group_size": 1,
                    "max_steps_per_rollout_epoch": 8,
                    "enable_offload": False,
                    "init_params": {},
                },
                "eval": {
                    "env_type": "maniskill",
                    "total_num_envs": 8,
                    "group_size": 1,
                    "max_steps_per_rollout_epoch": 8,
                    "init_params": {},
                },
            },
        }
    )


@pytest.fixture(autouse=True)
def patch_cluster(monkeypatch):
    class _FakePlacement:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_world_size(self, component_name):
            del component_name
            return 1

    monkeypatch.setattr(config_module, "Cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_module, "HybridComponentPlacement", _FakePlacement)


def test_async_sac_qwen_config_validates():
    cfg = _make_async_sac_qwen_cfg()

    validated_cfg = validate_embodied_cfg(cfg)

    assert validated_cfg.algorithm.loss_type == "embodied_sac"
    assert validated_cfg.reward.use_reward_model is True
    assert validated_cfg.reward.reward_mode == "history_buffer"
    assert validated_cfg.reward.pending_step_window == 1
    assert validated_cfg.reward.aggregate_request_count == 1
    assert validated_cfg.reward.use_output_step == 0
    assert validated_cfg.actor.model.policy_setup == "panda-ee-dpos"
    assert validated_cfg.actor.model.action_dim == 4
    assert validated_cfg.env.train.init_params.control_mode == "pd_ee_delta_pos"


def test_embodied_eval_config_without_algorithm_validates():
    cfg = _make_async_sac_qwen_cfg()
    del cfg.algorithm
    del cfg.actor
    del cfg.env.train
    cfg.runner.task_type = "embodied_eval"
    cfg.runner.only_eval = True
    cfg.rollout.model = {
        "model_type": "mlp_policy",
        "num_action_chunks": 1,
        "policy_setup": "panda-ee-dpos",
        "action_dim": 4,
    }

    validated_cfg = validate_embodied_cfg(cfg)

    assert "algorithm" not in validated_cfg
    assert validated_cfg.runner.only_eval is True
    assert validated_cfg.env.eval.init_params.control_mode == "pd_ee_delta_pos"


def test_async_sac_qwen_config_allows_no_pending_reward():
    cfg = _make_async_sac_qwen_cfg()
    cfg.reward.pending_step_window = 0
    cfg.reward.aggregate_request_count = 1

    validated_cfg = validate_embodied_cfg(cfg)

    assert validated_cfg.reward.pending_step_window == 0
    assert validated_cfg.reward.aggregate_request_count == 1


def test_async_sac_qwen_config_defaults_to_no_pending_reward():
    cfg = _make_async_sac_qwen_cfg()
    del cfg.reward.pending_step_window
    cfg.reward.aggregate_request_count = 1

    validated_cfg = validate_embodied_cfg(cfg)

    assert validated_cfg.reward.pending_step_window == 0
    assert validated_cfg.reward.aggregate_request_count == 1


def test_async_sac_qwen_config_rejects_aggregated_no_pending_reward():
    cfg = _make_async_sac_qwen_cfg()
    cfg.reward.pending_step_window = 0
    cfg.reward.aggregate_request_count = 2

    with pytest.raises(AssertionError, match="aggregate_request_count must be 1"):
        validate_embodied_cfg(cfg)


def test_async_sac_qwen_config_rejects_delayed_reward():
    cfg = _make_async_sac_qwen_cfg()
    cfg.reward.use_output_step = 1

    with pytest.raises(AssertionError, match="reward.use_output_step=0"):
        validate_embodied_cfg(cfg)
