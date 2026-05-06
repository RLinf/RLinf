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

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import rlinf.config as config_module
from rlinf.config import validate_cfg


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose_async_sac_qwen_cfg(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    embodied_path = repo_root / "examples" / "embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(embodied_path))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(embodied_path / "config")
    ):
        cfg = compose(
            config_name="maniskill_sac_mlp_qwen3vl4b_dualview_history_reward_async"
        )
    cfg.actor.global_batch_size = cfg.actor.micro_batch_size * 8
    return cfg


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


def test_async_sac_qwen_config_validates(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)

    validated_cfg = validate_cfg(cfg)

    assert validated_cfg.algorithm.loss_type == "embodied_sac"
    assert validated_cfg.reward.use_reward_model is True
    assert validated_cfg.reward.reward_mode == "history_buffer"
    assert validated_cfg.reward.pending_step_window == 1
    assert validated_cfg.reward.aggregate_request_count == 1
    assert validated_cfg.reward.use_output_step == 0
    assert validated_cfg.algorithm.replay_buffer.save_checkpoint is False
    assert validated_cfg.algorithm.replay_buffer.load_checkpoint is False
    assert validated_cfg.actor.model.policy_setup == "panda-ee-dpos"
    assert validated_cfg.actor.model.action_dim == 4
    assert validated_cfg.env.train.init_params.control_mode == "pd_ee_delta_pos"


def test_async_sac_qwen_config_allows_no_pending_reward(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)
    cfg.reward.pending_step_window = 0
    cfg.reward.aggregate_request_count = 1

    validated_cfg = validate_cfg(cfg)

    assert validated_cfg.reward.pending_step_window == 0
    assert validated_cfg.reward.aggregate_request_count == 1


def test_async_sac_qwen_config_defaults_to_no_pending_reward(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)
    del cfg.reward.pending_step_window
    cfg.reward.aggregate_request_count = 1

    validated_cfg = validate_cfg(cfg)

    assert validated_cfg.reward.pending_step_window == 0
    assert validated_cfg.reward.aggregate_request_count == 1


def test_async_sac_qwen_config_rejects_aggregated_no_pending_reward(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)
    cfg.reward.pending_step_window = 0
    cfg.reward.aggregate_request_count = 2

    with pytest.raises(AssertionError, match="aggregate_request_count must be 1"):
        validate_cfg(cfg)


def test_async_sac_qwen_config_rejects_delayed_reward(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)
    cfg.reward.use_output_step = 1

    with pytest.raises(AssertionError, match="reward.use_output_step=0"):
        validate_cfg(cfg)
