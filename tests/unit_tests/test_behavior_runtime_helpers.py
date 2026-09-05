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
"""Unit tests for BEHAVIOR runtime helper modules.

These tests directly cover helper-only logic split out of ``behavior_env.py``.
They use fake robot / scene objects and do not require OmniGibson startup.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.action_controls import (
    apply_action_mask,
    apply_first_chunk_action_override,
    parse_action_mask,
    parse_first_chunk_action_override,
    r1pro_noop_action,
)
from rlinf.envs.behavior.instance_loader import RLINF_REPLAY_METADATA_KEY
from rlinf.envs.behavior.replay_runtime import (
    apply_replay_tro_metadata,
    stage_idx_from_info,
    stage_idx_from_reward,
)
from rlinf.envs.behavior.stage_rewards import (
    extract_episode_done,
    extract_episode_success,
    is_target_stage_success,
    stage_cumulative_reward_tensor,
    stage_sparse_reward_tensor,
    stage_weighted_reward_tensor,
)
from rlinf.envs.behavior.utils import normalize_omnigibson_reward_config


class FakeRobot:
    def __init__(self):
        self.joint_positions = torch.arange(28, dtype=torch.float32)

    def get_joint_positions(self):
        return self.joint_positions


class FakeScene:
    def __init__(self, metadata):
        self.metadata = metadata

    def get_task_metadata(self, key):
        if key == RLINF_REPLAY_METADATA_KEY:
            return self.metadata
        return None


class FakeTaskReward:
    def __init__(self):
        self._total_stages = 4
        self._stage_defs = [
            {"name": "move"},
            {"name": "pickup"},
            {"name": "press"},
            {"name": "place"},
        ]
        self.current_stage_idx = None

    def set_active_stage_index(self, stage_idx):
        self.current_stage_idx = stage_idx


class FakeTask:
    def __init__(self):
        self.task_reward = FakeTaskReward()


class FakeChildEnv:
    def __init__(self, metadata):
        self.scene = FakeScene(metadata)
        self.task = FakeTask()


class FakeWrapper:
    def __init__(self, env):
        self.env = env


class TestActionControls:
    def test_parse_action_mask_freezes_base_and_trunk(self):
        cfg = OmegaConf.create(
            {
                "action_mask": {
                    "action_dim": 8,
                    "freeze_base": True,
                    "freeze_trunk": True,
                }
            }
        )

        assert parse_action_mask(cfg) == [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ]

    def test_parse_action_mask_disabled(self):
        cfg = OmegaConf.create({"action_mask": {"enabled": False, "mask": [False]}})

        assert parse_action_mask(cfg) is None

    def test_parse_action_mask_rejects_empty_mask(self):
        cfg = OmegaConf.create({"action_mask": {"mask": []}})

        with pytest.raises(ValueError, match="non-empty bool list"):
            parse_action_mask(cfg)

    def test_r1pro_noop_action_maps_joint_positions(self):
        noop = r1pro_noop_action(FakeRobot(), torch.zeros(23))

        assert torch.equal(noop[3:7], torch.tensor([6.0, 7.0, 8.0, 9.0]))
        assert torch.equal(
            noop[7:14], torch.tensor([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
        )
        assert torch.equal(
            noop[14:21], torch.tensor([11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0])
        )
        assert noop[21].item() == 49.0
        assert noop[22].item() == 53.0

    def test_apply_action_mask_replaces_frozen_dimensions(self):
        actions = torch.full((1, 23), -1.0)
        mask = [True] * 23
        mask[3] = False
        mask[7] = False

        masked = apply_action_mask(
            actions,
            mask,
            child_envs=[object()],
            get_robot_from_child_env=lambda _: FakeRobot(),
        )

        assert masked[0, 0].item() == -1.0
        assert masked[0, 3].item() == 6.0
        assert masked[0, 7].item() == 10.0

    def test_first_chunk_override_parsing_and_application(self):
        cfg = OmegaConf.create(
            {
                "first_chunk_action_override": {
                    "enabled": True,
                    "action_ids": [0, 2],
                    "value": 0.5,
                }
            }
        )
        enabled, action_ids, action_value = parse_first_chunk_action_override(cfg)
        actions = np.zeros((3, 4), dtype=np.float32)

        overridden = apply_first_chunk_action_override(
            actions,
            env_mask=np.asarray([True, False, True]),
            enabled=enabled,
            action_ids=action_ids,
            action_value=action_value,
        )

        np.testing.assert_allclose(overridden[[0, 2], 0], [0.5, 0.5])
        np.testing.assert_allclose(overridden[[0, 2], 2], [0.5, 0.5])
        assert overridden[1, 0] == 0.0

    def test_first_chunk_override_rejects_invalid_action_id(self):
        with pytest.raises(ValueError, match="invalid"):
            apply_first_chunk_action_override(
                torch.zeros((1, 2)),
                env_mask=np.asarray([True]),
                enabled=True,
                action_ids=[2],
                action_value=-1.0,
            )


class TestStageRewards:
    def test_target_stage_success_requires_matching_stage_and_bonus(self):
        task_reward = {"current_stage_idx": 3, "completion_bonus": 1.0}

        assert is_target_stage_success(task_reward, success_stage_idx=3) is True
        assert is_target_stage_success(task_reward, success_stage_idx=2) is False
        assert (
            is_target_stage_success(
                {"current_stage_idx": 3, "completion_bonus": 0.0},
                success_stage_idx=3,
            )
            is False
        )

    def test_stage_sparse_reward_only_rewards_target_stage(self):
        infos = [
            {
                "reward": {
                    "task_specific": {"current_stage_idx": 3, "completion_bonus": 1}
                }
            },
            {
                "reward": {
                    "task_specific": {"current_stage_idx": 2, "completion_bonus": 1}
                }
            },
        ]

        reward = stage_sparse_reward_tensor(
            torch.zeros(2), infos, reward_coef=2.0, success_stage_idx=3
        )

        assert torch.equal(reward, torch.tensor([2.0, 0.0]))

    def test_stage_weighted_reward_applies_per_stage_weights(self):
        infos = [
            {
                "reward": {
                    "task_specific": {"current_stage_idx": 1, "completion_bonus": 1}
                }
            },
            {
                "reward": {
                    "task_specific": {"current_stage_idx": 3, "completion_bonus": 2}
                }
            },
        ]

        reward = stage_weighted_reward_tensor(
            torch.zeros(2), infos, reward_coef=1.0, stage_reward_weights=[0.5, 1.0, 3.0]
        )

        assert torch.equal(reward, torch.tensor([0.5, 6.0]))

    def test_stage_cumulative_reward_uses_completed_stage_count(self):
        infos = [
            {"reward": {"task_specific": {"completed_stage_count": 2}}},
            {"reward": {"task_specific": {"completed_stage_count": 3}}},
        ]

        reward = stage_cumulative_reward_tensor(torch.zeros(2), infos, reward_coef=0.5)

        assert torch.equal(reward, torch.tensor([1.0, 1.5]))

    def test_episode_success_and_done_use_target_stage_when_configured(self):
        info = {
            "done": {"success": False},
            "reward": {
                "task_specific": {"current_stage_idx": 3, "completion_bonus": 1}
            },
        }

        assert extract_episode_success(info, success_stage_idx=3) is True
        assert extract_episode_done(
            info, success_stage_idx=3, default_done_extractor=bool
        )

    def test_episode_success_falls_back_to_done_success(self):
        info = {"done": {"success": True}}

        assert extract_episode_success(info, success_stage_idx=None) is True


class TestReplayRuntime:
    def test_apply_replay_tro_metadata_injects_info_and_sets_reward_stage(self):
        child_env = FakeChildEnv(
            {
                "stage_index": "2",
                "stage_prompts": ["move radio", "pick radio", "press radio"],
                "source_instance_id": 7,
            }
        )

        info = apply_replay_tro_metadata(child_env, info={})

        assert child_env.task.task_reward.current_stage_idx == 2
        assert info["replay_init"]["replay_stage_idx"] == 2
        assert info["replay_init"]["source_instance_id"] == 7
        assert info["reward"]["task_specific"]["current_stage_idx"] == 2
        assert info["reward"]["task_specific"]["completed_stage_count"] == 2
        assert info["reward"]["task_specific"]["total_stage_count"] == 4
        assert info["reward"]["task_specific"]["current_stage_prompt"] == "press radio"
        assert info["reward"]["task_specific"]["current_stage_name"] == "press"

    def test_apply_replay_tro_metadata_ignores_invalid_metadata_shape(self):
        child_env = FakeChildEnv(metadata=None)

        assert apply_replay_tro_metadata(child_env, info=None) == {}

    def test_apply_replay_tro_metadata_unwraps_child_env(self):
        child_env = FakeChildEnv(
            {
                "stage_index": "1",
                "stage_prompts": ["move radio", "press radio"],
            }
        )

        info = apply_replay_tro_metadata(FakeWrapper(child_env), info={})

        assert child_env.task.task_reward.current_stage_idx == 1
        assert info["replay_init"]["replay_stage_idx"] == 1
        assert info["reward"]["task_specific"]["current_stage_prompt"] == "press radio"

    def test_stage_idx_helpers_parse_info_and_reward(self):
        child_env = FakeChildEnv({"stage_index": 1})
        child_env.task.task_reward.current_stage_idx = "3"

        assert stage_idx_from_info({"task_reward": {"current_stage_idx": "2"}}) == 2
        assert stage_idx_from_info({"current_stage_idx": "bad"}) is None
        assert stage_idx_from_reward(child_env) == 3


class TestOmniGibsonRewardConfig:
    def test_rlinf_stage_reward_mode_is_normalized_for_omnigibson(self):
        cfg = OmegaConf.create(
            {
                "task": {
                    "reward_config": {
                        "reward_mode": "stage_weighted",
                        "stage_reward_weights": [0.5, 1.0, 2.0],
                    }
                }
            }
        )

        normalized = normalize_omnigibson_reward_config(cfg)

        assert normalized.task.reward_config.reward_mode == "stage"
        assert "stage_reward_weights" not in normalized.task.reward_config

    def test_omnigibson_reward_mode_is_left_unchanged(self):
        cfg = OmegaConf.create({"task": {"reward_config": {"reward_mode": "task"}}})

        normalized = normalize_omnigibson_reward_config(cfg)

        assert normalized.task.reward_config.reward_mode == "task"
