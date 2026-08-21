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
"""Unit tests for BEHAVIOR replay pipeline (pure Python, no OmniGibson).

Tests cover:
- parse_activity_instance_ids
- ReplayEpisode / ReplayPlan dataclasses
- BehaviorReplayInitializer config validation
- replay_plans_to_infos conversion
- action noise index parsing
- episode index parsing
"""

from pathlib import Path

import numpy as np
import pytest

from rlinf.envs.behavior.instance_loader import (
    DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS,
    RLINF_REPLAY_METADATA_KEY,
    parse_activity_instance_ids,
)
from rlinf.envs.behavior.replay_initializer import (
    BehaviorReplayInitializer,
    ReplayEpisode,
    ReplayPlan,
    maybe_make_replay_initializer,
    replay_plans_to_infos,
)

# ---------------------------------------------------------------------------
# parse_activity_instance_ids
# ---------------------------------------------------------------------------

class TestParseActivityInstanceIds:
    def test_single_int(self):
        result = parse_activity_instance_ids(42)
        assert result == [42]

    def test_none(self):
        assert parse_activity_instance_ids(None) is None

    def test_list_of_ints(self):
        result = parse_activity_instance_ids([1, 2, 3])
        assert result == [1, 2, 3]

    def test_comma_separated_string(self):
        result = parse_activity_instance_ids("1000,1005,1010")
        assert result == [1000, 1005, 1010]

    def test_range_string(self):
        result = parse_activity_instance_ids(["1000-1005"])
        assert result == [1000, 1001, 1002, 1003, 1004, 1005]

    def test_mixed_range_and_int(self):
        result = parse_activity_instance_ids(["1000-1002,2000"])
        assert result == [1000, 1001, 1002, 2000]

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="start > end"):
            parse_activity_instance_ids(["1005-1000"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_activity_instance_ids([])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_replay_metadata_key(self):
        assert RLINF_REPLAY_METADATA_KEY == "rlinf_replay"

    def test_midrollout_settle_steps_default(self):
        assert DEFAULT_MIDROLLOUT_RESTORE_SETTLE_STEPS == 120


# ---------------------------------------------------------------------------
# ReplayEpisode / ReplayPlan dataclasses
# ---------------------------------------------------------------------------

class TestReplayDataclasses:
    def test_replay_episode_frozen(self):
        episode = ReplayEpisode(
            episode_index=0,
            instance_id=1001,
            parquet_path=Path("/tmp/ep_0.parquet"),
            annotation_path=None,
            orchestrator_path=None,
        )
        assert episode.episode_index == 0
        assert episode.instance_id == 1001
        with pytest.raises(Exception):
            episode.instance_id = 999  # frozen dataclass

    def test_replay_plan_defaults(self):
        plan = ReplayPlan(
            episode_index=0,
            instance_id=1001,
            actions=np.zeros((10, 23), dtype=np.float32),
            replay_steps=5,
            target_step=10,
        )
        assert plan.stage_prompts == ()

    def test_replay_plan_with_prompts(self):
        plan = ReplayPlan(
            episode_index=0,
            instance_id=1001,
            actions=np.zeros((10, 23), dtype=np.float32),
            replay_steps=5,
            target_step=10,
            stage_prompts=("grasp the cup", "pour water"),
        )
        assert len(plan.stage_prompts) == 2
        assert plan.stage_prompts[0] == "grasp the cup"


# ---------------------------------------------------------------------------
# replay_plans_to_infos
# ---------------------------------------------------------------------------

class TestReplayPlansToInfos:
    def test_single_plan(self):
        plan = ReplayPlan(
            episode_index=5,
            instance_id=1001,
            actions=np.zeros((3, 23), dtype=np.float32),
            replay_steps=3,
            target_step=3,
        )
        infos = replay_plans_to_infos([plan])
        assert len(infos) == 1
        assert infos[0]["replay_episode_index"] == 5
        assert infos[0]["replay_instance_id"] == 1001
        assert infos[0]["replay_steps"] == 3
        assert infos[0]["replay_target_step"] == 3
        assert infos[0]["replay_stage_prompts"] == []

    def test_multiple_plans(self):
        plans = [
            ReplayPlan(
                episode_index=i,
                instance_id=1000 + i,
                actions=np.zeros((5, 23), dtype=np.float32),
                replay_steps=5,
                target_step=5,
            )
            for i in range(3)
        ]
        infos = replay_plans_to_infos(plans)
        assert len(infos) == 3
        for i, info in enumerate(infos):
            assert info["replay_episode_index"] == i
            assert info["replay_instance_id"] == 1000 + i

    def test_stage_prompts_preserved(self):
        plan = ReplayPlan(
            episode_index=0,
            instance_id=1001,
            actions=np.zeros((5, 23), dtype=np.float32),
            replay_steps=5,
            target_step=5,
            stage_prompts=("step 1", "step 2"),
        )
        infos = replay_plans_to_infos([plan])
        assert infos[0]["replay_stage_prompts"] == ["step 1", "step 2"]


# ---------------------------------------------------------------------------
# BehaviorReplayInitializer config validation
# ---------------------------------------------------------------------------

class TestReplayInitializerConfig:
    def test_disabled_when_no_replay_init(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"seed": 42})
        initializer = maybe_make_replay_initializer(cfg)
        assert initializer is None

    def test_disabled_when_enabled_false(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"replay_init": {"enabled": False}, "seed": 42})
        initializer = maybe_make_replay_initializer(cfg)
        assert initializer is None

    def test_invalid_dataset_root_raises(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "replay_init": {
                    "enabled": True,
                    "dataset_root": "/nonexistent/path/12345",
                },
                "seed": 42,
            }
        )
        with pytest.raises(ValueError, match="dataset_root"):
            BehaviorReplayInitializer(cfg)

    def test_invalid_replay_ratio_raises(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "replay_init": {
                    "enabled": True,
                    "dataset_root": "/tmp",
                    "replay_ratio": -0.5,
                },
                "seed": 42,
            }
        )
        with pytest.raises(ValueError, match="replay_ratio"):
            BehaviorReplayInitializer(cfg)

    def test_invalid_stage_boundary_raises(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "replay_init": {
                    "enabled": True,
                    "dataset_root": "/tmp",
                    "stage_boundary": "middle",
                },
                "seed": 42,
            }
        )
        with pytest.raises(ValueError, match="stage_boundary"):
            BehaviorReplayInitializer(cfg)


# ---------------------------------------------------------------------------
# Episode index parsing
# ---------------------------------------------------------------------------

class TestEpisodeIndexParsing:
    def test_valid_stem(self):
        result = BehaviorReplayInitializer._parse_episode_index("episode_42")
        assert result == 42

    def test_invalid_stem_raises(self):
        with pytest.raises(ValueError, match="Invalid BEHAVIOR episode"):
            BehaviorReplayInitializer._parse_episode_index("wrong_prefix_42")


# ---------------------------------------------------------------------------
# Episode index to instance ID
# ---------------------------------------------------------------------------

class TestEpisodeIndexToInstanceId:
    @pytest.mark.parametrize(
        "task_id,episode_index,expected",
        [
            (0, 10, 1),       # within_task_index=10, 10%10==0 → 10//10=1
            (0, 10001, 10001), # within_task_index=10001, 10001%10!=0 → 10001
            (0, 10020, 1002), # within_task_index=10020, 10020%10==0 → 1002
        ],
    )
    def test_conversion(self, task_id, episode_index, expected):
        initializer = BehaviorReplayInitializer.__new__(BehaviorReplayInitializer)
        initializer.task_id = task_id
        result = initializer._episode_index_to_instance_id(episode_index)
        assert result == expected

    def test_wrong_task_raises(self):
        initializer = BehaviorReplayInitializer.__new__(BehaviorReplayInitializer)
        initializer.task_id = 1
        # episode_index=5 → within_task_index = 5-10000 = -9995, which is <= 0
        with pytest.raises(ValueError, match="does not belong"):
            initializer._episode_index_to_instance_id(5)


# ---------------------------------------------------------------------------
# Action noise index parsing
# ---------------------------------------------------------------------------

class TestActionNoiseIndices:
    def test_default_none(self):
        # This will fail on dataset_root but we just want to check the parse
        # We'll test via the static method directly
        result = BehaviorReplayInitializer._parse_int_sequence(None)
        assert result is None

    def test_parse_range_string(self):
        result = BehaviorReplayInitializer._parse_int_sequence("0-2,5")
        assert result == (0, 1, 2, 5)

    def test_parse_single_int(self):
        result = BehaviorReplayInitializer._parse_int_sequence(3)
        assert result == (3,)


# ---------------------------------------------------------------------------
# parse_reset_payload
# ---------------------------------------------------------------------------


class TestParseResetPayload:
    """Tests for reset_runtime.parse_reset_payload (CPU-only, no env)."""

    @staticmethod
    def _parse(payload):
        from rlinf.envs.behavior.reset_runtime import parse_reset_payload

        return parse_reset_payload(payload)

    def test_none_payload(self):
        """None payload → reset all."""
        indices, instance_ids, is_full = self._parse(None)
        assert indices is None
        assert instance_ids is None
        assert is_full is False

    def test_bool_list_payload(self):
        """list[bool] → reset indices for True elements."""
        indices, instance_ids, is_full = self._parse(
            [True, False, True, False]
        )
        assert indices == [0, 2]
        assert instance_ids is None
        assert is_full is False

    def test_bool_list_all_false(self):
        """All False → empty reset indices."""
        indices, instance_ids, is_full = self._parse([False, False])
        assert indices == []
        assert instance_ids is None

    def test_dict_payload_with_instance_ids(self):
        """list[dict] with instance_id → parsed correctly."""
        payload = [
            {"reset": True, "instance_id": 100},
            {"reset": False},
            {"reset": True, "instance_id": 200},
        ]
        indices, instance_ids, is_full = self._parse(payload)
        assert indices == [0, 2]
        assert instance_ids == [100, 200]
        assert is_full is False

    def test_dict_payload_full_reset(self):
        """All dicts have full_reset=True."""
        payload = [
            {"reset": True, "full_reset": True},
            {"reset": True, "full_reset": True},
        ]
        indices, instance_ids, is_full = self._parse(payload)
        assert indices == [0, 1]
        assert instance_ids is None
        assert is_full is True

    def test_dict_payload_no_instance_ids(self):
        """list[dict] without instance_id → instance_ids=None."""
        payload = [
            {"reset": True},
            {"reset": True, "full_reset": False},
        ]
        indices, instance_ids, is_full = self._parse(payload)
        assert indices == [0, 1]
        assert instance_ids is None
        assert is_full is False

    def test_dict_payload_mixed_instance_ids_raises(self):
        """Some dicts have instance_id, some don't → ValueError."""
        payload = [
            {"reset": True, "instance_id": 100},
            {"reset": True},  # missing instance_id
        ]
        try:
            self._parse(payload)
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "instance_id" in str(e)

    def test_dict_payload_empty(self):
        """Empty dict list → empty reset."""
        indices, instance_ids, is_full = self._parse([])
        assert indices == []
        assert instance_ids is None

    def test_dict_payload_partial_reset_with_instance_ids(self):
        """Only some envs reset, all with instance_ids."""
        payload = [
            {"reset": True, "instance_id": 42},
            {"reset": False},
            {"reset": True, "instance_id": 99},
        ]
        indices, instance_ids, is_full = self._parse(payload)
        assert indices == [0, 2]
        assert instance_ids == [42, 99]
        assert is_full is False
