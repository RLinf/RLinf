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
"""Unit tests for BehaviorProcessPool seed_offset wiring (CPU-only, no Ray cluster).

Locks in the fix that each ``BehaviorProcess`` actor receives a distinct
``replay_seed_offset = pool.seed_offset + process_idx``, matching the old
subprocess-based repo's ``seed_offset + process_idx`` per-shard semantics.
"""

from unittest.mock import MagicMock

from omegaconf import OmegaConf


def test_seed_offset_threading(monkeypatch):
    from rlinf.envs.behavior.behavior_env import (
        BehaviorProcess,
        BehaviorProcessPool,
    )

    captured = []

    def fake_remote(cfg, num_envs, pipeline_stage_num, replay_seed_offset):
        captured.append(replay_seed_offset)
        handle = MagicMock()
        handle.get_activity_name.remote.return_value = "turning_on_radio"
        return handle

    monkeypatch.setattr(BehaviorProcess, "remote", fake_remote)
    monkeypatch.setattr(
        "rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs
    )

    cfg = OmegaConf.create({"skip_intermediate_obs_in_chunk": False})
    pool = BehaviorProcessPool(
        cfg=cfg,
        total_num_envs=4,
        num_env_subprocess=2,
        pipeline_stage_num=1,
        seed_offset=5,
    )

    assert pool.seed_offset == 5
    assert captured == [5, 6]


def test_seed_offset_defaults_to_zero(monkeypatch):
    from rlinf.envs.behavior.behavior_env import (
        BehaviorProcess,
        BehaviorProcessPool,
    )

    captured = []

    def fake_remote(cfg, num_envs, pipeline_stage_num, replay_seed_offset):
        captured.append(replay_seed_offset)
        handle = MagicMock()
        handle.get_activity_name.remote.return_value = "turning_on_radio"
        return handle

    monkeypatch.setattr(BehaviorProcess, "remote", fake_remote)
    monkeypatch.setattr(
        "rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs
    )

    cfg = OmegaConf.create({"skip_intermediate_obs_in_chunk": False})
    pool = BehaviorProcessPool(
        cfg=cfg,
        total_num_envs=4,
        num_env_subprocess=2,
        pipeline_stage_num=1,
    )

    assert pool.seed_offset == 0
    assert captured == [0, 1]
