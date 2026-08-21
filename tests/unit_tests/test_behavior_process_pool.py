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
    monkeypatch.setattr("rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs)

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
    monkeypatch.setattr("rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs)

    cfg = OmegaConf.create({"skip_intermediate_obs_in_chunk": False})
    pool = BehaviorProcessPool(
        cfg=cfg,
        total_num_envs=4,
        num_env_subprocess=2,
        pipeline_stage_num=1,
    )

    assert pool.seed_offset == 0
    assert captured == [0, 1]


def test_reset_payload_shards_follow_interleaved_process_mapping():
    from rlinf.envs.behavior.reset_runtime import build_reset_payload_shards

    payload = [
        {"reset": True, "instance_id": 10},
        {"reset": True, "instance_id": 20},
        {"reset": True, "instance_id": 30},
        {"reset": True, "instance_id": 40},
    ]
    plan = [
        (0, [0, 2], [0, 1]),
        (1, [1, 3], [0, 1]),
    ]

    payload_shards, reset_positions = build_reset_payload_shards(
        payload,
        plan,
        num_env_shard=2,
        num_env_subprocess=2,
    )

    assert [item["instance_id"] for item in payload_shards[0]] == [10, 30]
    assert [item["instance_id"] for item in payload_shards[1]] == [20, 40]
    assert reset_positions == [[0, 2], [1, 3]]


def test_reset_payload_shards_track_sparse_reset_positions():
    from rlinf.envs.behavior.reset_runtime import build_reset_payload_shards

    payload = [False, True, True, False]
    plan = [
        (0, [0, 2], [0, 1]),
        (1, [1, 3], [0, 1]),
    ]

    payload_shards, reset_positions = build_reset_payload_shards(
        payload,
        plan,
        num_env_shard=2,
        num_env_subprocess=2,
    )

    assert payload_shards == [[False, True], [True, False]]
    assert reset_positions == [[2], [1]]


def test_env_reset_slice_partial_merges_sparse_payload_results(monkeypatch):
    from rlinf.envs.behavior.behavior_env import BehaviorProcessPool

    class RemoteMethod:
        def __init__(self, func):
            self.func = func

        def remote(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    class FakeProcess:
        def __init__(self, process_idx):
            self.process_idx = process_idx
            self.payloads = []
            self.reset = RemoteMethod(self._reset)

        def _reset(self, payload):
            self.payloads.append(payload)
            reset_rows = [idx for idx, item in enumerate(payload) if bool(item)]
            return (
                [f"obs-{self.process_idx}-{idx}" for idx in reset_rows],
                [
                    {"process_idx": self.process_idx, "local_row": idx}
                    for idx in reset_rows
                ],
            )

    monkeypatch.setattr("rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs)
    pool = BehaviorProcessPool.__new__(BehaviorProcessPool)
    pool.num_env_subprocess = 2
    pool.num_env_shard = 2
    pool.env_processes = [FakeProcess(0), FakeProcess(1)]

    raw_obs, infos = pool.env_reset_slice_partial(
        global_start=0,
        num_envs=4,
        payload=[False, True, True, False],
    )

    assert pool.env_processes[0].payloads == [[False, True]]
    assert pool.env_processes[1].payloads == [[True, False]]
    assert raw_obs == [None, "obs-1-0", "obs-0-1", None]
    assert infos == [
        None,
        {"process_idx": 1, "local_row": 0},
        {"process_idx": 0, "local_row": 1},
        None,
    ]


def test_behavior_env_initial_reset_uses_fixed_instance_ids():
    from rlinf.envs.behavior.behavior_env import BehaviorEnv

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "is_eval": True,
            "ignore_terminations": False,
            "auto_reset": False,
            "max_episode_steps": 10,
            "use_fixed_reset_state_ids": True,
            "use_rel_reward": False,
            "enable_offload": True,
            "enable_init_offload": False,
            "omni_config": {
                "task": {
                    "activity_instance_id": "10,20,30",
                    "reward_config": {"reward_mode": "task"},
                }
            },
        }
    )
    worker_info = MagicMock()
    worker_info.group_world_size = 1
    env = BehaviorEnv(
        cfg,
        num_envs=2,
        seed_offset=1,
        total_num_processes=2,
        worker_info=worker_info,
        record_metrics=False,
    )

    class FakePool:
        def __init__(self):
            self.calls = []

        def env_reset_slice(self, *_args):
            raise AssertionError("fixed reset ids should use payload reset")

        def env_reset_slice_partial(self, global_start, num_envs, payload):
            self.calls.append((global_start, num_envs, payload))
            return ["obs0", "obs1"], [{"env": 0}, {"env": 1}]

    fake_pool = FakePool()
    env.pool = fake_pool
    env.pool_offset = 7

    raw_obs, infos = env.env_reset()

    assert raw_obs == ["obs0", "obs1"]
    assert infos == [{"env": 0}, {"env": 1}]
    assert env._ordered_reset_instance_ids == [10, 20, 30]
    assert fake_pool.calls == [
        (
            7,
            2,
            [
                {"reset": True, "full_reset": True, "instance_id": 30},
                {"reset": True, "full_reset": True, "instance_id": 10},
            ],
        )
    ]
