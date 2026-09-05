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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
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


class _ResetFakeReward:
    _total_stages = 3
    _stage_defs = [{"name": "first"}, {"name": "second"}, {"name": "third"}]

    def __init__(self):
        self.current_stage_idx = None

    def set_active_stage_index(self, stage_idx):
        self.current_stage_idx = stage_idx


class _ResetFakeChild:
    def __init__(self, index):
        self.index = index
        self.scene = SimpleNamespace(
            get_task_metadata=lambda key: {
                "stage_index": 2,
                "stage_prompts": ["first prompt", "second prompt", "third prompt"],
            }
        )
        self.task = SimpleNamespace(task_reward=_ResetFakeReward())
        self.reset_calls = []

    def reset(self, get_obs=True):
        self.reset_calls.append(get_obs)
        return (f"obs-{self.index}" if get_obs else None, {})


class _ResetFakeLoader:
    def __init__(self):
        self.calls = []

    def prepare_reset(self, env, instance_ids=None, group_size=1):
        self.calls.append(
            ([child.index for child in env.envs], instance_ids, group_size)
        )


def _build_reset_process(num_envs=4):
    from rlinf.envs.behavior.behavior_env import BehaviorProcess

    children = [_ResetFakeChild(index) for index in range(num_envs)]
    vector_env = SimpleNamespace(envs=children)
    vector_env.reset_calls = []

    def reset(**kwargs):
        vector_env.reset_calls.append(kwargs)
        get_obs = kwargs.get("get_obs", True)
        return (
            [f"full-obs-{index}" for index in range(num_envs)] if get_obs else None,
            [{} for _ in range(num_envs)],
        )

    vector_env.reset = reset
    # ``BehaviorProcess`` is a ``@ray.remote`` ActorClass, so ``__new__`` must go
    # through the underlying plain class it wraps (CPU-only tests, no cluster).
    process_cls = BehaviorProcess.__ray_metadata__.modified_class
    process = process_cls.__new__(process_cls)
    process.env = vector_env
    process.instance_loader = _ResetFakeLoader()
    process.group_size = 4
    process.first_chunk_action_override_enabled = True
    process._first_chunk_action_override_pending = np.zeros(num_envs, dtype=bool)
    return process, children


def test_behavior_process_reset_paths_apply_metadata_and_selected_prepare():
    process, children = _build_reset_process()

    raw_obs, infos = process.reset(None)
    assert raw_obs == ["full-obs-0", "full-obs-1", "full-obs-2", "full-obs-3"]
    assert all(info["replay_init"]["replay_stage_idx"] == 2 for info in infos)
    assert all(child.task.task_reward.current_stage_idx == 2 for child in children)
    assert process._first_chunk_action_override_pending.tolist() == [True] * 4
    assert process.instance_loader.calls[-1] == ([0, 1, 2, 3], None, 1)

    process._first_chunk_action_override_pending[:] = False
    raw_obs, infos = process.reset([False, True, False, True])
    assert raw_obs == ["obs-1", "obs-3"]
    assert [info["replay_init"]["replay_stage_idx"] for info in infos] == [2, 2]
    assert process.instance_loader.calls[-1] == ([1, 3], None, 1)
    assert children[0].reset_calls == []
    assert children[2].reset_calls == []
    assert process._first_chunk_action_override_pending.tolist() == [
        False,
        True,
        False,
        True,
    ]

    process._first_chunk_action_override_pending[:] = False
    raw_obs, infos = process.reset([0, 2])
    assert raw_obs == ["obs-0", "obs-2"]
    assert [info["reward"]["task_specific"]["current_stage_idx"] for info in infos] == [2, 2]
    assert process.instance_loader.calls[-1] == ([0, 2], None, 1)

    process._first_chunk_action_override_pending[:] = False
    payload = [
        {"reset": False},
        {"reset": True, "instance_id": 41},
        {"reset": False},
        {"reset": True, "instance_id": 43},
    ]
    raw_obs, infos = process.reset(payload)
    assert raw_obs == ["obs-1", "obs-3"]
    assert [info["replay_init"]["replay_stage_idx"] for info in infos] == [2, 2]
    assert process.instance_loader.calls[-1] == ([1, 3], [41, 43], 1)


def test_behavior_process_reset_without_obs_restores_stage_and_does_not_arm_on_failure():
    process, children = _build_reset_process()

    assert process.reset([False, True, False, True], get_obs=False) == (None, None)
    assert all(
        child.task.task_reward.current_stage_idx == 2 for child in children[1::2]
    )
    assert process._first_chunk_action_override_pending.tolist() == [
        False,
        True,
        False,
        True,
    ]

    process, children = _build_reset_process()
    explicit_payload = [
        {"reset": False},
        {"reset": True, "instance_id": 41},
        {"reset": False},
        {"reset": True, "instance_id": 43},
    ]
    assert process.reset(explicit_payload, get_obs=False) == (None, None)
    assert all(
        child.task.task_reward.current_stage_idx == 2 for child in children[1::2]
    )
    assert process._first_chunk_action_override_pending.tolist() == [False, True, False, True]

    process, children = _build_reset_process()

    def fail_reset(get_obs=True):
        raise RuntimeError("reset failed")

    children[3].reset = fail_reset
    with pytest.raises(RuntimeError, match="reset failed"):
        process.reset([False, True, False, True])
    assert process._first_chunk_action_override_pending.tolist() == [False] * 4


def test_first_chunk_override_consumes_only_current_slice_rows():
    process, _children = _build_reset_process()
    process.first_chunk_action_ids = [0]
    process.first_chunk_action_value = 7.0
    process.action_mask = None
    process.skip_intermediate_obs_in_chunk = False
    process.joint_tracer = MagicMock()
    seen_actions = []

    def fake_step(actions, env_indices, need_obs):
        seen_actions.append(actions.clone())
        count = len(env_indices)
        return (
            [{} for _ in env_indices],
            torch.zeros(count),
            torch.zeros(count, dtype=torch.bool),
            torch.zeros(count, dtype=torch.bool),
            [{} for _ in env_indices],
        )

    process._step_shard = fake_step
    process.reset([False, True, False, True])
    process.chunk_step(torch.zeros((4, 1, 2)), [1])
    assert seen_actions[0][1, 0].item() == 7.0
    assert process._first_chunk_action_override_pending.tolist() == [
        False,
        False,
        False,
        True,
    ]
    process.chunk_step(torch.zeros((4, 1, 2)), [3])
    assert seen_actions[1][3, 0].item() == 7.0
    assert process._first_chunk_action_override_pending.tolist() == [False] * 4
    process.chunk_step(torch.zeros((4, 1, 2)), [1])
    assert seen_actions[2][1, 0].item() == 0.0


def test_replay_dump_jobs_follow_pool_mapping_and_preserve_order(monkeypatch):
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
            self.dump_replay_tro_states = RemoteMethod(self._dump)

        def _dump(self, payload):
            self.payloads.append(payload)
            return [
                {
                    "output_instance_id": job["output_instance_id"],
                    "process_idx": self.process_idx,
                    "local_row": job["env_slot"],
                }
                for job in payload["jobs"]
            ]

    monkeypatch.setattr("rlinf.envs.behavior.behavior_env.ray.get", lambda refs: refs)
    pool = BehaviorProcessPool.__new__(BehaviorProcessPool)
    pool.num_env_subprocess = 2
    pool.num_env_shard = 4
    pool.env_processes = [FakeProcess(0), FakeProcess(1)]
    payload = {
        "jobs": [
            {"output_instance_id": 10, "env_slot": 0},
            {"output_instance_id": 11, "env_slot": 1},
            {"output_instance_id": 12, "env_slot": 2},
            {"output_instance_id": 13, "env_slot": 3},
        ],
        "replay": {},
        "output_dir": "/tmp/replay-dump-test",
    }

    results = pool.dump_replay_tro_states_slice(2, 4, payload)

    assert [
        job["output_instance_id"]
        for job in pool.env_processes[0].payloads[0]["jobs"]
    ] == [10, 12]
    assert [job["env_slot"] for job in pool.env_processes[0].payloads[0]["jobs"]] == [
        1,
        2,
    ]
    assert [
        job["output_instance_id"]
        for job in pool.env_processes[1].payloads[0]["jobs"]
    ] == [11, 13]
    assert [job["env_slot"] for job in pool.env_processes[1].payloads[0]["jobs"]] == [
        1,
        2,
    ]
    assert [result["output_instance_id"] for result in results] == [10, 11, 12, 13]

    single_pool = BehaviorProcessPool.__new__(BehaviorProcessPool)
    single_pool.num_env_subprocess = 1
    single_pool.num_env_shard = 2
    single_process = FakeProcess(0)
    single_pool.env_processes = [single_process]
    single_payload = {
        "jobs": [
            {"output_instance_id": 20},
            {"output_instance_id": 21},
            {"output_instance_id": 22},
        ],
        "replay": {},
        "output_dir": "/tmp/replay-dump-test",
    }

    single_results = single_pool.dump_replay_tro_states_slice(0, 2, single_payload)

    assert [result["output_instance_id"] for result in single_results] == [20, 21, 22]
    assert [job["env_slot"] for job in single_process.payloads[0]["jobs"]] == [0, 1, 0]
