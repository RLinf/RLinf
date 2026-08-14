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

import asyncio
import inspect
from types import SimpleNamespace

import torch

from rlinf.algorithms.rlt.critical_phase_gate import RLT_GATE_INFO_KEYS
from rlinf.data.schema.embodied_types import EnvOutput, PolicyOutput
from rlinf.scheduler import (
    build_recv_plan,
    build_route_channel_key,
    build_send_plan,
    merge_batches,
    split_batch,
)
from rlinf.scheduler.worker.routing import validate_batch_size
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _make_obs(start: int, batch_size: int) -> dict:
    return {
        "states": torch.arange(start, start + batch_size * 2, dtype=torch.float32).view(
            batch_size, 2
        ),
        "main_images": None,
        "wrist_images": None,
        "extra_view_images": None,
        "task_descriptions": [
            f"task-{idx}" for idx in range(start, start + batch_size)
        ],
    }


def test_build_send_plan_load_balance_env_to_rollout():
    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=0,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 4),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=1,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (1, 2),
        (2, 4),
    ]


def test_build_send_plan_load_balance_rollout_to_env():
    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=0,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(0, 4)]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=1,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 2),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=2,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(1, 4)]


def test_build_recv_plan_matches_expected_receive_sizes():
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=0,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 4)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=1,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 2), (1, 2)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=2,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(1, 4)]


def test_build_route_channel_key_is_stable():
    assert build_route_channel_key("env", "rollout", 2, 1, "train") == (
        "scheduler_route",
        "env",
        "rollout",
        "train",
        "",
        2,
        1,
    )
    assert build_route_channel_key("rollout", "env", 0, 3, "eval", "k") == (
        "scheduler_route",
        "rollout",
        "env",
        "eval",
        "k",
        0,
        3,
    )


def test_split_and_merge_nested_batches():
    batch = {
        "obs": _make_obs(0, 6),
        "final_obs": None,
        "rewards": torch.arange(6, dtype=torch.float32).unsqueeze(-1),
    }
    shards = split_batch(batch, [4, 2])
    assert shards[0]["obs"]["states"].shape[0] == 4
    assert len(shards[1]["obs"]["task_descriptions"]) == 2

    merged = merge_batches(shards)
    assert torch.equal(merged["obs"]["states"], batch["obs"]["states"])
    assert merged["obs"]["task_descriptions"] == batch["obs"]["task_descriptions"]
    assert torch.equal(merged["rewards"], batch["rewards"])


def test_policy_output_split_merge_invariant():
    policy_output = PolicyOutput(
        actions=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_logprobs=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        bootstrap_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        intervene_flags=torch.ones((6, 3), dtype=torch.bool),
        forward_inputs={
            "action": torch.arange(12, dtype=torch.float32).view(6, 2),
            "states": torch.arange(18, dtype=torch.float32).view(6, 3),
        },
        versions=torch.arange(6, dtype=torch.float32).view(6, 1),
    )

    worker = object.__new__(MultiStepRolloutWorker)
    shards = worker._split_policy_output(policy_output, [4, 2])
    merged = PolicyOutput.merge(shards)

    assert torch.equal(merged.actions, policy_output.actions)
    assert torch.equal(merged.prev_logprobs, policy_output.prev_logprobs)
    assert torch.equal(merged.prev_values, policy_output.prev_values)
    assert torch.equal(merged.bootstrap_values, policy_output.bootstrap_values)
    assert torch.equal(merged.intervene_flags, policy_output.intervene_flags)
    assert torch.equal(
        merged.forward_inputs["action"], policy_output.forward_inputs["action"]
    )
    assert torch.equal(
        merged.forward_inputs["states"], policy_output.forward_inputs["states"]
    )
    assert torch.equal(merged.versions, policy_output.versions)


def test_evaluate_splits_gate_policy_output():
    class _ImmediateWork:
        async def async_wait(self):
            return {
                "obs": _make_obs(0, 6),
                "final_obs": None,
                "rlt_switch_flags": None,
                "intervene_flags": None,
                "dones": torch.zeros((6, 1), dtype=torch.bool),
            }

    worker = object.__new__(MultiStepRolloutWorker)
    worker.enable_offload = False
    worker.env_decoupled_mode = False
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.num_pipeline_stages = 1
    worker.eval_batch_size = 6
    worker._rank = 0
    worker.cfg = SimpleNamespace(env=SimpleNamespace(group_name="env"))
    worker.rlt_critical_phase_gate = SimpleNamespace(reset=lambda **_: None)
    worker.recv_from = lambda **_: _ImmediateWork()

    actions = torch.arange(12, dtype=torch.float32).view(6, 2)
    gate_info = {
        key: torch.arange(6, dtype=torch.float32).view(6, 1)
        for key in RLT_GATE_INFO_KEYS
    }
    worker._predict_rollout_actions = lambda *_, **__: (
        actions,
        {
            "forward_inputs": gate_info,
            "intervene_flags": torch.zeros((6, 1), dtype=torch.bool),
        },
    )
    sends = []
    worker.send_to = lambda **kwargs: sends.append(kwargs)

    asyncio.run(
        inspect.unwrap(MultiStepRolloutWorker.evaluate)(
            worker,
            input_channel=None,
            output_channel=None,
        )
    )

    assert len(sends) == 1
    assert isinstance(sends[0]["data"], PolicyOutput)
    shards = sends[0]["split_fn"](sends[0]["data"], [4, 2])
    assert [shard.actions.shape[0] for shard in shards] == [4, 2]
    assert all(set(shard.forward_inputs) == set(RLT_GATE_INFO_KEYS) for shard in shards)


def test_env_evaluate_accepts_gate_policy_output():
    class _EvalEnv:
        is_start = False

        def reset(self):
            return _make_obs(0, 6), {}

    worker = object.__new__(EnvWorker)
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.stage_num = 1
    worker.eval_num_envs_per_stage = 6
    worker.eval_batch_size = 6
    worker.env_decoupled_mode = False
    worker.enable_rlt = False
    worker.eval_env_list = [_EvalEnv()]
    worker.eval_prev_done = [None]
    worker.eval_enable_offload = False
    worker.cfg = SimpleNamespace(
        env=SimpleNamespace(
            eval=SimpleNamespace(auto_reset=False),
        ),
        rollout=SimpleNamespace(group_name="rollout"),
    )
    worker.send_to = lambda **_: None

    policy_output = PolicyOutput(
        actions=torch.arange(12, dtype=torch.float32).view(6, 2),
        forward_inputs={},
    )

    def _recv_from(**kwargs):
        validate_batch_size(
            policy_output,
            kwargs["batch_size"],
            kwargs.get("infer_batch_size_fn"),
        )
        return policy_output

    worker.recv_from = _recv_from
    worker.env_evaluate_step = lambda *_: (None, {})
    worker.finish_rollout = lambda **_: None

    result = inspect.unwrap(EnvWorker.evaluate)(
        worker,
        input_channel=None,
        rollout_channel=None,
    )

    assert result == {}


def test_merge_env_outputs_with_partial_optional_fields():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        final_obs=None,
        dones=torch.zeros((2, 1), dtype=torch.bool),
        terminations=torch.zeros((2, 1), dtype=torch.bool),
        truncations=torch.zeros((2, 1), dtype=torch.bool),
        rewards=torch.ones((2, 1), dtype=torch.float32),
        intervene_actions=None,
        intervene_flags=None,
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        final_obs=_make_obs(200, 3),
        dones=torch.zeros((3, 1), dtype=torch.bool),
        terminations=torch.zeros((3, 1), dtype=torch.bool),
        truncations=torch.zeros((3, 1), dtype=torch.bool),
        rewards=torch.ones((3, 1), dtype=torch.float32) * 2,
        intervene_actions=torch.ones((3, 4), dtype=torch.float32),
        intervene_flags=torch.ones((3, 1), dtype=torch.bool),
        rlt_switch_flags=torch.ones((3, 1), dtype=torch.bool),
    ).to_dict()

    merged = EnvOutput.merge_env_outputs([env_output_0, env_output_1])

    assert merged["obs"]["states"].shape[0] == 5
    assert len(merged["obs"]["task_descriptions"]) == 5
    assert merged["rewards"].shape[0] == 5
    assert merged["final_obs"] is not None
    assert torch.equal(merged["final_obs"]["states"][:2], env_output_0["obs"]["states"])
    assert torch.equal(
        merged["final_obs"]["states"][2:], env_output_1["final_obs"]["states"]
    )

    assert merged["intervene_actions"].shape == (5, 4)
    assert torch.equal(
        merged["intervene_actions"][:2], torch.zeros((2, 4), dtype=torch.float32)
    )
    assert merged["intervene_flags"].shape == (5, 1)
    assert torch.equal(
        merged["intervene_flags"][:2], torch.zeros((2, 1), dtype=torch.bool)
    )
    assert merged["rlt_switch_flags"].shape == (5, 1)
    assert torch.equal(
        merged["rlt_switch_flags"][:2], torch.zeros((2, 1), dtype=torch.bool)
    )
