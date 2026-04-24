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

import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.scheduler.worker.routing import (
    build_recv_plan,
    build_route_channel_key,
    build_send_plan,
    merge_batches,
    split_batch,
)
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
        local_batch_size=6,
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
        local_batch_size=6,
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
        local_batch_size=4,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(0, 4)]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=1,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        local_batch_size=4,
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
        local_batch_size=4,
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
            local_batch_size=4,
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
            local_batch_size=4,
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
            local_batch_size=4,
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


def test_rollout_result_split_merge_invariant():
    rollout_result = RolloutResult(
        actions=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_logprobs=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        bootstrap_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        save_flags=torch.ones((6, 3), dtype=torch.bool),
        forward_inputs={
            "action": torch.arange(12, dtype=torch.float32).view(6, 2),
            "states": torch.arange(18, dtype=torch.float32).view(6, 3),
        },
        versions=torch.arange(6, dtype=torch.float32).view(6, 1),
    )

    worker = object.__new__(MultiStepRolloutWorker)
    shards = worker._split_rollout_result(rollout_result, [4, 2])
    merged = RolloutResult.merge_rollout_results(shards)

    assert torch.equal(merged.actions, rollout_result.actions)
    assert torch.equal(merged.prev_logprobs, rollout_result.prev_logprobs)
    assert torch.equal(merged.prev_values, rollout_result.prev_values)
    assert torch.equal(merged.bootstrap_values, rollout_result.bootstrap_values)
    assert torch.equal(merged.save_flags, rollout_result.save_flags)
    assert torch.equal(merged.forward_inputs["action"], rollout_result.forward_inputs["action"])
    assert torch.equal(merged.forward_inputs["states"], rollout_result.forward_inputs["states"])
    assert torch.equal(merged.versions, rollout_result.versions)


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
