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

from rlinf.scheduler import (
    build_recv_plan,
    build_route_channel_key,
    build_send_plan,
    merge_batches,
    split_batch,
)


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
