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
from rlinf.utils.comm_mapping import CommMapper


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


def test_setup_dst_ranks_env_to_rollout():
    # total batch=12, env_world=2, rollout_world=3
    assert CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, src_rank=0
    ) == [
        (0, 4),
        (1, 2),
    ]
    assert CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, src_rank=1
    ) == [
        (1, 2),
        (2, 4),
    ]


def test_setup_dst_ranks_rollout_to_env():
    # total batch=12, rollout_world=3, env_world=2
    assert CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=3, dst_world_size=2, src_rank=0
    ) == [(0, 4)]
    assert CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=3, dst_world_size=2, src_rank=1
    ) == [
        (0, 2),
        (1, 2),
    ]
    assert CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=3, dst_world_size=2, src_rank=2
    ) == [(1, 4)]


def test_setup_src_ranks_matches_expected_receive_sizes():
    # Reverse lookup for destination ranks when env_world=2, rollout_world=3.
    assert CommMapper.get_src_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, dst_rank=0
    ) == [(0, 4)]
    assert CommMapper.get_src_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, dst_rank=1
    ) == [(0, 2), (1, 2)]
    assert CommMapper.get_src_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, dst_rank=2
    ) == [(1, 4)]


def test_build_channel_key_is_stable():
    assert CommMapper.build_channel_key(2, 1, "train") == "2_1_train"
    assert CommMapper.build_channel_key(0, 3, "eval") == "0_3_eval"


def test_rank_mapping_results_are_stable():
    first_dst = CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, src_rank=0
    )
    second_dst = CommMapper.get_dst_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, src_rank=0
    )
    assert first_dst == second_dst

    first_src = CommMapper.get_src_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, dst_rank=1
    )
    second_src = CommMapper.get_src_ranks(
        batch_size=12, src_world_size=2, dst_world_size=3, dst_rank=1
    )
    assert first_src == second_src


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
