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

from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.utils.nested_dict_process import split_dict_to_chunk


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


def _make_trajectory_builder(batch_size: int) -> EmbodiedTrajectoryBuilder:
    sample_ids = torch.arange(batch_size)
    builder = EmbodiedTrajectoryBuilder()
    builder.curr_obs.append({"sample_ids": sample_ids})
    builder.actions.append(sample_ids[:, None])
    builder.rewards.append(sample_ids)
    return builder


def test_trajectory_chunks_keep_top_level_fields_aligned():
    trajectories = _make_trajectory_builder(10).to_splited_trajectories(3)

    expected_ids = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert [
        trajectory.curr_obs["sample_ids"][0].tolist() for trajectory in trajectories
    ] == expected_ids
    assert [
        trajectory.actions[0, :, 0].tolist() for trajectory in trajectories
    ] == expected_ids
    assert [
        trajectory.rewards[0].tolist() for trajectory in trajectories
    ] == expected_ids


def test_trajectory_split_returns_requested_number_of_chunks():
    trajectories = _make_trajectory_builder(2).to_splited_trajectories(4)

    assert len(trajectories) == 4
    assert [trajectory.actions.shape[1] for trajectory in trajectories] == [1, 1, 0, 0]
    assert [
        trajectory.curr_obs["sample_ids"].shape[1] for trajectory in trajectories
    ] == [1, 1, 0, 0]
