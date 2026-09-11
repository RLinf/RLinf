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
from rlinf.utils.nested_dict_process import split_dict, split_dict_to_chunk


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


def _make_multi_step_builder(
    num_steps: int, batch_size: int
) -> EmbodiedTrajectoryBuilder:
    builder = EmbodiedTrajectoryBuilder()
    for step in range(num_steps):
        base = step * batch_size
        ids = torch.arange(base, base + batch_size)
        builder.curr_obs.append({"sample_ids": ids[:, None].repeat(1, 2)})
        builder.next_obs.append({"sample_ids": ids[:, None].repeat(1, 2)})
        builder.forward_inputs.append({"states": ids[:, None].repeat(1, 3)})
        builder.actions.append(ids[:, None])
        builder.rewards.append(ids)
    return builder


def _count_contiguous_tensors(value, path: str) -> int:
    # Returns how many tensors were checked so callers can pin the expected
    # coverage: an assertion that never reaches a tensor proves nothing.
    if value is None or isinstance(value, (int, str)):
        return 0
    if isinstance(value, torch.Tensor):
        assert value.is_contiguous(), f"{path} is not contiguous"
        return 1
    if isinstance(value, dict):
        return sum(
            _count_contiguous_tensors(item, f"{path}.{key}")
            for key, item in value.items()
        )
    raise AssertionError(f"unexpected value type {type(value)} at {path}")


def test_split_dict_to_chunk_returns_contiguous_tensors():
    # Splitting a contiguous tensor on any dim but the first returns strided
    # views, which P2P communication rejects. Chunks must be materialized.
    batch = {
        "values": torch.arange(24).view(4, 6),
        "nested": {"values": torch.arange(24).view(4, 6) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3, dim=1)

    for index, chunk in enumerate(chunks):
        assert _count_contiguous_tensors(chunk, f"chunk[{index}]") == 2
    assert [chunk["values"].tolist() for chunk in chunks] == [
        [[0, 1], [6, 7], [12, 13], [18, 19]],
        [[2, 3], [8, 9], [14, 15], [20, 21]],
        [[4, 5], [10, 11], [16, 17], [22, 23]],
    ]


def test_split_dict_returns_contiguous_tensors():
    batch = {
        "values": torch.arange(24).view(4, 6),
        "nested": {"values": torch.arange(24).view(4, 6) + 100},
    }

    splits = split_dict(batch, [4, 2], dim=1)

    for index, split in enumerate(splits):
        assert _count_contiguous_tensors(split, f"split[{index}]") == 2
    assert [split["values"].shape[1] for split in splits] == [4, 2]
    assert splits[1]["values"].tolist() == [[4, 5], [10, 11], [16, 17], [22, 23]]


def test_trajectory_chunks_are_contiguous():
    trajectories = _make_multi_step_builder(3, 6).to_splited_trajectories(3)

    for index, trajectory in enumerate(trajectories):
        checked = sum(
            _count_contiguous_tensors(
                getattr(trajectory, field_name), f"trajectory[{index}].{field_name}"
            )
            for field_name in trajectory.__dataclass_fields__
        )
        # actions, rewards, curr_obs, next_obs and forward_inputs all propagate.
        assert checked == 5
    assert [trajectory.actions.shape[1] for trajectory in trajectories] == [2, 2, 2]


def test_trajectory_splits_by_sizes_are_contiguous():
    trajectories = _make_multi_step_builder(3, 6).to_splited_trajectories_by_sizes(
        [4, 2]
    )

    for index, trajectory in enumerate(trajectories):
        checked = sum(
            _count_contiguous_tensors(
                getattr(trajectory, field_name), f"trajectory[{index}].{field_name}"
            )
            for field_name in trajectory.__dataclass_fields__
        )
        # actions, rewards, curr_obs, next_obs and forward_inputs all propagate.
        assert checked == 5
    assert [trajectory.actions.shape[1] for trajectory in trajectories] == [4, 2]
