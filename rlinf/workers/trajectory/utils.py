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


def assign_trajectory_rank(
    peer_rank: int, peer_world_size: int, trajectory_world_size: int
) -> int:
    if peer_world_size <= 0 or trajectory_world_size <= 0:
        raise ValueError(
            f"Invalid peer_world_size ({peer_world_size}) or "
            f"trajectory_world_size ({trajectory_world_size})."
        )
    return min(
        ((int(peer_rank) + 1) * trajectory_world_size - 1) // peer_world_size,
        trajectory_world_size - 1,
    )


def assign_peer_ranks(
    trajectory_rank: int, peer_world_size: int, trajectory_world_size: int
) -> list[int]:
    if peer_world_size <= 0 or trajectory_world_size <= 0:
        raise ValueError(
            f"Invalid peer_world_size ({peer_world_size}) or "
            f"trajectory_world_size ({trajectory_world_size})."
        )
    return [
        peer_rank
        for peer_rank in range(peer_world_size)
        if assign_trajectory_rank(peer_rank, peer_world_size, trajectory_world_size)
        == int(trajectory_rank)
    ]


def assign_trajectory_ranks(
    peer_rank: int, peer_world_size: int, trajectory_world_size: int
) -> list[int]:
    if peer_world_size <= 0 or trajectory_world_size <= 0:
        raise ValueError(
            f"Invalid peer_world_size ({peer_world_size}) or "
            f"trajectory_world_size ({trajectory_world_size})."
        )
    return [
        trajectory_rank
        for trajectory_rank in range(trajectory_world_size)
        if int(peer_rank)
        in assign_peer_ranks(trajectory_rank, peer_world_size, trajectory_world_size)
    ]
