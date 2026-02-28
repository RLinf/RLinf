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


class CommMapper:
    """Communication mapping helpers with batch sharding among two worker groups that require fixed rank pairing in communications.

    For example, env and rollout should always use the same rank pair for communications.
    """

    @staticmethod
    def build_channel_key(src_rank: int, dst_rank: int, extra: str) -> str:
        """Build a canonical point-to-point channel key."""
        return f"{src_rank}_{dst_rank}_{extra}"

    @staticmethod
    def get_dst_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, src_rank: int
    ) -> list[tuple[int, int]]:
        """Compute destination ranks and transfer sizes for one source rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= src_rank < src_world_size, (
            f"src_rank ({src_rank}) must be in [0, {src_world_size})."
        )

        batch_size_per_src_rank = batch_size // src_world_size
        batch_size_per_dst_rank = batch_size // dst_world_size

        dst_ranks_and_sizes: list[tuple[int, int]] = []
        batch_begin = src_rank * batch_size_per_src_rank
        batch_end = (src_rank + 1) * batch_size_per_src_rank
        while batch_begin < batch_end:
            dst_rank = batch_begin // batch_size_per_dst_rank
            dst_batch_begin = dst_rank * batch_size_per_dst_rank
            dst_remaining = batch_size_per_dst_rank - (batch_begin - dst_batch_begin)
            src_remaining = batch_end - batch_begin
            dst_size = min(dst_remaining, src_remaining)
            dst_ranks_and_sizes.append((dst_rank, dst_size))
            batch_begin += dst_size
        return dst_ranks_and_sizes

    @staticmethod
    def get_src_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, dst_rank: int
    ) -> list[tuple[int, int]]:
        """Compute source ranks/sizes for one destination rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= dst_rank < dst_world_size, (
            f"dst_rank ({dst_rank}) must be in [0, {dst_world_size})."
        )

        src_ranks_and_sizes: list[tuple[int, int]] = []
        for src_rank in range(src_world_size):
            dst_ranks_and_sizes = CommMapper.get_dst_ranks(
                batch_size=batch_size,
                src_world_size=src_world_size,
                dst_world_size=dst_world_size,
                src_rank=src_rank,
            )
            for mapped_dst_rank, size in dst_ranks_and_sizes:
                if mapped_dst_rank == dst_rank:
                    src_ranks_and_sizes.append((src_rank, size))

        expected_size = batch_size // dst_world_size
        actual_size = sum(size for _, size in src_ranks_and_sizes)
        assert actual_size == expected_size, (
            f"Expected receive size {expected_size} for destination rank {dst_rank}, "
            f"got {actual_size} from mappings {src_ranks_and_sizes}."
        )
        return src_ranks_and_sizes
