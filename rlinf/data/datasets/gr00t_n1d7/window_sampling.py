# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Uniform legal windows and a checkpointable distributed sample stream."""

import random
from bisect import bisect_right
from itertools import accumulate


class WindowIndex:
    """Map each legal H-frame target to one episode row and start frame."""

    def __init__(self, lengths: list[int], horizon: int = 40):
        if (
            type(horizon) is not int
            or horizon <= 0
            or not lengths
            or any(n < horizon for n in lengths)
        ):
            raise ValueError(
                "Expected nonempty episodes with length >= positive integer horizon"
            )
        self.ends = list(accumulate(n - horizon + 1 for n in lengths))

    def __len__(self) -> int:
        return self.ends[-1]

    def locate(self, index: int) -> tuple[int, int]:
        """Return the episode and start for a flat legal-window index."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = bisect_right(self.ends, index)
        return row, index - (self.ends[row - 1] if row else 0)


class WindowSampler:
    """Shard an infinite shuffled-window stream without dropping epoch tails."""

    def __init__(self, size: int, seed: int, rank: int, world_size: int):
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Expected positive size/world size and rank in range")
        self.size, self.seed, self.rank, self.world_size = size, seed, rank, world_size

    def __iter__(self):
        return _WindowIterator(self)

    def state_dict(self) -> dict:
        """Return sampler identity; the iterator owns its consumed cursor."""
        return {
            "size": self.size,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject a resume with a different sampling identity."""
        if state != self.state_dict():
            raise ValueError("Window sampler identity changed on resume")


class _WindowIterator:
    def __init__(self, sampler: WindowSampler):
        self.sampler = sampler
        self.cursor = 0
        self.cached_epoch = -1
        self.order = []

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, int]:
        sampler = self.sampler
        position = self.cursor * sampler.world_size + sampler.rank
        epoch, offset = divmod(position, sampler.size)
        if epoch != self.cached_epoch:
            self.order = list(range(sampler.size))
            random.Random(f"{sampler.seed}:{epoch}:window").shuffle(self.order)
            self.cached_epoch = epoch
        result = epoch, self.order[offset]
        self.cursor += 1
        return result

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        cursor = state["cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Window cursor must be a nonnegative integer")
        self.cursor, self.cached_epoch = cursor, -1
