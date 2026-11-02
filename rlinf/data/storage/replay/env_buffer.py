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

"""Replay buffers of a run's env configurations, sampled as one buffer."""

import math
import os
from functools import reduce

import torch

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
from rlinf.utils.nested_dict_process import concat_batch


class EnvReplayBuffer:
    """One :class:`TrajectoryReplayBuffer` per env configuration, used as one.

    A trajectory goes to the buffer of the env it came from, and a sampled batch
    takes from each env in proportion to its weight, so how full each buffer
    is never shifts the mix. A homogeneous run has a single buffer named
    ``None``, which behaves exactly like the plain buffer and keeps its
    checkpoint at the given path.
    """

    def __init__(
        self,
        buffers: dict[str | None, TrajectoryReplayBuffer],
        weights: dict[str | None, float],
        *,
        mix_envs: bool = True,
        seed: int = 0,
    ) -> None:
        """Wrap per-env buffers.

        Args:
            buffers: Buffer of each env, keyed by env name.
            weights: Sampling weight of each env, keyed like ``buffers``.
            mix_envs: Whether a batch draws from every env; otherwise each batch
                comes from one env, picked by weight.
            seed: Seed of the env pick, shared by all actor ranks so they pick
                the same env.
        """
        assert buffers.keys() == weights.keys(), (
            f"Expected a weight for each env buffer {list(buffers)}, got weights "
            f"for {list(weights)}."
        )
        total = sum(weights.values())
        self.buffers = buffers
        self.shares = {name: weight / total for name, weight in weights.items()}
        self.mix_envs = mix_envs
        self._generator = torch.Generator().manual_seed(seed)

    def add_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Add each trajectory to the buffer of its env."""
        by_env: dict[str | None, list[Trajectory]] = {}
        for trajectory in trajectories:
            by_env.setdefault(trajectory.env_name, []).append(trajectory)
        for name, items in by_env.items():
            self.buffers[name].add_trajectories(items)

    def sample(self, num_chunks: int) -> dict[str, torch.Tensor]:
        """Sample ``num_chunks`` transitions, split among the envs by weight.

        Each transition carries ``env_index``, the position of its env in the
        buffer order, which matches the env layout's order.
        """
        parts = []
        for index, (name, count) in enumerate(self._counts(num_chunks).items()):
            if count == 0:
                continue
            part = self.buffers[name].sample(count)
            rows = next(
                value.shape[0]
                for value in part.values()
                if isinstance(value, torch.Tensor)
            )
            part["env_index"] = torch.full((rows,), index, dtype=torch.long)
            parts.append(part)
        return reduce(concat_batch, parts)

    def is_ready(self, min_size: int) -> bool:
        """Return whether every env holds its share of ``min_size`` trajectories."""
        return all(
            buffer.is_ready(math.ceil(min_size * self.shares[name]))
            for name, buffer in self.buffers.items()
        )

    async def is_ready_async(self, min_size: int) -> bool:
        """Async form of :meth:`is_ready`."""
        return self.is_ready(min_size)

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self.buffers.values())

    def get_stats(self) -> dict[str, float]:
        """Return the summed buffer stats, plus a ``<name>/`` copy per named env."""
        stats: dict[str, float] = {}
        for name, buffer in self.buffers.items():
            for key, value in buffer.get_stats().items():
                stats[key] = stats.get(key, 0) + value
                if name is not None:
                    stats[f"{name}/{key}"] = value
        return stats

    def save_checkpoint(self, save_path: str) -> None:
        """Save each env's buffer under ``save_path/<name>``."""
        for name, buffer in self.buffers.items():
            buffer.save_checkpoint(self._path(save_path, name))

    def load_checkpoint(self, load_path: str, **kwargs) -> None:
        """Load each env's buffer from ``load_path/<name>``."""
        for name, buffer in self.buffers.items():
            buffer.load_checkpoint(self._path(load_path, name), **kwargs)

    def clear(self) -> None:
        """Clear every env's buffer."""
        for buffer in self.buffers.values():
            buffer.clear()

    def close(self, wait: bool = True) -> None:
        """Close every env's buffer."""
        for buffer in self.buffers.values():
            buffer.close(wait=wait)

    @staticmethod
    def _path(path: str, name: str | None) -> str:
        return path if name is None else os.path.join(path, name)

    def _counts(self, num_chunks: int) -> dict[str | None, int]:
        if not self.mix_envs:
            names = list(self.shares)
            index = torch.multinomial(
                torch.tensor(list(self.shares.values())), 1, generator=self._generator
            ).item()
            return {
                name: num_chunks if i == index else 0 for i, name in enumerate(names)
            }
        exact = {name: num_chunks * share for name, share in self.shares.items()}
        counts = {name: int(value) for name, value in exact.items()}
        leftover = num_chunks - sum(counts.values())
        for name in sorted(exact, key=lambda n: counts[n] - exact[n])[:leftover]:
            counts[name] += 1
        return counts
