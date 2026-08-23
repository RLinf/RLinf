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

"""Training helpers scoped exclusively to StreamingVLA."""

import random

import numpy as np
import torch

from .sfp import sample_beta


def _normalize_device(device: torch.device | str | int) -> torch.device:
    """Normalize RLinf's integer accelerator index to a PyTorch device."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def seed_streamingvla_training(seed: int, rank: int) -> int:
    """Match OpenPI PyTorch's per-rank ``seed + local_rank`` RNG behavior."""
    normalized_seed = int(seed) + int(rank)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed % (2**32))
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)
    return normalized_seed


def streamingvla_step_seed(seed: int, rank: int, global_step: int) -> int:
    """Return a stable per-rank seed for one StreamingVLA optimizer step."""
    if global_step < 0:
        raise ValueError(f"global_step must be non-negative, got {global_step}.")
    return int(seed) + int(rank) + 1_000_003 * int(global_step)


def sample_streamingvla_step_inputs(
    *,
    seed: int,
    rank: int,
    global_step: int,
    local_batch_size: int,
    action_dim: int,
    device: torch.device | str | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one partition-invariant local batch of SFP time and noise.

    RLinf may split a rank-local batch across multiple gradient-accumulation
    forwards, while the OpenPI reference consumes it in one forward. Sampling
    the complete local batch once preserves the reference distribution and
    makes the random inputs independent of the micro-batch partition.
    """
    if local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive.")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive.")

    normalized_device = _normalize_device(device)
    fork_devices: list[int] = []
    if normalized_device.type == "cuda":
        fork_devices = [
            torch.cuda.current_device()
            if normalized_device.index is None
            else normalized_device.index
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(streamingvla_step_seed(seed, rank, global_step))
        time = (
            sample_beta(1.5, 1.0, local_batch_size, normalized_device) * 0.999 + 0.001
        )
        noise = torch.randn(
            (local_batch_size, 1, action_dim),
            dtype=torch.float32,
            device=normalized_device,
        )
    return time.float(), noise


class StreamingVLAStepInputBuffer:
    """Serve one deterministic rank-local SFP batch across micro-batches."""

    def __init__(
        self,
        *,
        seed: int,
        rank: int,
        local_batch_size: int,
        action_dim: int,
        device: torch.device | str | int,
    ) -> None:
        self._seed = int(seed)
        self._rank = int(rank)
        self._local_batch_size = int(local_batch_size)
        self._action_dim = int(action_dim)
        self._device = _normalize_device(device)
        self._step = 0
        self._time: torch.Tensor | None = None
        self._noise: torch.Tensor | None = None
        self._offset = 0

    @property
    def step(self) -> int:
        """Return the optimizer step whose inputs will be served next."""
        return self._step

    def set_step(self, global_step: int) -> None:
        """Select an optimizer step and discard any partially consumed batch."""
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}.")
        self._step = int(global_step)
        self._time = None
        self._noise = None
        self._offset = 0

    def next_micro_batch(
        self, micro_batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next micro-batch slice and advance after a full local batch."""
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if self._time is None or self._noise is None:
            self._time, self._noise = sample_streamingvla_step_inputs(
                seed=self._seed,
                rank=self._rank,
                global_step=self._step,
                local_batch_size=self._local_batch_size,
                action_dim=self._action_dim,
                device=self._device,
            )

        start = self._offset
        end = start + int(micro_batch_size)
        if end > self._local_batch_size:
            raise RuntimeError(
                "StreamingVLA micro-batches exceed the rank-local batch: "
                f"end={end}, local_batch={self._local_batch_size}."
            )
        time = self._time[start:end]
        noise = self._noise[start:end]
        self._offset = end

        if end == self._local_batch_size:
            self._time = None
            self._noise = None
            self._offset = 0
            self._step += 1
        return time, noise
