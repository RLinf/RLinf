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

"""Return-based reset sampling over 1 s bins.

The paper says hard segments are sampled more often. The 50% / 40% / 10%
mixture, the temperature, and the two EMAs are CuboiLeo's record of that
rule, not a paper table:

* 50% uniform over bins;
* 40% ``softmax(-25 * G_b / G_max)``;
* 10% a slow EMA of that softmax.

``G_b`` is an EMA (alpha 0.05) of the discounted return-to-go of the steps
whose reference frame falls in the bin. A timeout bootstraps the tail with
the critic. A failure and a clip end do not.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import EPISODE_TIMEOUT

BIN_FRAMES = 50
UNIFORM_PROB = 0.5
ADAPTIVE_PROB = 0.4
MEMORY_PROB = 0.1
TEMPERATURE = 25.0
RETURN_EMA = 0.05
MEMORY_EMA = 0.01


class BinResetSampler:
    """One sampler shared by every env in the vector."""

    def __init__(
        self,
        lengths: torch.Tensor,
        num_envs: int,
        g_max: float,
        gamma: float,
        max_episode_steps: int,
        device: torch.device,
    ):
        self.device = device
        self.g_max = float(g_max)
        self.gamma = float(gamma)
        self.bin_frames = BIN_FRAMES
        clips: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        bins_per_clip: list[int] = []
        for clip, length in enumerate(lengths.tolist()):
            n_bins = max(1, (int(length) + BIN_FRAMES - 1) // BIN_FRAMES)
            bins_per_clip.append(n_bins)
            for bin_id in range(n_bins):
                start = bin_id * BIN_FRAMES
                end = min(int(length), start + BIN_FRAMES)
                clips.append(clip)
                starts.append(start)
                ends.append(max(end, start + 1))
        self.bin_clip = torch.tensor(clips, dtype=torch.long, device=device)
        self.bin_start = torch.tensor(starts, dtype=torch.long, device=device)
        self.bin_end = torch.tensor(ends, dtype=torch.long, device=device)
        n = len(clips)
        self.bin_return = torch.full((n,), self.g_max, device=device)
        self.bin_memory = torch.full((n,), 1.0 / n, device=device)
        max_local = max(bins_per_clip)
        flat = torch.full((len(bins_per_clip), max_local), -1, dtype=torch.long, device=device)
        local = self.bin_start // BIN_FRAMES
        flat[self.bin_clip, local] = torch.arange(n, device=device)
        self._flat = flat
        self._bins_per_clip = torch.tensor(bins_per_clip, dtype=torch.long, device=device)
        horizon = max(int(max_episode_steps), 1)
        self.ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._reward = torch.zeros(num_envs, horizon, device=device)
        self._bin = torch.zeros(num_envs, horizon, dtype=torch.long, device=device)

    @property
    def num_bins(self) -> int:
        return int(self.bin_return.shape[0])

    def bin_of(self, motion_id: torch.Tensor, motion_time: torch.Tensor) -> torch.Tensor:
        local = motion_time.long() // self.bin_frames
        last = self._bins_per_clip[motion_id] - 1
        local = torch.minimum(local, last).clamp_min(0)
        return self._flat[motion_id, local]

    def log_step(
        self,
        reward: torch.Tensor,
        motion_id: torch.Tensor,
        motion_time: torch.Tensor,
    ) -> None:
        idx = self.ep_len
        room = idx < self._reward.shape[1]
        if not bool(room.any()):
            return
        rows = torch.arange(reward.shape[0], device=self.device)[room]
        self._reward[rows, idx[room]] = reward.detach()[room]
        self._bin[rows, idx[room]] = self.bin_of(motion_id[room], motion_time[room])
        self.ep_len = torch.where(room, idx + 1, idx)

    def mixture(self, update_memory: bool) -> torch.Tensor:
        """Sampling distribution. Memory advances only when a start is drawn."""
        n_bins = self.num_bins
        hard = torch.softmax(-TEMPERATURE * self.bin_return / self.g_max, dim=0)
        memory = self.bin_memory
        if update_memory:
            memory = (1.0 - MEMORY_EMA) * memory + MEMORY_EMA * hard
            self.bin_memory = memory
        probs = (
            UNIFORM_PROB / n_bins
            + ADAPTIVE_PROB * hard
            + MEMORY_PROB * memory / memory.sum().clamp_min(1e-8)
        )
        return probs / probs.sum()

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Bin mixture, then a uniform frame inside the chosen bin."""
        if n <= 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty
        probs = self.mixture(update_memory=True)
        choice = torch.multinomial(probs, n, replacement=True)
        span = (self.bin_end[choice] - self.bin_start[choice]).clamp_min(1)
        motion_time = self.bin_start[choice] + (
            torch.rand(n, device=self.device) * span.float()
        ).long()
        motion_id = self.bin_clip[choice]
        return motion_id, motion_time

    def finish(
        self,
        env_ids: torch.Tensor,
        end_kind: torch.Tensor,
        bootstrap: torch.Tensor | None,
    ) -> None:
        """Fold finished episodes into ``G_b`` and clear their step logs."""
        if env_ids.numel() == 0:
            return
        lens = self.ep_len[env_ids]
        if int(lens.sum()) == 0:
            self.ep_len[env_ids] = 0
            return
        horizon = int(lens.max())
        reward = self._reward[env_ids, :horizon]
        bins = self._bin[env_ids, :horizon]
        steps = torch.arange(horizon, device=self.device)
        valid = steps.unsqueeze(0) < lens.unsqueeze(1)
        # A short episode's tail still holds the previous longer episode until
        # it is masked. The reverse cumsum would otherwise add those rewards.
        reward = torch.where(valid, reward, torch.zeros_like(reward))
        bins = torch.where(valid, bins, torch.zeros_like(bins))
        discount = self.gamma ** torch.arange(horizon, device=self.device, dtype=reward.dtype)
        discounted = reward * discount
        returns = torch.flip(torch.cumsum(torch.flip(discounted, [1]), 1), [1]) / discount
        timeout = end_kind == EPISODE_TIMEOUT
        if timeout.any() and bootstrap is not None:
            remaining = (lens.unsqueeze(1) - steps.unsqueeze(0)).clamp_min(0).to(reward.dtype)
            returns = returns + timeout.unsqueeze(1).to(reward.dtype) * (
                self.gamma**remaining
            ) * bootstrap.unsqueeze(1)
        flat_bins = bins.reshape(-1)[valid.reshape(-1)]
        flat_returns = returns.reshape(-1)[valid.reshape(-1)]
        sums = torch.zeros_like(self.bin_return).index_add_(0, flat_bins, flat_returns)
        counts = torch.zeros_like(self.bin_return).index_add_(
            0, flat_bins, torch.ones_like(flat_returns)
        )
        touched = counts > 0
        mean = sums[touched] / counts[touched]
        self.bin_return[touched] = (1.0 - RETURN_EMA) * self.bin_return[touched] + RETURN_EMA * mean
        self._reward[env_ids] = 0
        self._bin[env_ids] = 0
        self.ep_len[env_ids] = 0

    def reset_statistics(self) -> None:
        """Drop learned bin weights. Used on resume, which refills the replay."""
        self.bin_return.fill_(self.g_max)
        self.bin_memory.fill_(1.0 / max(self.num_bins, 1))
        self.ep_len.zero_()
