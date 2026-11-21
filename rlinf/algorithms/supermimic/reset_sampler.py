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

"""Reset frames drawn toward the parts of the library the policy tracks worst.

The library timeline is cut into 0.2 s bins. ``value[b]`` estimates the
discounted return the policy earns from bin ``b`` on its own simulator
rollouts. The reset distribution mixes a uniform share over bins, a softmax
focus on low-return bins and a keep share on bins that once held a sharp
focus. Follows the return sampler of HybridRobotics/supermimic (Apache-2.0).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from rlinf.algorithms.supermimic.objectives import value_ceiling

BIN_FRAMES = 10
RETURN_GAMMA = 0.99
RECORD_STEPS = 20
MEMORY = 256
STEEPNESS = 25.0
EXPLORE = 0.5
KEEP = 0.1


class ReturnResetSampler:
    """One distribution over library frames shared by every env of every rank.

    Args:
        clip_ends: Exclusive end row of every clip in the library block.
        num_envs: Envs on this rank.
        device: Device of the statistics.
        group: Env ranks whose bin statistics are summed; ``None`` for one.
    """

    def __init__(
        self,
        clip_ends: torch.Tensor,
        num_envs: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ):
        self.device = device
        self.group = group
        self.clip_ends = clip_ends.to(device=device, dtype=torch.long)
        self.frame_count = int(self.clip_ends[-1])
        self.bin_count = self.frame_count // BIN_FRAMES + 1
        self.value_max = value_ceiling(RETURN_GAMMA)
        self.value = torch.full((self.bin_count,), self.value_max, device=device)
        self.count = torch.zeros(self.bin_count, device=device)
        self.effort = torch.zeros(self.bin_count, dtype=torch.float64, device=device)
        self.probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=device)
        shape = (int(num_envs), RECORD_STEPS)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._failed = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._column = 0

    def _bins(self, frames: torch.Tensor) -> torch.Tensor:
        return frames * self.bin_count // self.frame_count

    def sample(self, n: int) -> torch.Tensor:
        """``n`` library frames; a clip's last frame moves one frame back."""
        bins = torch.multinomial(self.probs, n, replacement=True)
        last = self.frame_count - 2
        low = bins * last // self.bin_count
        span = ((bins + 1) * last // self.bin_count - low).clamp(min=1)
        jitter = (torch.rand(n, device=self.device) * span).long()
        frames = (low + jitter).clamp(max=last)
        on_last = torch.isin(frames, self.clip_ends - 1)
        return torch.where(on_last, frames - 1, frames)

    def record(
        self,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        failed: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> None:
        """Buffer one vector step and refresh the distribution every 20 steps.

        ``frames`` are the frames each env was scored against. An episode ends
        by a fall (``failed``), by reaching its clip's last frame or by its
        time limit; one end kind per env. Every rank of ``group`` records in
        lockstep.
        """
        col = self._column
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._failed[:, col] = failed
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._column = (col + 1) % RECORD_STEPS
        if self._column == 0:
            self._fold()
            self._refresh()

    def _returns(self) -> torch.Tensor:
        """Discounted return of every buffered step.

        Where an episode ends, and after the last buffered step, the return
        continues with the value of the following state: none after a fall,
        the mean bin value after a clip's last frame, the next frame's bin
        value otherwise.
        """
        after = (self._frames + 1).clamp(max=self.frame_count - 1)
        following = torch.where(
            self._failed,
            0.0,
            torch.where(
                self._clip_end, self.value.mean(), self.value[self._bins(after)]
            ),
        )
        ended = self._failed | self._clip_end | self._timed_out
        returns = torch.empty_like(self._rewards)
        tail = following[:, -1]
        for t in reversed(range(RECORD_STEPS)):
            tail = torch.where(ended[:, t], following[:, t], tail)
            returns[:, t] = self._rewards[:, t] + RETURN_GAMMA * tail
            tail = returns[:, t]
        return returns

    def _fold(self) -> None:
        """Fold the buffer into the bin values.

        An env's visits to one bin within one episode make one sample, their
        mean return. A bin's earlier samples weigh at most ``MEMORY`` samples.
        """
        returns = self._returns()
        ended = (self._failed | self._clip_end | self._timed_out).long()
        num_envs = ended.shape[0]
        episode = ended.cumsum(1) - ended
        env = torch.arange(num_envs, device=self.device).unsqueeze(1)
        key = (env * RECORD_STEPS + episode) * self.bin_count + self._bins(self._frames)
        keys, inverse, visits = torch.unique(
            key.reshape(-1), return_inverse=True, return_counts=True
        )
        samples = torch.zeros(keys.numel(), device=self.device)
        samples.scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        sums = torch.zeros(2, self.bin_count, device=self.device)
        sums[0].scatter_add_(0, bins, torch.ones_like(samples))
        sums[1].scatter_add_(0, bins, samples)
        if self.group is not None and dist.get_world_size(self.group) > 1:
            dist.all_reduce(sums, group=self.group)
        new, total = sums
        self.count = self.count.clamp(max=MEMORY) + new
        self.value = torch.where(
            new > 0, self.value + (total - new * self.value) / self.count, self.value
        )

    def _refresh(self) -> None:
        focus = torch.softmax(-STEEPNESS * self.value / self.value_max, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        self.probs = (
            EXPLORE / self.bin_count + (1.0 - EXPLORE - KEEP) * focus + KEEP * keep
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Bin values, sample counts, keep effort and the distribution, on CPU."""
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in ("value", "count", "effort", "probs")
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore :meth:`state_dict`; the bin count must match this library."""
        if state["value"].shape != self.value.shape:
            raise ValueError(
                f"checkpoint has {state['value'].shape[0]} reset bins, "
                f"this motion library has {self.bin_count}"
            )
        for name in ("value", "count", "effort", "probs"):
            setattr(self, name, state[name].to(self.device).clone())
