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

"""Per-env ring of simulator transitions with episode-respecting windows."""

from __future__ import annotations

from typing import NamedTuple

import torch

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, SIM_CONTEXT_DIM

# Name -> (trailing width, dtype). Width 0 stores one scalar per transition.
TRANSITION_FIELDS = {
    "action": (ACTION_DIM, torch.float32),
    "prev_action": (ACTION_DIM, torch.float32),
    "motion_time": (0, torch.long),
    "terminated": (0, torch.bool),
    "truncated": (0, torch.bool),
    "sim_context": (SIM_CONTEXT_DIM, torch.float32),
}


class WindowStarts(NamedTuple):
    """Legal window starts of one window length, from :meth:`TransitionRing.windows`.

    ``cumulative[e, a]`` counts env ``e``'s legal starts up to age ``a``,
    ``per_env`` is its last column and ``oldest`` the ring column of age 0.
    """

    length: int
    oldest: int
    cumulative: torch.Tensor
    per_env: torch.Tensor


class TransitionRing:
    """All envs write one transition per vector step into the same column.

    ``state`` is the representation's full state before the step, ``action``
    the executed (noisy) action, ``prev_action`` the action executed one step
    earlier (zero after a reset), ``motion_time`` the library frame the step
    was scored against. A window may end on a step that finished an episode
    but never contains one before its last step.
    """

    def __init__(
        self, num_envs: int, capacity: int, state_dim: int, device: torch.device
    ):
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.device = device
        self.size = 0
        self.column = 0
        fields = {"state": (int(state_dim), torch.float32), **TRANSITION_FIELDS}
        self.storage = {
            name: torch.zeros(
                (self.num_envs, self.capacity, width)
                if width
                else (self.num_envs, self.capacity),
                dtype=dtype,
                device=device,
            )
            for name, (width, dtype) in fields.items()
        }

    def add(self, step: dict[str, torch.Tensor]) -> None:
        """Write one vector step; ``step`` holds every stored field for all envs."""
        if set(step) != set(self.storage):
            raise ValueError(
                f"transition fields {sorted(step)} != {sorted(self.storage)}"
            )
        for name, value in step.items():
            self.storage[name][:, self.column] = value.to(self.device)
        self.column = (self.column + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def _legal_starts(self, length: int) -> tuple[torch.Tensor, int]:
        """Legal window starts ``[num_envs, capacity]`` by age, and the column
        of age 0."""
        if length > self.capacity:
            raise ValueError(
                f"window {length} exceeds the ring capacity {self.capacity}"
            )
        cap, size = self.capacity, self.size
        oldest = self.column if size == cap else 0
        ages = torch.arange(cap, device=self.device)
        ended = (self.storage["terminated"] | self.storage["truncated"])[
            :, (oldest + ages) % cap
        ].long()
        # Steps a .. a+length-2 hold no episode end iff the running end count
        # agrees at a and a+length-1; the padding keeps late ages in range.
        running = torch.cat(
            (
                ended.new_zeros(self.num_envs, 1),
                ended.cumsum(1),
                ended.new_zeros(self.num_envs, length - 1),
            ),
            dim=1,
        )
        legal = (running[:, length - 1 : length - 1 + cap] == running[:, :cap]) & (
            ages + length - 1 < size
        )
        return legal, oldest

    def has_window(self, length: int) -> bool:
        """Whether some env holds a legal ``length``-step window."""
        if self.size < length:
            return False
        if length <= 1:
            return True
        # Fast path: check the newest ``2 * length`` steps before scanning the full ring.
        tail = min(self.size, 2 * length)
        if tail < self.capacity:
            start = (self.column - tail) % self.capacity
            cols = (start + torch.arange(tail, device=self.device)) % self.capacity
            ended = (self.storage["terminated"] | self.storage["truncated"])[
                :, cols
            ].long()
            running = torch.cat((ended.new_zeros(self.num_envs, 1), ended.cumsum(1)), 1)
            # Steps i .. i+length-2 have no episode end iff running[:, i+length-1] == running[:, i].
            if bool(
                (running[:, length - 1 : tail] == running[:, : tail - length + 1]).any()
            ):
                return True
            if tail == self.size:
                return False
        return bool(self._legal_starts(length)[0].any())

    def windows(self, length: int) -> WindowStarts:
        """The legal ``length``-step window starts, checked once.

        Valid until the next :meth:`add`; :meth:`draw` reuses it for every
        batch drawn in between.
        """
        legal, oldest = self._legal_starts(length)
        cumulative = legal.cumsum(1)
        per_env = cumulative[:, -1]
        if not bool((per_env > 0).any()):
            raise RuntimeError(f"no legal {length}-step window among {self.size} steps")
        return WindowStarts(length, oldest, cumulative, per_env)

    def draw(
        self,
        starts: WindowStarts,
        batch: int,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``batch`` windows from ``starts``, ``[batch, length, ...]``.

        Envs with at least one legal window are drawn uniformly, then a
        uniform legal start within the env.
        """
        env = torch.multinomial((starts.per_env > 0).float(), batch, replacement=True)
        pick = (
            torch.rand(batch, device=self.device) * starts.per_env[env].float()
        ).long()
        start = torch.searchsorted(
            starts.cumulative[env], pick.unsqueeze(-1), right=True
        )
        cols = (
            starts.oldest + start + torch.arange(starts.length, device=self.device)
        ) % self.capacity
        rows = env.unsqueeze(-1)
        selected = (
            self.storage if fields is None else {k: self.storage[k] for k in fields}
        )
        return {name: value[rows, cols] for name, value in selected.items()}

    def sample(
        self,
        length: int,
        batch: int,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``batch`` windows of ``length`` consecutive steps; see :meth:`draw`."""
        if length == 1:
            if self.size < 1:
                raise RuntimeError(f"no legal 1-step window among {self.size} steps")
            oldest = self.column if self.size == self.capacity else 0
            rows = torch.randint(0, self.num_envs, (batch, 1), device=self.device)
            cols = (
                oldest + torch.randint(0, self.size, (batch, 1), device=self.device)
            ) % self.capacity
            selected = (
                self.storage if fields is None else {k: self.storage[k] for k in fields}
            )
            return {name: value[rows, cols] for name, value in selected.items()}
        return self.draw(self.windows(length), batch, fields=fields)
