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

    def sample(self, length: int, batch: int) -> dict[str, torch.Tensor]:
        """``batch`` windows of ``length`` consecutive steps, ``[batch, length, ...]``.

        Envs with at least one legal window are drawn uniformly, then a
        uniform legal start within the env.
        """
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
        cumulative = legal.cumsum(1)
        per_env = cumulative[:, -1]
        if not bool((per_env > 0).any()):
            raise RuntimeError(f"no legal {length}-step window among {size} steps")
        env = torch.multinomial((per_env > 0).float(), batch, replacement=True)
        pick = (torch.rand(batch, device=self.device) * per_env[env].float()).long()
        start = torch.searchsorted(cumulative[env], pick.unsqueeze(-1), right=True)
        cols = (oldest + start + torch.arange(length, device=self.device)) % cap
        rows = env.unsqueeze(-1)
        return {name: value[rows, cols] for name, value in self.storage.items()}
