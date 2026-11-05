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

"""Per-env ring buffer that keeps clean and executed actions separate."""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CONT,
    HISTORY_LEN,
    OBS_DIM,
    STATE_DIM,
    XI_DIM,
)
from rlinf.envs.sim.g1_mimic.obs import analytic_obs


class _SlotMask:
    """Valid ``(env, physical)`` slots as a mask, revised one write at a time.

    ``add`` rewrites only the few slots a new frame can change, and never
    reads the device back. Sampling is uniform over the set slots.
    """

    def __init__(self, num_envs: int, capacity: int, device: torch.device):
        self.device = device
        self.ok = torch.zeros(num_envs, capacity, dtype=torch.bool, device=device)

    def fill(self, env_i: torch.Tensor, phys_i: torch.Tensor) -> None:
        """Replace the mask with one full scan."""
        self.ok.zero_()
        self.ok[env_i.to(self.device), phys_i.to(self.device)] = True

    def entries(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Env ids and physical slots of the set slots."""
        return self.ok.nonzero(as_tuple=True)

    def draw(
        self, batch: int, keep: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """``min(batch, n)`` uniform slots with replacement, and ``n``.

        ``None`` when no slot is set. ``keep`` narrows the mask first. The
        count is the only value read back.
        """
        mask = self.ok if keep is None else self.ok & keep
        total_by_slot = mask.reshape(-1).cumsum(0)
        total = int(total_by_slot[-1])
        if total == 0:
            return None
        n = min(batch, total)
        rank = torch.rand(n, device=self.device, dtype=torch.float64) * total
        rank = rank.long().clamp_max(total - 1)
        flat = torch.searchsorted(total_by_slot, rank, right=True)
        capacity = self.ok.shape[1]
        return flat // capacity, flat % capacity, total


class SuperMimicReplay:
    """Stores transitions written *before* any reset of a finished env.

    Fields needed to rebuild observation and the action-smoothness term:

    * ``action_policy``: clean ``π(o, g)``. Observation at ``t`` uses
      ``action_policy[t-1]``, or 0 at an episode start.
    * ``action_executed``: ``a + ε``, the world-model input.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        obs_dim: int = OBS_DIM,
    ):
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.obs_dim = int(obs_dim)
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_id = torch.zeros(
            num_envs, capacity, dtype=torch.long, device=device
        )
        self.time_in_episode = torch.zeros(
            num_envs, capacity, dtype=torch.long, device=device
        )
        self.state = torch.zeros(
            num_envs, capacity, STATE_DIM, device=device, dtype=dtype
        )
        self.next_state = torch.zeros(
            num_envs, capacity, STATE_DIM, device=device, dtype=dtype
        )
        self.action_policy = torch.zeros(
            num_envs, capacity, ACTION_DIM, device=device, dtype=dtype
        )
        self.action_executed = torch.zeros(
            num_envs, capacity, ACTION_DIM, device=device, dtype=dtype
        )
        # Clean previous action that built this step's observation, stored on
        # the step itself so a later ring overwrite of a_{t-10} cannot zero it.
        self.prev_action = torch.zeros(
            num_envs, capacity, ACTION_DIM, device=device, dtype=dtype
        )
        self.obs_noisy = torch.zeros(
            num_envs, capacity, self.obs_dim, device=device, dtype=dtype
        )
        self.task_id = torch.zeros(num_envs, capacity, dtype=torch.long, device=device)
        self.command = torch.zeros(num_envs, capacity, 3, device=device, dtype=dtype)
        self.command_from_ref = torch.zeros(
            num_envs, capacity, dtype=torch.bool, device=device
        )
        self.reward = torch.zeros(num_envs, capacity, device=device, dtype=dtype)
        self.end_kind = torch.zeros(num_envs, capacity, dtype=torch.long, device=device)
        self.xi = torch.zeros(num_envs, capacity, XI_DIM, device=device, dtype=dtype)
        self.motion_id = torch.zeros(
            num_envs, capacity, dtype=torch.long, device=device
        )
        self.motion_time = torch.zeros(
            num_envs, capacity, dtype=torch.long, device=device
        )
        self.next_episode_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Masks of legal windows and imagination starts. The first query scans
        # the ring once. Later writes revise only the slots that write touches.
        self._windows: _SlotMask | None = None
        self._win_k: int | None = None
        self._starts: _SlotMask | None = None

    def add(
        self,
        env_ids: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action_policy: torch.Tensor,
        action_executed: torch.Tensor,
        reward: torch.Tensor,
        end_kind: torch.Tensor,
        xi: torch.Tensor,
        motion_id: torch.Tensor,
        motion_time: torch.Tensor,
        time_in_episode: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        obs_noisy: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        command: torch.Tensor | None = None,
        command_from_ref: torch.Tensor | None = None,
    ) -> None:
        """Append one vector step for ``env_ids`` at each env's cursor.

        ``prev_action`` defaults to zeros and ``obs_noisy`` to the noiseless
        observation; the remaining optional fields default to the motion task.
        """
        idx = self.cursor[env_ids]
        count_before = self.count[env_ids].clone()
        if prev_action is None:
            prev_action = torch.zeros_like(action_policy)
        if obs_noisy is None:
            obs_noisy = analytic_obs(state, prev_action, xi)
        self.state[env_ids, idx] = state
        self.next_state[env_ids, idx] = next_state
        self.action_policy[env_ids, idx] = action_policy
        self.action_executed[env_ids, idx] = action_executed
        self.prev_action[env_ids, idx] = prev_action
        self.obs_noisy[env_ids, idx] = obs_noisy
        if task_id is None:
            task_id = torch.zeros(
                env_ids.shape[0], dtype=torch.long, device=self.device
            )
        if command is None:
            command = torch.zeros(
                env_ids.shape[0], 3, device=self.device, dtype=self.dtype
            )
        if command_from_ref is None:
            command_from_ref = torch.zeros(
                env_ids.shape[0], dtype=torch.bool, device=self.device
            )
        self.task_id[env_ids, idx] = task_id
        self.command[env_ids, idx] = command
        self.command_from_ref[env_ids, idx] = command_from_ref
        self.reward[env_ids, idx] = reward
        self.end_kind[env_ids, idx] = end_kind
        self.xi[env_ids, idx] = xi
        self.motion_id[env_ids, idx] = motion_id
        self.motion_time[env_ids, idx] = motion_time
        self.time_in_episode[env_ids, idx] = time_in_episode
        self.episode_id[env_ids, idx] = self.next_episode_id[env_ids]
        self.cursor[env_ids] = (idx + 1) % self.capacity
        self.count[env_ids] = torch.minimum(
            self.count[env_ids] + 1, torch.full_like(self.count[env_ids], self.capacity)
        )
        finished = end_kind != EPISODE_CONT
        self.next_episode_id[env_ids] += finished.long()
        self._revise_indexes(env_ids, idx, count_before)

    def _live_physical(self) -> torch.Tensor:
        """Physical index of each logical slot. Shape ``[num_envs, capacity]``."""
        logical = torch.arange(self.capacity, device=self.device).view(1, -1)
        return (
            self.cursor.unsqueeze(1) - self.count.unsqueeze(1) + logical
        ) % self.capacity

    def _window_env_phys(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Env id and physical start of every legal ``k``-step window."""
        if self.capacity < k:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty
        phys = self._live_physical()
        logical = torch.arange(self.capacity, device=self.device).view(1, -1)
        live = logical < self.count.unsqueeze(1)
        ep = self.episode_id.gather(1, phys)
        tm = self.time_in_episode.gather(1, phys)
        ep_w = ep.unfold(1, k, 1)
        tm_w = tm.unfold(1, k, 1)
        live_w = live.unfold(1, k, 1)
        same_ep = (ep_w == ep_w[..., :1]).all(dim=-1)
        expect = tm_w[..., :1] + torch.arange(k, device=self.device)
        same_time = (tm_w == expect).all(dim=-1)
        mask = live_w.all(dim=-1) & same_ep & same_time
        env_i, logical_i = mask.nonzero(as_tuple=True)
        return env_i, phys[env_i, logical_i]

    def _window_mask(self, k: int) -> _SlotMask:
        if self._windows is None or self._win_k != k:
            env_i, phys_i = self._window_env_phys(k)
            self._windows = _SlotMask(self.num_envs, self.capacity, self.device)
            self._windows.fill(env_i, phys_i)
            self._win_k = k
        return self._windows

    def _start_mask(self) -> _SlotMask:
        if self._starts is None:
            env_i, phys_i = self._consistent_starts()
            self._starts = _SlotMask(self.num_envs, self.capacity, self.device)
            self._starts.fill(env_i, phys_i)
        return self._starts

    def _cached_windows(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._window_mask(k).entries()

    def _cached_starts(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._start_mask().entries()

    def _revise_indexes(
        self, env_ids: torch.Tensor, written: torch.Tensor, count_before: torch.Tensor
    ) -> None:
        """Update the slot masks after ``written`` was stored for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        if self._windows is not None and self._win_k is not None:
            self._revise_windows(env_ids, written, count_before, self._win_k)
        if self._starts is not None:
            self._revise_starts(env_ids, written, count_before)

    def _revise_windows(
        self,
        env_ids: torch.Tensor,
        written: torch.Tensor,
        count_before: torch.Tensor,
        k: int,
    ) -> None:
        # Overwriting the oldest frame drops only the window that started
        # there. Every later window keeps its physical start. The new frame
        # adds at most one window, ending on the slot just written.
        full = count_before == self.capacity
        assert self._windows is not None
        mask = self._windows.ok
        mask[env_ids, written] = mask[env_ids, written] & ~full
        start = torch.remainder(written - (k - 1), self.capacity)
        live = self.count[env_ids] >= k
        offsets = torch.arange(k, device=self.device)
        pos = torch.remainder(start.unsqueeze(1) + offsets, self.capacity)
        env = env_ids.unsqueeze(1).expand_as(pos)
        episode = self.episode_id[env, pos]
        moment = self.time_in_episode[env, pos]
        same_ep = (episode == episode[:, :1]).all(dim=-1)
        same_time = (moment == moment[:, :1] + offsets).all(dim=-1)
        ok = live & same_ep & same_time
        mask[env_ids, start] = mask[env_ids, start] | ok

    def _revise_starts(
        self, env_ids: torch.Tensor, written: torch.Tensor, count_before: torch.Tensor
    ) -> None:
        if HISTORY_LEN > self.capacity:
            self._starts = None
            return
        assert self._starts is not None
        # A fresh slot only checks itself. Dropping the oldest frame changes
        # the history check of the next HISTORY_LEN slots, including the slot
        # that now holds the new frame.
        full = count_before == self.capacity
        offsets = torch.arange(HISTORY_LEN, device=self.device)
        phys = torch.remainder(written.unsqueeze(1) + offsets, self.capacity)
        touched = (offsets == 0) | full.unsqueeze(1)
        flat_env = env_ids.unsqueeze(1).expand_as(phys).reshape(-1)
        flat_phys = phys.reshape(-1)
        ok = self._start_slot_ok(flat_env, flat_phys)
        mask = self._starts.ok
        was = mask[flat_env, flat_phys]
        mask[flat_env, flat_phys] = torch.where(touched.reshape(-1), ok, was)

    def _start_slot_ok(self, env: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        """Whether these live slots still have a consecutive in-buffer prefix."""
        newest = torch.remainder(self.cursor[env] - 1, self.capacity)
        dist = torch.remainder(newest - phys, self.capacity)
        ok = dist < self.count[env]
        moment = self.time_in_episode[env, phys]
        episode = self.episode_id[env, phys]
        for age in range(1, HISTORY_LEN):
            prev = torch.remainder(phys - age, self.capacity)
            need = moment >= age
            match = (self.episode_id[env, prev] == episode) & (
                self.time_in_episode[env, prev] == moment - age
            )
            ok = ok & (~need | match)
        return ok

    def num_valid_windows(self, k: int) -> int:
        """Number of ``k``-step windows that stay inside one episode."""
        return int(self._window_mask(k).ok.sum())

    def valid_window_starts(self, k: int) -> list[tuple[int, int]]:
        """``(env, slot)`` of every valid ``k``-step window start."""
        env_i, phys_i = self._cached_windows(k)
        return list(zip(env_i.tolist(), phys_i.tolist(), strict=True))

    def _predecessor_ok(self, env: int, physical: int) -> bool:
        t = int(self.time_in_episode[env, physical])
        if t == 0:
            return False
        prev = (physical - 1) % self.capacity
        return (
            int(self.episode_id[env, prev]) == int(self.episode_id[env, physical])
            and int(self.time_in_episode[env, prev]) == t - 1
        )

    def prev_action_at(self, env: int, physical: int) -> torch.Tensor:
        """Clean previous action. Overwritten ring slots are not predecessors."""
        if not self._predecessor_ok(env, physical):
            return torch.zeros(ACTION_DIM, device=self.device, dtype=self.dtype)
        prev = (physical - 1) % self.capacity
        return self.action_policy[env, prev]

    def _history_index(self, env: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        """Physical slots of the 10-frame window, oldest first.

        A missing older slot repeats the oldest slot that is still the same
        episode and the expected time. The observation stored on that slot
        already contains its previous action.
        """
        t = self.time_in_episode[env, physical]
        ep = self.episode_id[env, physical]
        age = torch.arange(HISTORY_LEN - 1, -1, -1, device=self.device)
        desired = t.unsqueeze(1) - age.view(1, -1)
        pos = torch.remainder(physical.unsqueeze(1) - age.view(1, -1), self.capacity)
        valid = desired >= 0
        t_pos = self.time_in_episode[env.unsqueeze(1), pos]
        ep_pos = self.episode_id[env.unsqueeze(1), pos]
        valid = valid & (t_pos == desired) & (ep_pos == ep.unsqueeze(1))
        # First valid column, then stick to it for any older invalid column.
        first = valid.to(torch.long).argmax(dim=1)
        cols = torch.arange(HISTORY_LEN, device=self.device).view(1, -1)
        use = torch.maximum(cols, first.unsqueeze(1))
        return pos.gather(1, use)

    def history_at(self, env: int, physical: int) -> torch.Tensor:
        """Ten stored observations ending at this slot.

        The frame was written with the previous action and the noise sample
        from that control step. Rebuilding it would redraw noise and would
        lose ``a_{t-10}`` after the ring overwrites that slot.
        """
        env_t = torch.tensor([env], dtype=torch.long, device=self.device)
        phys_t = torch.tensor([physical], dtype=torch.long, device=self.device)
        index = self._history_index(env_t, phys_t)
        frames = self.obs_noisy[env, index[0]]
        return frames.reshape(-1)

    def _consistent_starts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Live slots whose in-buffer prefix still has consecutive times."""
        phys = self._live_physical()
        logical = torch.arange(self.capacity, device=self.device).view(1, -1)
        live = logical < self.count.unsqueeze(1)
        t = self.time_in_episode.gather(1, phys)
        ep = self.episode_id.gather(1, phys)
        ok = live
        for age in range(1, HISTORY_LEN):
            prev = torch.remainder(phys - age, self.capacity)
            t_prev = self.time_in_episode.gather(1, prev)
            ep_prev = self.episode_id.gather(1, prev)
            need = t >= age
            match = (ep_prev == ep) & (t_prev == (t - age))
            ok = ok & (~need | match)
        env_i, logical_i = ok.nonzero(as_tuple=True)
        return env_i, phys[env_i, logical_i]

    def sample_windows(self, k: int, batch: int) -> dict[str, torch.Tensor] | None:
        """Up to ``batch`` random ``k``-step windows, or ``None`` before any exist."""
        drawn = self._window_mask(k).draw(batch)
        if drawn is None:
            return None
        env, start, total = drawn
        b = int(env.shape[0])
        pos = (start.unsqueeze(1) + torch.arange(k, device=self.device)) % self.capacity
        env_k = env.unsqueeze(1).expand_as(pos)
        states = torch.empty(b, k + 1, STATE_DIM, device=self.device, dtype=self.dtype)
        states[:, 0] = self.state[env, start]
        states[:, 1:] = self.next_state[env_k, pos]
        prev_actions = self.prev_action[env, start]
        actions_exec = self.action_executed[env_k, pos]
        end_kind = self.end_kind[env_k, pos]
        xi = self.xi[env, start]
        motion_id = self.motion_id[env, start]
        motion_time = self.motion_time[env, start]
        return {
            "state": states,
            "action_executed": actions_exec,
            "prev_action": prev_actions,
            "end_kind": end_kind,
            "xi": xi,
            "motion_id": motion_id,
            "motion_time": motion_time,
            "task_id": self.task_id[env, start],
            "valid_windows": torch.full((), total, device=self.device),
        }

    def sample_starts(
        self, batch: int, task: int | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Up to ``batch`` imagination starts with their history, or ``None``.

        ``task`` keeps only motion (0) or velocity (1) rows.
        """
        keep = None if task is None else self.task_id == int(task)
        drawn = self._start_mask().draw(batch, keep)
        if drawn is None:
            return None
        picked_env, picked_phys, _ = drawn
        state = self.state[picked_env, picked_phys]
        prev_action = self.prev_action[picked_env, picked_phys]
        xi = self.xi[picked_env, picked_phys]
        motion_id = self.motion_id[picked_env, picked_phys]
        motion_time = self.motion_time[picked_env, picked_phys]
        index = self._history_index(picked_env, picked_phys)
        history = self.obs_noisy[picked_env.unsqueeze(1), index].reshape(
            picked_env.shape[0], -1
        )
        return {
            "state": state,
            "prev_action": prev_action,
            "xi": xi,
            "motion_id": motion_id,
            "motion_time": motion_time,
            "history": history,
            "command": self.command[picked_env, picked_phys],
            "command_from_ref": self.command_from_ref[picked_env, picked_phys],
            "task_id": self.task_id[picked_env, picked_phys],
        }
