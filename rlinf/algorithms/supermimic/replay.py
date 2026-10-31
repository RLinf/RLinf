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
        # Legal windows do not change until the next ``add``. One update
        # draws many batches from the same buffer, so the scan is reused.
        self._window_cache: tuple | None = None
        self._start_cache: tuple | None = None

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
        if finished.any():
            self.next_episode_id[env_ids[finished]] += 1
        self._window_cache = None
        self._start_cache = None

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

    def _cached_windows(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._window_cache
        if cached is not None and cached[0] == k:
            return cached[1], cached[2]
        env_i, phys_i = self._window_env_phys(k)
        self._window_cache = (k, env_i, phys_i)
        return env_i, phys_i

    def _cached_starts(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._start_cache is None:
            self._start_cache = self._consistent_starts()
        return self._start_cache

    def num_valid_windows(self, k: int) -> int:
        """Number of ``k``-step windows that stay inside one episode."""
        env_i, _ = self._cached_windows(k)
        return int(env_i.shape[0])

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

    def history_at(self, env: int, physical: int, q0: torch.Tensor) -> torch.Tensor:
        """Ten stored observations ending at this slot. ``q0`` is unused.

        The frame was written with the previous action and the noise sample
        from that control step. Rebuilding it would redraw noise and would
        lose ``a_{t-10}`` after the ring overwrites that slot.
        """
        del q0
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
        env_all, phys_all = self._cached_windows(k)
        if env_all.numel() == 0:
            return None
        choice = torch.randint(
            env_all.shape[0],
            (min(batch, int(env_all.shape[0])),),
            device=self.device,
        )
        env = env_all[choice]
        start = phys_all[choice]
        b = int(env.shape[0])
        pos = (start.unsqueeze(1) + torch.arange(k, device=self.device)) % self.capacity
        env_k = env.unsqueeze(1).expand_as(pos)
        states = torch.empty(b, k + 1, STATE_DIM, device=self.device, dtype=self.dtype)
        states[:, 0] = self.state[env, start]
        states[:, 1:] = self.next_state[env_k, pos]
        prev_actions = self.prev_action[env, start]
        actions_exec = self.action_executed[env_k, pos]
        actions_policy = self.action_policy[env_k, pos]
        rewards = self.reward[env_k, pos]
        end_kind = self.end_kind[env_k, pos]
        xi = self.xi[env, start]
        motion_id = self.motion_id[env, start]
        motion_time = self.motion_time[env, start]
        return {
            "state": states,
            "action_executed": actions_exec,
            "action_policy": actions_policy,
            "prev_action": prev_actions,
            "reward": rewards,
            "end_kind": end_kind,
            "xi": xi,
            "motion_id": motion_id,
            "motion_time": motion_time,
            "task_id": self.task_id[env, start],
            "valid_windows": torch.tensor(int(env_all.shape[0]), device=self.device),
        }

    def sample_starts(
        self, batch: int, q0: torch.Tensor, task: int | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Up to ``batch`` imagination starts with their history, or ``None``.

        ``task`` keeps only motion (0) or velocity (1) rows.
        """
        env_all, phys_all = self._cached_starts()
        if task is not None and env_all.numel():
            keep = self.task_id[env_all, phys_all] == int(task)
            env_all = env_all[keep]
            phys_all = phys_all[keep]
        if env_all.numel() == 0:
            return None
        choice = torch.randint(
            env_all.shape[0],
            (min(batch, int(env_all.shape[0])),),
            device=self.device,
        )
        picked_env = env_all[choice]
        picked_phys = phys_all[choice]
        del q0
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
