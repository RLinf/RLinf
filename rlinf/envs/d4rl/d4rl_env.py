# Copyright 2025 The RLinf Authors.
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

"""D4RL environment wrapper compatible with ``EnvWorker``.

The wrapper exposes the same high-level interface used by other embodied envs:
- ``reset() -> (obs_dict, infos)``
- ``chunk_step(chunk_actions) -> (obs_list, rewards, terminations, truncations, infos_list)``

Observation schema is normalized to ``{"states": Tensor}``, and auto-reset metadata
uses the same keys as other env implementations (for example ``final_observation``,
``final_info``, ``_final_observation``).
"""

import time
from typing import Any

import gym
import d4rl
import numpy as np
import torch

__all__ = ["D4RLEnv"]


def _compat_reset(env: gym.Env) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _compat_step(env: gym.Env, action: np.ndarray) -> tuple:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info
    return out


class D4RLEnv:
    """D4RL env wrapper compatible with EnvWorker chunk API."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ):
        _ = total_num_processes
        _ = worker_info
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self._env_name = getattr(cfg, "env_name", None) or (
            cfg.get("env_name") if hasattr(cfg, "get") else None
        )
        if self._env_name is None:
            raise ValueError("D4RLEnv requires cfg.env_name.")
        if hasattr(cfg, "seed"):
            base_seed = cfg.seed
        elif hasattr(cfg, "get"):
            base_seed = cfg.get("seed", None)
        else:
            base_seed = None
        if base_seed is None:
            raise ValueError("D4RLEnv requires cfg.seed (int).")
        self._seed = int(base_seed) + int(seed_offset)
        self._auto_reset = bool(getattr(cfg, "auto_reset", True))

        self._record_metrics = bool(record_metrics)
        self._envs: list[gym.Env] = []
        for i in range(self.num_envs):
            seed = self._seed + i
            e = gym.make(self._env_name)
            try:
                e.reset(seed=seed)
            except TypeError:
                if hasattr(e, "seed"):
                    e.seed(seed)
            e.action_space.seed(seed)
            e.observation_space.seed(seed)
            self._envs.append(e)

        self._is_start = True
        self._last_obs: np.ndarray | None = None
        # Per-env episode stats (used when record_metrics=True).
        self._reward_sum = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_length = np.zeros((self.num_envs,), dtype=np.int64)
        self._start_time = np.array([time.time()] * self.num_envs, dtype=np.float64)
        self._total_timesteps = 0

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    def close(self) -> None:
        for e in self._envs:
            e.close()

    def update_reset_state_ids(self) -> None:
        # No-op for D4RL.
        return None

    @staticmethod
    def _to_states_obs(obs: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert raw state observation to EnvWorker-compatible observation dict."""
        return {"states": torch.as_tensor(obs, dtype=torch.float32)}

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all env instances and return a batch observation dict."""
        obs_list = [_compat_reset(e) for e in self._envs]
        obs = np.stack(obs_list, axis=0).astype(np.float32)
        self._last_obs = obs
        # Keep the start flag semantics aligned with other envs: reset finishes
        # initialization and the next call should be treated as in-episode step.
        self._is_start = False
        if self._record_metrics:
            self._reward_sum.fill(0.0)
            self._episode_length.fill(0)
            self._start_time[:] = time.time()
        return self._to_states_obs(obs), {"final_observation": None}

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Step a chunk of actions and return EnvWorker-compatible batch outputs."""
        acts = (
            chunk_actions.detach().cpu().numpy()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        if acts.ndim == 2:
            acts = acts[:, None, :]
        b, k, _ = acts.shape
        assert b == self.num_envs, f"Expected batch={self.num_envs}, got {b}."

        obs_dim = int(self._last_obs.shape[-1]) if self._last_obs is not None else None
        if obs_dim is None:
            self.reset()
            obs_dim = int(self._last_obs.shape[-1])

        rewards = np.zeros((b, k), dtype=np.float32)
        terminations = np.zeros((b, k), dtype=bool)
        truncations = np.zeros((b, k), dtype=bool)
        next_obs_last = np.zeros((b, obs_dim), dtype=np.float32)
        ep_returns = np.zeros((b,), dtype=np.float32)
        ep_lengths = np.zeros((b,), dtype=np.float32)
        done_last = np.zeros((b,), dtype=bool)

        for t in range(k):
            for i, e in enumerate(self._envs):
                o, r, done, info = _compat_step(e, acts[i, t])
                if self._record_metrics:
                    self._reward_sum[i] += float(r)
                    self._episode_length[i] += 1
                    self._total_timesteps += 1
                    info = dict(info)
                    info["total"] = {"timesteps": self._total_timesteps}
                rewards[i, t] = float(r)
                is_trunc = bool(info.get("TimeLimit.truncated", False))
                truncations[i, t] = is_trunc and done
                terminations[i, t] = (not is_trunc) and done
                if t == k - 1:
                    next_obs_last[i] = np.asarray(o, dtype=np.float32)
                    done_last[i] = bool(done)
                if done:
                    if self._record_metrics:
                        ep_ret = float(self._reward_sum[i])
                        if hasattr(e, "get_normalized_score"):
                            try:
                                ep_ret = float(e.get_normalized_score(ep_ret) * 100.0)
                            except Exception:
                                pass
                        ep_returns[i] = ep_ret
                        ep_lengths[i] = float(self._episode_length[i])
                        # Reset stats for next episode after auto reset.
                        self._reward_sum[i] = 0.0
                        self._episode_length[i] = 0
                        self._start_time[i] = time.time()
                    if self._auto_reset:
                        o_reset = _compat_reset(e)
                        if t == k - 1:
                            next_obs_last[i] = np.asarray(o_reset, dtype=np.float32)

        self._last_obs = next_obs_last
        self._is_start = False

        obs_dict = self._to_states_obs(next_obs_last)
        # Align with other envs: keep a full-batch pre-reset snapshot and
        # use masks to indicate which envs actually finished in this chunk.
        final_obs = obs_dict["states"].clone()
        infos: dict[str, Any] = {"final_observation": {"states": final_obs}}
        if bool(np.any(done_last)):
            infos["final_info"] = {
                "episode": {
                    "return": torch.as_tensor(ep_returns, dtype=torch.float32),
                    "length": torch.as_tensor(ep_lengths, dtype=torch.float32),
                }
            }
            done_mask = torch.as_tensor(done_last, dtype=torch.bool)
            infos["_final_info"] = done_mask
            infos["_final_observation"] = done_mask
            infos["_elapsed_steps"] = done_mask

        return (
            [obs_dict],
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(terminations, dtype=torch.bool),
            torch.as_tensor(truncations, dtype=torch.bool),
            [infos],
        )
