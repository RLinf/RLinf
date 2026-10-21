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

import copy
import time
from typing import Any, Optional, Union

import d4rl  # noqa: F401  # registers D4RL envs with gym
import gym
import numpy as np
import torch

from rlinf.envs.venv.venv import DummyVectorEnv, SubprocVectorEnv

__all__ = ["D4RLEnv"]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _to_info_list(infos: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(infos, list):
        return [dict(info) if isinstance(info, dict) else {} for info in infos]
    if isinstance(infos, tuple):
        return [dict(info) if isinstance(info, dict) else {} for info in infos]
    if isinstance(infos, np.ndarray):
        return [dict(info) if isinstance(info, dict) else {} for info in infos.tolist()]
    if isinstance(infos, dict):
        info_list: list[dict[str, Any]] = [{} for _ in range(batch_size)]
        for key, value in infos.items():
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and len(value) == batch_size
            ):
                for i in range(batch_size):
                    info_list[i][key] = value[i]
            else:
                for i in range(batch_size):
                    info_list[i][key] = value
        return info_list
    return [{} for _ in range(batch_size)]


class D4RLEnv(gym.Env):
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
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.seed_offset = int(seed_offset)
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.task_name = _cfg_get(cfg, "task_name", None)
        if self.task_name is None:
            raise ValueError("D4RLEnv requires cfg.task_name.")
        env_seed = _cfg_get(cfg, "seed", None)
        if env_seed is None:
            raise ValueError("D4RLEnv requires cfg.seed (int).")
        self.seed = int(env_seed) + int(seed_offset)
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self.max_episode_steps = _cfg_get(cfg, "max_steps_per_rollout_epoch", None)
        self.max_episode_steps = (
            int(self.max_episode_steps) if self.max_episode_steps is not None else None
        )
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, self.num_envs // max(1, self.group_size))
        self.use_fixed_reset_state_ids = bool(
            _cfg_get(cfg, "use_fixed_reset_state_ids", False)
        )
        self.use_ordered_reset_state_ids = bool(
            _cfg_get(cfg, "use_ordered_reset_state_ids", False)
        )
        self.is_eval = bool(_cfg_get(cfg, "is_eval", False))
        self.specific_reset_id = _cfg_get(cfg, "specific_reset_id", None)
        self.use_subproc_vector_env = bool(
            _cfg_get(cfg, "use_subproc_vector_env", self.num_envs > 1)
        )

        self.record_metrics = bool(record_metrics)
        _save_video = False
        if self.video_cfg is not None:
            if hasattr(self.video_cfg, "save_video"):
                _save_video = bool(self.video_cfg.save_video)
            elif hasattr(self.video_cfg, "get") and self.video_cfg.get("save_video"):
                _save_video = True
        _render_mode = "rgb_array" if _save_video else None
        self._save_video = _save_video
        _task_name = self.task_name

        def _make_env() -> gym.Env:
            if _render_mode is not None:
                try:
                    return gym.make(_task_name, render_mode=_render_mode)
                except TypeError:
                    # Older gym may not accept render_mode in make().
                    return gym.make(_task_name)
            return gym.make(_task_name)

        env_fns = [_make_env for _ in range(self.num_envs)]
        if self.use_subproc_vector_env:
            self.env = SubprocVectorEnv(env_fns)
        else:
            self.env = DummyVectorEnv(env_fns)
        self._seed_list = [self.seed + i for i in range(self.num_envs)]
        self.env.seed(self._seed_list)
        self._generator = np.random.default_rng(seed=self.seed)
        self.start_idx = 0
        self._score_env = self._build_score_env(self.task_name)

        self._is_start = True
        self._last_obs: np.ndarray | None = None
        # Per-env episode stats (used when record_metrics=True).
        self._reward_sum = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_length = np.zeros((self.num_envs,), dtype=np.int64)
        self._start_time = np.array([time.time()] * self.num_envs, dtype=np.float64)
        self._total_timesteps = 0
        self._elapsed_steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.prev_step_reward = np.zeros((self.num_envs,), dtype=np.float32)
        self._init_reset_state_ids()

    @staticmethod
    def _build_score_env(task_name: str) -> gym.Env | None:
        """Build a lightweight env handle for D4RL score normalization."""
        try:
            score_env = gym.make(task_name)
        except Exception:
            return None
        if not hasattr(score_env, "get_normalized_score"):
            try:
                score_env.close()
            except Exception:
                pass
            return None
        return score_env

    def _compute_normalized_scores(self, returns: np.ndarray) -> np.ndarray | None:
        """Convert episode returns to D4RL normalized scores (0-100 scale)."""
        if self._score_env is None:
            return None
        get_score = getattr(self._score_env, "get_normalized_score", None)
        if not callable(get_score):
            return None
        try:
            normalized = np.asarray(
                [float(get_score(float(ret))) * 100.0 for ret in returns],
                dtype=np.float32,
            )
        except Exception:
            return None
        return normalized

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    def close(self) -> None:
        if self._score_env is not None:
            try:
                self._score_env.close()
            except Exception:
                pass
        self.env.close()

    def render(self, **kwargs: Any) -> Any:
        """Render all envs; used by RecordVideo when obs has no image keys."""
        if hasattr(self.env, "render"):
            if not kwargs:
                kwargs = {"mode": "rgb_array"}
            return self.env.render(**kwargs)
        return None

    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self.specific_reset_id is not None:
            return np.full(
                (num_reset_states,), int(self.specific_reset_id), dtype=np.int64
            )
        return self._generator.integers(
            low=0,
            high=np.iinfo(np.int32).max,
            size=(num_reset_states,),
            dtype=np.int64,
        )

    def _get_ordered_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self.specific_reset_id is not None:
            return np.full(
                (num_reset_states,), int(self.specific_reset_id), dtype=np.int64
            )
        reset_state_ids = np.arange(
            self.start_idx, self.start_idx + num_reset_states, dtype=np.int64
        )
        self.start_idx += num_reset_states
        return reset_state_ids

    def _init_reset_state_ids(self) -> None:
        self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
        if self.is_eval or self.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = np.repeat(reset_state_ids, self.group_size)[
            : self.num_envs
        ]

    @staticmethod
    def _wrap_obs(obs: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert raw state observation to EnvWorker-compatible observation dict."""
        return {"states": torch.as_tensor(obs, dtype=torch.float32)}

    def _inject_render_if_needed(self, obs_dict: dict[str, Any]) -> None:
        """When save_video is True, add rendered frames to obs so RecordVideo can use them."""
        if not self._save_video or not hasattr(self.env, "render"):
            return
        try:
            result = self.env.render(mode="rgb_array")
            if result is None:
                return
            if isinstance(result, np.ndarray):
                if result.ndim == 3:
                    result = np.expand_dims(result, axis=0)
                frames = np.asarray(result, dtype=np.uint8)
            elif isinstance(result, (list, tuple)):
                frames = np.stack(
                    [np.asarray(r, dtype=np.uint8) for r in result if r is not None]
                )
            else:
                return
            if frames.size > 0:
                obs_dict["images"] = torch.as_tensor(frames)
        except Exception:
            pass

    def _vector_reset(
        self,
        env_idx: np.ndarray | None = None,
        reset_state_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx, dtype=np.int64)

        if reset_state_ids is None:
            reset_out = self.env.reset(id=env_idx)
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            return np.asarray(obs, dtype=np.float32)

        if len(reset_state_ids) != len(env_idx):
            raise ValueError("reset_state_ids and env_idx length mismatch.")

        obs_chunks: list[np.ndarray] = []
        for idx, state_id in zip(env_idx, reset_state_ids):
            try:
                reset_out = self.env.reset(
                    id=[int(idx)], seed=int(self.seed + int(state_id))
                )
            except TypeError as exc:
                # Old Gym Mujoco reset does not accept `seed` kwarg.
                if "unexpected keyword argument 'seed'" not in str(exc):
                    raise
                reset_out = self.env.reset(id=[int(idx)])
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            obs_chunks.append(np.asarray(obs, dtype=np.float32))
        return np.concatenate(obs_chunks, axis=0)

    def _vector_step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        step_out = self.env.step(actions)
        if len(step_out) == 5:
            obs, rewards, terminations, truncations, infos = step_out
            terminations = np.asarray(terminations, dtype=bool)
            truncations = np.asarray(truncations, dtype=bool)
        elif len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            dones = np.asarray(dones, dtype=bool)
            info_list = _to_info_list(infos, self.num_envs)
            truncations = np.asarray(
                [bool(info.get("TimeLimit.truncated", False)) for info in info_list],
                dtype=bool,
            )
            terminations = np.logical_and(dones, np.logical_not(truncations))
            infos = info_list
        else:
            raise RuntimeError(f"Unexpected step output length: {len(step_out)}")

        info_list = _to_info_list(infos, self.num_envs)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            terminations,
            truncations,
            info_list,
        )

    def _sample_reset_state_ids(self, env_idx: np.ndarray) -> np.ndarray:
        if self.use_fixed_reset_state_ids:
            return self.reset_state_ids[env_idx]
        if self.is_eval or self.use_ordered_reset_state_ids:
            return self._get_ordered_reset_state_ids(len(env_idx))
        return self._get_random_reset_state_ids(len(env_idx))

    def _build_full_obs_after_partial_reset(
        self, env_idx: np.ndarray, partial_obs: np.ndarray
    ) -> np.ndarray:
        """Merge partial reset observations into a full-size observation batch."""
        is_full_reset = len(env_idx) == self.num_envs and np.array_equal(
            env_idx, np.arange(self.num_envs, dtype=np.int64)
        )
        if is_full_reset:
            return partial_obs

        if self._last_obs is None:
            # Keep reset contract consistent: always return full batch.
            full_env_idx = np.arange(self.num_envs, dtype=np.int64)
            full_reset_state_ids = self._sample_reset_state_ids(full_env_idx)
            full_obs = self._vector_reset(
                env_idx=full_env_idx, reset_state_ids=full_reset_state_ids
            )
        else:
            full_obs = self._last_obs.copy()
        full_obs[env_idx] = partial_obs
        return full_obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset env instances and return full-batch observations."""
        had_last_obs = self._last_obs is not None
        if env_idx is None:
            env_idx_arr = np.arange(self.num_envs, dtype=np.int64)
        else:
            env_idx_arr = np.asarray(env_idx, dtype=np.int64)
            if env_idx_arr.ndim == 0:
                env_idx_arr = env_idx_arr.reshape(1)
        if reset_state_ids is None:
            reset_state_ids = self._sample_reset_state_ids(env_idx_arr)
        partial_obs = self._vector_reset(
            env_idx=env_idx_arr, reset_state_ids=reset_state_ids
        )
        full_obs = self._build_full_obs_after_partial_reset(env_idx_arr, partial_obs)
        self._last_obs = full_obs
        self._is_start = False

        reset_all_stats = (not had_last_obs) or (len(env_idx_arr) == self.num_envs)
        stat_env_idx = (
            np.arange(self.num_envs, dtype=np.int64) if reset_all_stats else env_idx_arr
        )
        if self.record_metrics:
            self._reward_sum[stat_env_idx] = 0.0
            self._episode_length[stat_env_idx] = 0
            self._start_time[stat_env_idx] = time.time()
        self._elapsed_steps[stat_env_idx] = 0
        self.prev_step_reward[stat_env_idx] = 0.0
        obs_dict = self._wrap_obs(full_obs)
        self._inject_render_if_needed(obs_dict)
        return obs_dict, {}

    def step(
        self, actions: torch.Tensor | np.ndarray, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        acts_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else actions
        )
        obs, rewards, terminations, truncations, _ = self._vector_step(
            np.asarray(acts_np)
        )
        self._elapsed_steps += 1
        if self.max_episode_steps is not None and self.max_episode_steps > 0:
            truncations = np.logical_or(
                truncations, self._elapsed_steps >= int(self.max_episode_steps)
            )

        step_rewards = rewards.copy()
        if self.use_rel_reward:
            step_rewards = step_rewards - self.prev_step_reward
            self.prev_step_reward = rewards.copy()

        if self.record_metrics:
            self._reward_sum += step_rewards
            self._episode_length += 1
            self._total_timesteps += int(self.num_envs)

        ep_returns = np.zeros((self.num_envs,), dtype=np.float32)
        ep_lengths = np.zeros((self.num_envs,), dtype=np.float32)
        ep_normalized_scores = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.logical_or(terminations, truncations)
        if self.record_metrics and np.any(dones):
            ep_returns[dones] = self._reward_sum[dones]
            ep_lengths[dones] = self._episode_length[dones].astype(np.float32)
            done_returns = ep_returns[dones]
            normalized_scores = self._compute_normalized_scores(done_returns)
            if normalized_scores is not None:
                ep_normalized_scores[dones] = normalized_scores
            self._reward_sum[dones] = 0.0
            self._episode_length[dones] = 0
            self._start_time[dones] = time.time()
        if np.any(dones):
            self._elapsed_steps[dones] = 0
            self.prev_step_reward[dones] = 0.0

        infos: dict[str, Any] = {
            "episode": {
                "return": torch.as_tensor(ep_normalized_scores, dtype=torch.float32),
                "episode_len": torch.as_tensor(ep_lengths, dtype=torch.float32),
            }
        }
        if self.ignore_terminations:
            infos["episode"]["terminated_at_end"] = torch.as_tensor(
                terminations.copy(), dtype=torch.bool
            )
            terminations = np.zeros_like(terminations, dtype=bool)

        if (
            np.any(np.logical_or(terminations, truncations))
            and auto_reset
            and self.auto_reset
        ):
            obs_dict = self._wrap_obs(obs)
            obs_dict, infos = self._handle_auto_reset(
                np.logical_or(terminations, truncations), obs_dict, infos
            )
        else:
            obs_dict = self._wrap_obs(obs)
        self._inject_render_if_needed(obs_dict)

        self._last_obs = obs_dict["states"].cpu().numpy()
        self._is_start = False
        return (
            obs_dict,
            torch.as_tensor(step_rewards, dtype=torch.float32),
            torch.as_tensor(terminations, dtype=torch.bool),
            torch.as_tensor(truncations, dtype=torch.bool),
            infos,
        )

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

        if self._last_obs is None:
            self.reset()
        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []
        for i in range(k):
            (
                extracted_obs,
                step_reward,
                terminations,
                truncations,
                infos,
            ) = self.step(acts[:, i], auto_reset=False)
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_tensor = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations_tensor = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations_tensor = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations_tensor.any(dim=1)
        past_truncations = raw_chunk_truncations_tensor.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations_tensor)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations_tensor)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations_tensor.clone()
            chunk_truncations = raw_chunk_truncations_tensor.clone()

        return (
            obs_list,
            chunk_rewards_tensor,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(
        self,
        dones: np.ndarray,
        _final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = copy.deepcopy(_final_obs)
        final_info = copy.deepcopy(infos)
        env_idx = np.arange(self.num_envs, dtype=np.int64)[dones]
        if self.is_eval:
            self.update_reset_state_ids()
        reset_state_ids = self._sample_reset_state_ids(env_idx)
        obs, infos = self.reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos
