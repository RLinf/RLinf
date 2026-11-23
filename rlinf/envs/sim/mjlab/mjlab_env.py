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

"""Batched mjlab environment adapter compatible with RLinf's EnvWorker."""

from __future__ import annotations

import copy
import importlib
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlinf.scheduler import Worker

__all__ = ["MjlabEnv"]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


class MjlabEnv(gym.Env):
    """Wrap any registered ``mjlab`` ``ManagerBasedRlEnv`` for RLinf.

    The underlying ``ManagerBasedRlEnv`` runs with ``auto_reset=False`` so this
    adapter captures pre-reset terminal observations for timeout bootstrapping
    and resets completed worlds explicitly.

    Args:
        cfg: Environment configuration containing ``task_id`` and runtime flags.
        num_envs: Number of parallel simulation worlds hosted by this instance.
        seed_offset: Offset added to ``cfg.seed`` for per-worker seeding.
        total_num_processes: Total environment worker processes across the run.
        worker_info: Scheduler metadata for the owning worker.
        record_metrics: Whether to track per-episode returns, lengths, and logs.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.record_metrics = bool(record_metrics)

        task_id = _cfg_get(cfg, "task_id", None)
        if not task_id:
            raise ValueError("mjlab env requires a non-empty cfg.task_id.")
        self.task_id = str(task_id)
        self.task_modules = tuple(_cfg_get(cfg, "task_modules", ()) or ())
        self.actor_obs_key = str(_cfg_get(cfg, "actor_obs_key", "actor"))
        critic_obs_key = _cfg_get(cfg, "critic_obs_key", "critic")
        self.critic_obs_key = str(critic_obs_key) if critic_obs_key else None

        self.seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_offset)
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.reward_scale = float(_cfg_get(cfg, "reward_scale", 1.0))
        max_episode_steps = _cfg_get(cfg, "max_episode_steps", None)
        self.max_episode_steps = (
            int(max_episode_steps) if max_episode_steps is not None else None
        )
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}."
            )
        clip_actions = _cfg_get(cfg, "clip_actions", None)
        self.clip_actions = float(clip_actions) if clip_actions is not None else None
        self.is_eval = bool(_cfg_get(cfg, "is_eval", False))
        self.play_mode = bool(_cfg_get(cfg, "play_mode", False))
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self._save_video = bool(_cfg_get(self.video_cfg, "save_video", False))
        self._render_width = _cfg_get(cfg, "render_width", None)
        self._render_height = _cfg_get(cfg, "render_height", None)

        requested_device = str(_cfg_get(cfg, "device", "auto"))
        if requested_device == "auto":
            if (
                Worker.torch_platform is not None
                and Worker.torch_platform.is_available()
            ):
                device_str = f"{Worker.torch_device_type}:0"
            else:
                device_str = "cpu"
        else:
            device_str = requested_device
        self.device = torch.device(device_str)

        self.env = self._build_env()
        self.device = torch.device(str(self.env.device))
        self.action_dim = int(self.env.action_manager.total_action_dim)
        self.step_dt = float(self.env.step_dt)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": max(1, int(round(1.0 / self.step_dt))),
        }

        initial_step_counter = int(_cfg_get(cfg, "initial_step_counter", 0))
        if initial_step_counter > 0:
            self.env.common_step_counter = initial_step_counter

        self._is_start = True
        self._has_reset = False
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self.prev_step_reward = torch.zeros(self.num_envs, device=self.device)
        self._reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_length = torch.zeros(self.num_envs, device=self.device)

    def _build_env(self) -> Any:
        import mjlab.tasks  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        for module_name in self.task_modules:
            importlib.import_module(module_name)

        env_cfg = load_env_cfg(self.task_id, play=self.play_mode)
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.auto_reset = False
        if self.ignore_terminations and isinstance(
            getattr(env_cfg, "terminations", None), dict
        ):
            env_cfg.terminations = {
                name: term
                for name, term in env_cfg.terminations.items()
                if getattr(term, "time_out", False)
            }

        if self._save_video:
            if self._render_width is not None:
                env_cfg.viewer.width = int(self._render_width)
            if self._render_height is not None:
                env_cfg.viewer.height = int(self._render_height)
            render_mode = "rgb_array"
        else:
            render_mode = None

        env = ManagerBasedRlEnv(
            cfg=env_cfg,
            device=str(self.device),
            render_mode=render_mode,
        )
        env.seed(self.seed)
        return env

    @property
    def is_start(self) -> bool:
        """Whether the environment is waiting for its initial episode start."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Current episode step count per world."""
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        """Additional info keys exposed for metric logging."""
        return []

    def update_reset_state_ids(self) -> None:
        """No-op hook matching the EnvWorker reset-state interface."""

    def close(self) -> None:
        """Release underlying simulator and renderer resources."""
        self.env.close()

    def _wrap_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        if self.actor_obs_key not in obs_dict:
            raise KeyError(
                f"Expected actor observation group {self.actor_obs_key!r}, "
                f"found groups {sorted(obs_dict.keys())}."
            )
        wrapped: dict[str, Any] = {"states": obs_dict[self.actor_obs_key].clone()}
        if self.critic_obs_key is not None and self.critic_obs_key in obs_dict:
            wrapped["critic_states"] = obs_dict[self.critic_obs_key].clone()
        if self._save_video:
            frame = self.env.render()
            if frame is not None:
                wrapped["images"] = frame
        return wrapped

    def _attach_mjlab_logs(
        self,
        episode_info: dict[str, torch.Tensor],
        extras: dict[str, Any],
        mask: torch.Tensor,
    ) -> None:
        log = extras.get("log") if isinstance(extras, dict) else None
        if not isinstance(log, dict) or not mask.any():
            return
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = float(value.detach().item())
            elif isinstance(value, (int, float, np.number)):
                scalar = float(value)
            else:
                continue
            if not np.isfinite(scalar):
                continue
            buf = episode_info.get(key)
            if buf is None:
                buf = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
                episode_info[key] = buf
            buf[mask] = scalar

    def reset(
        self,
        env_idx: int | list[int] | np.ndarray | torch.Tensor | None = None,
        reset_state_ids: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset selected worlds (or all worlds when ``env_idx`` is ``None``)."""
        del reset_state_ids
        if env_idx is None:
            env_ids = None
            stat_idx = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
            if env_ids.ndim == 0:
                env_ids = env_ids.reshape(1)
            stat_idx = env_ids

        obs_dict, extras = self.env.reset(env_ids=env_ids)
        self._elapsed_steps[stat_idx] = 0
        self.prev_step_reward[stat_idx] = 0.0
        self._reward_sum[stat_idx] = 0.0
        self._episode_length[stat_idx] = 0.0
        self._is_start = False
        self._has_reset = True
        return self._wrap_obs(obs_dict), {"mjlab_extras": extras}

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Advance every world by one control step."""
        act_tensor = (
            actions.detach()
            if isinstance(actions, torch.Tensor)
            else torch.as_tensor(np.asarray(actions))
        )
        act_tensor = act_tensor.to(device=self.device, dtype=torch.float32)
        if self.clip_actions is not None:
            act_tensor = act_tensor.clamp(-self.clip_actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, _ = self.env.step(act_tensor)

        rewards = torch.nan_to_num(
            rewards.to(device=self.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rewards = rewards * self.reward_scale
        step_rewards = rewards.clone()
        if self.use_rel_reward:
            step_rewards = step_rewards - self.prev_step_reward
            self.prev_step_reward = rewards.clone()

        self._elapsed_steps += 1
        if self.record_metrics:
            self._reward_sum += step_rewards
            self._episode_length += 1

        terminations = terminated.to(device=self.device, dtype=torch.bool)
        truncations = truncated.to(device=self.device, dtype=torch.bool)
        if self.max_episode_steps is not None:
            truncations = torch.logical_or(
                truncations, self._elapsed_steps >= self.max_episode_steps
            )
        terminated_at_end = terminations.clone()
        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)

        dones = torch.logical_or(terminations, truncations)
        ep_returns = torch.zeros(self.num_envs, device=self.device)
        ep_lengths = torch.zeros(self.num_envs, device=self.device)
        if self.record_metrics and dones.any():
            ep_returns[dones] = self._reward_sum[dones]
            ep_lengths[dones] = self._episode_length[dones]
            self._reward_sum[dones] = 0.0
            self._episode_length[dones] = 0.0
        if dones.any():
            self._elapsed_steps[dones] = 0
            self.prev_step_reward[dones] = 0.0

        infos: dict[str, Any] = {
            "episode": {
                "return": ep_returns,
                "episode_len": ep_lengths,
            }
        }
        if self.ignore_terminations:
            infos["episode"]["terminated_at_end"] = terminated_at_end

        wrapped_obs = self._wrap_obs(obs_dict)
        if dones.any() and auto_reset and self.auto_reset:
            wrapped_obs, infos = self._handle_auto_reset(dones, wrapped_obs, infos)

        self._is_start = False
        return wrapped_obs, step_rewards, terminations, truncations, infos

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        terminal_obs: dict[str, Any],
        terminal_infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = copy.deepcopy(terminal_obs)
        final_info = copy.deepcopy(terminal_infos)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        obs, reset_infos = self.reset(env_idx=done_ids)
        self._attach_mjlab_logs(
            final_info["episode"], reset_infos.get("mjlab_extras", {}), dones
        )
        dones_cpu = dones.cpu()
        infos: dict[str, Any] = {
            "final_observation": final_obs,
            "final_info": final_info,
            "_final_info": dones_cpu,
            "_final_observation": dones_cpu,
            "_elapsed_steps": dones_cpu,
        }
        return obs, infos

    @staticmethod
    def _masked_update(dst: Any, src: Any, mask: torch.Tensor) -> Any:
        if isinstance(src, dict):
            merged = {} if dst is None else copy.deepcopy(dst)
            for key, value in src.items():
                merged[key] = MjlabEnv._masked_update(merged.get(key), value, mask)
            return merged
        if isinstance(src, torch.Tensor):
            merged = src.clone() if dst is None else dst.clone()
            if src.ndim > 0 and src.shape[0] == mask.shape[0]:
                merged[mask] = src[mask]
            return merged
        if isinstance(src, np.ndarray):
            merged = src.copy() if dst is None else dst.copy()
            if src.ndim > 0 and src.shape[0] == mask.shape[0]:
                np_mask = mask.detach().cpu().numpy()
                merged[np_mask] = src[np_mask]
            return merged
        return copy.deepcopy(src)

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Step an action chunk and return EnvWorker-compatible batch outputs."""
        acts = (
            chunk_actions.detach()
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(np.asarray(chunk_actions))
        )
        acts = acts.to(device=self.device, dtype=torch.float32)
        if acts.ndim == 2:
            acts = acts[:, None, :]
        if (
            acts.ndim != 3
            or acts.shape[0] != self.num_envs
            or acts.shape[1] < 1
            or acts.shape[2] != self.action_dim
        ):
            raise ValueError(
                "chunk_actions must have shape "
                f"[{self.num_envs}, chunk_steps >= 1, {self.action_dim}], "
                f"got {tuple(acts.shape)}."
            )

        if not self._has_reset:
            self.reset()

        chunk_steps = acts.shape[1]
        if chunk_steps == 1:
            obs, rewards, terminations, truncations, infos = self.step(
                acts[:, 0], auto_reset=self.auto_reset
            )
            return (
                [obs],
                rewards.unsqueeze(1),
                terminations.unsqueeze(1),
                truncations.unsqueeze(1),
                [infos],
            )

        if not self.auto_reset:
            raise ValueError(
                "MjlabEnv requires auto_reset=True when chunk_steps > 1, "
                f"got chunk_steps={chunk_steps} with auto_reset=False."
            )

        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []

        finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        final_obs: dict[str, Any] | None = None
        final_info: dict[str, Any] | None = None
        last_step_new_dones = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        for i in range(chunk_steps):
            previously_finished = finished
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                acts[:, i], auto_reset=False
            )
            step_dones = torch.logical_or(terminations, truncations)
            if previously_finished.any():
                step_reward = step_reward.masked_fill(previously_finished, 0.0)
                terminations = terminations & ~previously_finished
                truncations = truncations & ~previously_finished

            new_dones = torch.logical_and(step_dones, ~previously_finished)
            if new_dones.any():
                final_obs = self._masked_update(final_obs, extracted_obs, new_dones)
                final_info = self._masked_update(final_info, infos, new_dones)
                finished = torch.logical_or(finished, new_dones)

            # mjlab requires resetting done worlds before stepping again when
            # env_cfg.auto_reset=False; defer final reset of all finished worlds
            # until the end of the chunk so counters and obs_list[-1] start at 0.
            if i + 1 < chunk_steps:
                if step_dones.any():
                    done_ids = step_dones.nonzero(as_tuple=False).squeeze(-1)
                    extracted_obs, reset_infos = self.reset(env_idx=done_ids)
                    if new_dones.any() and final_info is not None:
                        self._attach_mjlab_logs(
                            final_info["episode"],
                            reset_infos.get("mjlab_extras", {}),
                            new_dones,
                        )
            else:
                last_step_new_dones = new_dones

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

        if past_dones.any() and final_obs is not None and final_info is not None:
            reset_obs = obs_list[-1]
            if last_step_new_dones.any():
                last_done_ids = last_step_new_dones.nonzero(as_tuple=False).squeeze(-1)
                reset_obs, reset_infos = self.reset(env_idx=last_done_ids)
                self._attach_mjlab_logs(
                    final_info["episode"],
                    reset_infos.get("mjlab_extras", {}),
                    last_step_new_dones,
                )
            earlier_dones = past_dones & ~last_step_new_dones
            if earlier_dones.any():
                earlier_done_ids = earlier_dones.nonzero(as_tuple=False).squeeze(-1)
                reset_obs, _ = self.reset(env_idx=earlier_done_ids)
            obs_list[-1] = reset_obs
            dones_cpu = past_dones.cpu()
            infos_list[-1] = {
                "final_observation": self._masked_update(
                    obs_list[-1], final_obs, past_dones
                ),
                "final_info": self._masked_update(
                    infos_list[-1], final_info, past_dones
                ),
                "_final_info": dones_cpu,
                "_final_observation": dones_cpu,
                "_elapsed_steps": dones_cpu,
            }

        chunk_terminations = torch.zeros_like(raw_chunk_terminations_tensor)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations_tensor)
        chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards_tensor,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )
