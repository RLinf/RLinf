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

import copy
import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.sim.mujoco_warp.tasks.locomotion import TASK_SPECS, LocomotionTask
from rlinf.envs.sim.mujoco_warp.world import MjWarpWorld

__all__ = ["MujocoWarpEnv"]

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


class MujocoWarpEnv(gym.Env):
    """Batched MuJoCo Warp locomotion env compatible with the EnvWorker chunk API.

    Physics lives on ``MjWarpWorld``. Reward, termination, and the initial
    state come from ``LocomotionTask``. Observations contain ``states`` and,
    when video is enabled, an ``images`` batch.

    Args:
        cfg: Task, simulation, reset, reward, and video configuration.
        num_envs: Number of parallel simulation worlds.
        seed_offset: Offset added to the configured random seed.
        total_num_processes: Total number of environment worker processes.
        worker_info: Worker metadata supplied by the scheduler.
        record_metrics: Whether to accumulate episode returns and lengths.
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
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")

        self.task_name = str(_cfg_get(cfg, "task_name", "half_cheetah"))
        if self.task_name not in TASK_SPECS:
            raise ValueError(
                f"Unknown mujoco_warp task {self.task_name!r}. "
                f"Available tasks: {list(TASK_SPECS.keys())}"
            )
        spec = TASK_SPECS[self.task_name]

        env_seed = int(_cfg_get(cfg, "seed", 0))
        self.seed = env_seed + int(seed_offset)
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.reward_scale = float(_cfg_get(cfg, "reward_scale", 1.0))
        self.is_eval = bool(_cfg_get(cfg, "is_eval", False))
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self._save_video = bool(_cfg_get(self.video_cfg, "save_video", False))
        self._render_width = int(_cfg_get(cfg, "render_width", 480))
        self._render_height = int(_cfg_get(cfg, "render_height", 480))
        if self._render_width < 1 or self._render_height < 1:
            raise ValueError(
                "render_width and render_height must be positive, got "
                f"{self._render_width}x{self._render_height}."
            )
        self._renderer = None
        self._render_data = None
        self._render_camera = None
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, self.num_envs // max(1, self.group_size))
        self.use_fixed_reset_state_ids = bool(
            _cfg_get(cfg, "use_fixed_reset_state_ids", False)
        )
        self.use_ordered_reset_state_ids = bool(
            _cfg_get(cfg, "use_ordered_reset_state_ids", False)
        )
        self.specific_reset_id = _cfg_get(cfg, "specific_reset_id", None)
        max_episode_steps = _cfg_get(cfg, "max_episode_steps", None)
        if max_episode_steps is None:
            max_episode_steps = _cfg_get(cfg, "max_steps_per_rollout_epoch", 1000)
        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}."
            )
        self.record_metrics = bool(record_metrics)

        self.device = torch.device(str(_cfg_get(cfg, "device", "cpu")))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"MujocoWarpEnv supports only CPU and CUDA devices, got {self.device}."
            )

        xml_path = os.path.join(_ASSETS_DIR, spec.xml_file)
        self.world = MjWarpWorld.from_xml(xml_path, self.num_envs, self.device)
        self.task = LocomotionTask(spec, seed=self.seed)
        self.obs_dim = self.task.obs_dim(self.world)
        self.action_dim = self.world.nu
        self.frame_dt = self.world.timestep * self.task.frame_skip()
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": int(round(1.0 / self.frame_dt)),
        }
        ctrl_low = self.world.ctrl_low.detach().cpu().numpy()
        ctrl_high = self.world.ctrl_high.detach().cpu().numpy()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=ctrl_low, high=ctrl_high, dtype=np.float32
        )

        self._np_generator = np.random.default_rng(seed=self.seed)
        self.start_idx = 0
        self._is_start = True
        self._last_obs: torch.Tensor | None = None
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self.prev_step_reward = torch.zeros(self.num_envs, device=self.device)
        self._reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_length = torch.zeros(self.num_envs, device=self.device)
        self._init_reset_state_ids()

    @property
    def is_start(self) -> bool:
        """Whether the environment is in its initial state."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return the current episode step count for every world."""
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        """Return additional info keys exposed for metric logging."""
        return []

    def close(self) -> None:
        """Release the native MuJoCo renderer and its associated state."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._render_data = None
        self._render_camera = None

    # ------------------------------------------------------------------
    # Reset-state bookkeeping (same contract as D4RLEnv)
    # ------------------------------------------------------------------
    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self.specific_reset_id is not None:
            return np.full(
                (num_reset_states,), int(self.specific_reset_id), dtype=np.int64
            )
        return self._np_generator.integers(
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
        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
        """Sample new deterministic reset IDs for all environment groups."""
        if self.is_eval or self.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = np.repeat(reset_state_ids, self.group_size)[
            : self.num_envs
        ]

    def _sample_reset_state_ids(self, env_idx: np.ndarray) -> np.ndarray:
        if self.use_fixed_reset_state_ids:
            return self.reset_state_ids[env_idx]
        if self.is_eval or self.use_ordered_reset_state_ids:
            return self._get_ordered_reset_state_ids(len(env_idx))
        return self._get_random_reset_state_ids(len(env_idx))

    def _wrap_obs(self, states: torch.Tensor) -> dict[str, Any]:
        obs = {"states": states}
        if self._save_video:
            obs["images"] = self.render()
        return obs

    def _init_renderer(self) -> None:
        import mujoco

        mj_model = self.world.mj_model
        mj_model.vis.global_.offwidth = max(
            mj_model.vis.global_.offwidth, self._render_width
        )
        mj_model.vis.global_.offheight = max(
            mj_model.vis.global_.offheight, self._render_height
        )
        self._render_data = mujoco.MjData(mj_model)
        self._renderer = mujoco.Renderer(
            mj_model, height=self._render_height, width=self._render_width
        )
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._render_camera.trackbodyid = 1
        self._render_camera.distance = 3.0
        self._render_camera.azimuth = 90.0
        self._render_camera.elevation = -20.0

    def render(self) -> np.ndarray:
        """Render all worlds as an RGB batch using native MuJoCo.

        Returns:
            A ``uint8`` array with shape
            ``(num_envs, render_height, render_width, 3)``.
        """
        import mujoco

        if self._renderer is None:
            self._init_renderer()
        qpos = self.world.qpos.detach().cpu().numpy()
        qvel = self.world.qvel.detach().cpu().numpy()
        ctrl = self.world.ctrl.detach().cpu().numpy()
        frames = []
        for world_idx in range(self.num_envs):
            self._render_data.qpos[:] = qpos[world_idx]
            self._render_data.qvel[:] = qvel[world_idx]
            self._render_data.ctrl[:] = ctrl[world_idx]
            mujoco.mj_forward(self.world.mj_model, self._render_data)
            self._renderer.update_scene(self._render_data, camera=self._render_camera)
            frames.append(self._renderer.render())
        return np.stack(frames).astype(np.uint8, copy=False)

    # ------------------------------------------------------------------
    # EnvWorker contract
    # ------------------------------------------------------------------
    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset selected worlds and return observations for the full batch.

        Args:
            env_idx: World indices to reset. ``None`` resets all worlds.
            reset_state_ids: Optional deterministic noise ID for each selected
                world. New IDs are sampled when this is ``None``.

        Returns:
            The full-batch observation dictionary and an empty info dictionary.
        """
        had_last_obs = self._last_obs is not None
        if env_idx is None:
            env_idx_arr = np.arange(self.num_envs, dtype=np.int64)
        else:
            env_idx_arr = np.asarray(env_idx, dtype=np.int64)
            if env_idx_arr.ndim == 0:
                env_idx_arr = env_idx_arr.reshape(1)
        if env_idx_arr.ndim != 1:
            raise ValueError(
                f"env_idx must be a scalar or 1D sequence, got {env_idx_arr.shape}."
            )
        if np.any((env_idx_arr < 0) | (env_idx_arr >= self.num_envs)):
            raise IndexError(
                f"env_idx values must be in [0, {self.num_envs}), got {env_idx_arr}."
            )
        if reset_state_ids is None:
            reset_state_ids = self._sample_reset_state_ids(env_idx_arr)
        else:
            reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)
        if reset_state_ids.shape != (len(env_idx_arr),):
            raise ValueError(
                "reset_state_ids must have one ID per selected world; "
                f"expected {(len(env_idx_arr),)}, got {reset_state_ids.shape}."
            )

        if self._last_obs is None and len(env_idx_arr) < self.num_envs:
            full_env_idx = np.arange(self.num_envs, dtype=np.int64)
            full_reset_state_ids = self._sample_reset_state_ids(full_env_idx)
            full_reset_state_ids[env_idx_arr] = reset_state_ids
            env_idx_arr = full_env_idx
            reset_state_ids = full_reset_state_ids

        world_ids = torch.as_tensor(env_idx_arr, dtype=torch.long, device=self.device)
        partial_obs = self.task.reset(self.world, world_ids, reset_state_ids)
        if len(env_idx_arr) == self.num_envs:
            full_obs = self.task.observe(self.world).clone()
        else:
            full_obs = self._last_obs.clone()
            full_obs[env_idx_arr] = partial_obs
        self._last_obs = full_obs
        self._is_start = False

        reset_all_stats = (not had_last_obs) or (len(env_idx_arr) == self.num_envs)
        stat_idx = (
            torch.arange(self.num_envs, device=self.device)
            if reset_all_stats
            else torch.as_tensor(env_idx_arr, device=self.device)
        )
        self._elapsed_steps[stat_idx] = 0
        self.prev_step_reward[stat_idx] = 0.0
        self._reward_sum[stat_idx] = 0.0
        self._episode_length[stat_idx] = 0.0

        return self._wrap_obs(full_obs), {}

    def step(
        self, actions: torch.Tensor | np.ndarray, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Advance every world by one control step.

        Args:
            actions: Control batch with shape ``(num_envs, action_dim)``.
            auto_reset: Whether finished worlds may reset immediately.

        Returns:
            Observations, scaled rewards, terminations, truncations, and infos.
            Terminal observations and infos are preserved when auto-reset occurs.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        obs, rewards, terminations = self.task.step(self.world, actions)

        self._elapsed_steps += 1
        truncations = self._elapsed_steps >= self.max_episode_steps

        rewards = rewards * self.reward_scale
        step_rewards = rewards.clone()
        if self.use_rel_reward:
            step_rewards = step_rewards - self.prev_step_reward
            self.prev_step_reward = rewards.clone()

        if self.record_metrics:
            self._reward_sum += step_rewards
            self._episode_length += 1

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
            infos["episode"]["terminated_at_end"] = terminations.clone()
            terminations = torch.zeros_like(terminations)

        obs_dict = self._wrap_obs(obs.clone())
        all_dones = torch.logical_or(terminations, truncations)
        if all_dones.any() and auto_reset and self.auto_reset:
            obs_dict, infos = self._handle_auto_reset(all_dones, obs_dict, infos)

        self._last_obs = obs_dict["states"].clone()
        self._is_start = False
        return (
            obs_dict,
            step_rewards,
            terminations,
            truncations,
            infos,
        )

    @staticmethod
    def _masked_update(dst: Any, src: Any, mask: torch.Tensor) -> Any:
        if isinstance(src, dict):
            merged = {} if dst is None else copy.deepcopy(dst)
            for key, value in src.items():
                merged[key] = MujocoWarpEnv._masked_update(merged.get(key), value, mask)
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
        """Step a chunk of actions and return EnvWorker-compatible batch outputs.

        Args:
            chunk_actions: Actions with shape
                ``(num_envs, chunk_steps, action_dim)``. A single-step batch with
                shape ``(num_envs, action_dim)`` is also accepted.

        Returns:
            Per-step observations, rewards, terminations, truncations, and infos.
            Done flags are aggregated onto the last sub-step during auto-reset.
        """
        acts = (
            chunk_actions.detach()
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(np.asarray(chunk_actions))
        )
        acts = acts.to(self.device).float()
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

        if self._last_obs is None:
            self.reset()
        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []
        finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        final_obs = None
        final_info = None
        final_qpos = self.world.qpos.clone()
        final_qvel = self.world.qvel.clone()
        final_ctrl = self.world.ctrl.clone()
        final_terminations = torch.zeros_like(finished)
        final_truncations = torch.zeros_like(finished)

        for i in range(acts.shape[1]):
            previously_finished = finished
            (
                extracted_obs,
                step_reward,
                terminations,
                truncations,
                infos,
            ) = self.step(acts[:, i], auto_reset=False)

            if previously_finished.any():
                # MjWarp steps the whole batch, so restore finished worlds to
                # their first terminal state before caching this substep.
                finished_ids = previously_finished.nonzero(as_tuple=True)[0]
                self.world.reset_worlds(finished_ids)
                self.world.qpos[finished_ids] = final_qpos[finished_ids]
                self.world.qvel[finished_ids] = final_qvel[finished_ids]
                self.world.ctrl[finished_ids] = final_ctrl[finished_ids]
                self.world.forward()
                self._elapsed_steps[previously_finished] = 0
                self.prev_step_reward[previously_finished] = 0.0
                self._reward_sum[previously_finished] = 0.0
                self._episode_length[previously_finished] = 0.0
                step_reward = step_reward.masked_fill(previously_finished, 0.0)

            step_dones = torch.logical_or(terminations, truncations)
            new_dones = torch.logical_and(step_dones, ~previously_finished)
            if new_dones.any():
                final_obs = self._masked_update(final_obs, extracted_obs, new_dones)
                final_info = self._masked_update(final_info, infos, new_dones)
                final_qpos[new_dones] = self.world.qpos[new_dones]
                final_qvel[new_dones] = self.world.qvel[new_dones]
                final_ctrl[new_dones] = self.world.ctrl[new_dones]
                final_terminations[new_dones] = terminations[new_dones]
                final_truncations[new_dones] = truncations[new_dones]
                finished = torch.logical_or(finished, new_dones)

            if finished.any():
                extracted_obs = self._masked_update(extracted_obs, final_obs, finished)
                infos = self._masked_update(infos, final_info, finished)
                terminations = torch.where(finished, final_terminations, terminations)
                truncations = torch.where(finished, final_truncations, truncations)
                self._last_obs[finished] = final_obs["states"][finished]

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
            final_obs = self._masked_update(obs_list[-1], final_obs, past_dones)
            final_info = self._masked_update(infos_list[-1], final_info, past_dones)
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, final_obs, final_info
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
        dones: torch.Tensor,
        _final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = copy.deepcopy(_final_obs)
        final_info = copy.deepcopy(infos)
        env_idx = torch.arange(self.num_envs, device=self.device)[dones].cpu().numpy()
        reset_state_ids = self._sample_reset_state_ids(env_idx)
        obs, infos = self.reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        dones_cpu = dones.cpu()
        infos["_final_info"] = dones_cpu
        infos["_final_observation"] = dones_cpu
        infos["_elapsed_steps"] = dones_cpu
        return obs, infos
