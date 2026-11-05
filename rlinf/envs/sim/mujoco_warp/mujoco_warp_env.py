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

__all__ = ["MujocoWarpEnv"]

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


class _TaskSpec:
    """Static description of one locomotion task.

    Reward and termination follow the Gymnasium MuJoCo v4 task semantics:
    ``reward = healthy_reward + forward_velocity - ctrl_cost_weight * ||ctrl||^2``,
    and (for terminating tasks) an episode ends when the body leaves the
    healthy height/angle ranges or the observation becomes non-finite.
    """

    def __init__(
        self,
        xml_file: str,
        obs_qpos_skip: int,
        frame_skip: int,
        ctrl_cost_weight: float,
        healthy_reward: float,
        healthy_z_range: Optional[tuple[float, float]],
        healthy_angle_range: Optional[tuple[float, float]],
        reset_noise_scale: float,
        reset_noise_type: str,
    ):
        self.xml_file = xml_file
        self.obs_qpos_skip = obs_qpos_skip
        self.frame_skip = frame_skip
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range
        self.reset_noise_scale = reset_noise_scale
        self.reset_noise_type = reset_noise_type


TASK_SPECS: dict[str, _TaskSpec] = {
    "half_cheetah": _TaskSpec(
        xml_file="half_cheetah.xml",
        obs_qpos_skip=1,
        frame_skip=5,
        ctrl_cost_weight=0.1,
        healthy_reward=0.0,
        healthy_z_range=None,
        healthy_angle_range=None,
        reset_noise_scale=0.1,
        reset_noise_type="uniform",
    ),
    "hopper": _TaskSpec(
        xml_file="hopper.xml",
        obs_qpos_skip=1,
        frame_skip=4,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        reset_noise_type="uniform",
    ),
    "walker2d": _TaskSpec(
        xml_file="walker2d.xml",
        obs_qpos_skip=1,
        frame_skip=4,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        reset_noise_type="uniform",
    ),
    "ant": _TaskSpec(
        xml_file="ant.xml",
        obs_qpos_skip=2,
        frame_skip=5,
        ctrl_cost_weight=5e-4,
        healthy_reward=1.0,
        healthy_z_range=(0.2, 1.0),
        healthy_angle_range=None,
        reset_noise_scale=0.1,
        reset_noise_type="uniform",
    ),
    "humanoid": _TaskSpec(
        xml_file="humanoid.xml",
        obs_qpos_skip=2,
        frame_skip=5,
        ctrl_cost_weight=0.1,
        healthy_reward=1.25,
        healthy_z_range=(1.0, 2.0),
        healthy_angle_range=None,
        reset_noise_scale=1e-2,
        reset_noise_type="uniform",
    ),
}


class MujocoWarpEnv(gym.Env):
    """Batched MuJoCo Warp locomotion env compatible with the EnvWorker chunk API.

    All ``num_envs`` worlds are stepped together on device with one
    ``mujoco_warp.step`` call per frame, so there is no per-env Python loop
    in the physics path. Observations are ``{"states": Tensor[num_envs, obs_dim]}``.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ):
        super().__init__()
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")

        self.task_name = str(_cfg_get(cfg, "task_name", "half_cheetah"))
        assert self.task_name in TASK_SPECS, (
            f"Unknown mujoco_warp task {self.task_name!r}. "
            f"Available tasks: {list(TASK_SPECS.keys())}"
        )
        self.spec = TASK_SPECS[self.task_name]

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
        self._wp_device = str(self.device)

        self._build_simulation()

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

    def _build_simulation(self) -> None:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        wp.init()
        self._wp = wp
        self._mjw = mjw

        xml_path = os.path.join(_ASSETS_DIR, self.spec.xml_file)
        self.mjm = mujoco.MjModel.from_xml_path(xml_path)
        if self.mjm.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            # MJWarp does not implement the PGS solver; CG is the closest
            # supported option.
            self.mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.frame_dt = float(self.mjm.opt.timestep) * self.spec.frame_skip
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": int(round(1.0 / self.frame_dt)),
        }

        with wp.ScopedDevice(self._wp_device):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.make_data(self.mjm, nworld=self.num_envs)

        self._qpos = wp.to_torch(self.d.qpos)
        self._qvel = wp.to_torch(self.d.qvel)
        self._ctrl = wp.to_torch(self.d.ctrl)
        self._init_qpos = torch.as_tensor(
            np.array(self.mjm.qpos0), dtype=torch.float32, device=self.device
        )
        self._init_qvel = torch.zeros(self.mjm.nv, device=self.device)

        self.obs_dim = (self.mjm.nq - self.spec.obs_qpos_skip) + self.mjm.nv
        self.action_dim = int(self.mjm.nu)
        ctrl_range = np.asarray(self.mjm.actuator_ctrlrange, dtype=np.float32)
        self._ctrl_low = torch.as_tensor(ctrl_range[:, 0], device=self.device)
        self._ctrl_high = torch.as_tensor(ctrl_range[:, 1], device=self.device)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0], high=ctrl_range[:, 1], dtype=np.float32
        )

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    def close(self) -> None:
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
        self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
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

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------
    def _obs_from_state(self) -> torch.Tensor:
        skip = self.spec.obs_qpos_skip
        return torch.cat([self._qpos[:, skip:], self._qvel], dim=1)

    def _wrap_obs(self, states: torch.Tensor) -> dict[str, Any]:
        obs = {"states": states}
        if self._save_video:
            obs["images"] = self.render()
        return obs

    def _init_renderer(self) -> None:
        import mujoco

        self.mjm.vis.global_.offwidth = max(
            self.mjm.vis.global_.offwidth, self._render_width
        )
        self.mjm.vis.global_.offheight = max(
            self.mjm.vis.global_.offheight, self._render_height
        )
        self._render_data = mujoco.MjData(self.mjm)
        self._renderer = mujoco.Renderer(
            self.mjm, height=self._render_height, width=self._render_width
        )
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._render_camera.trackbodyid = 1
        self._render_camera.distance = 3.0
        self._render_camera.azimuth = 90.0
        self._render_camera.elevation = -20.0

    def render(self) -> np.ndarray:
        """Render all worlds as an RGB batch using native MuJoCo."""
        import mujoco

        if self._renderer is None:
            self._init_renderer()
        qpos = self._qpos.detach().cpu().numpy()
        qvel = self._qvel.detach().cpu().numpy()
        ctrl = self._ctrl.detach().cpu().numpy()
        frames = []
        for world_idx in range(self.num_envs):
            self._render_data.qpos[:] = qpos[world_idx]
            self._render_data.qvel[:] = qvel[world_idx]
            self._render_data.ctrl[:] = ctrl[world_idx]
            mujoco.mj_forward(self.mjm, self._render_data)
            self._renderer.update_scene(self._render_data, camera=self._render_camera)
            frames.append(self._renderer.render())
        return np.stack(frames).astype(np.uint8, copy=False)

    def _vector_reset(
        self,
        env_idx: np.ndarray,
        reset_state_ids: np.ndarray,
    ) -> torch.Tensor:
        """Reset the given worlds to ``qpos0`` plus task-specific noise."""
        reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)
        if reset_state_ids.shape != (len(env_idx),):
            raise ValueError(
                "reset_state_ids must have one ID per selected world; "
                f"expected {(len(env_idx),)}, got {reset_state_ids.shape}."
            )
        idx_t = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
        qpos_noise = torch.empty((len(env_idx), self.mjm.nq), dtype=torch.float32)
        qvel_noise = torch.empty((len(env_idx), self.mjm.nv), dtype=torch.float32)
        for row, state_id in enumerate(reset_state_ids):
            noise_gen = torch.Generator(device="cpu")
            noise_gen.manual_seed(self.seed + int(state_id))
            if self.spec.reset_noise_type == "normal":
                qpos_noise[row] = torch.randn(self.mjm.nq, generator=noise_gen)
                qvel_noise[row] = torch.randn(self.mjm.nv, generator=noise_gen)
            else:
                qpos_noise[row] = torch.rand(self.mjm.nq, generator=noise_gen) * 2 - 1
                qvel_noise[row] = torch.rand(self.mjm.nv, generator=noise_gen) * 2 - 1

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[idx_t] = True
        with self._wp.ScopedDevice(self._wp_device):
            self._mjw.reset_data(
                self.m,
                self.d,
                reset=self._wp.from_torch(reset_mask, dtype=self._wp.bool),
            )
        self._qpos[idx_t] = self._init_qpos + (
            qpos_noise * self.spec.reset_noise_scale
        ).to(self.device)
        self._qvel[idx_t] = self._init_qvel + (
            qvel_noise * self.spec.reset_noise_scale
        ).to(self.device)
        self._ctrl[idx_t] = 0.0
        # Rebuild contacts and derived quantities after selective resets.
        with self._wp.ScopedDevice(self._wp_device):
            self._mjw.forward(self.m, self.d)

        return self._obs_from_state()[idx_t].clone()

    def _vector_step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_shape = (self.num_envs, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {tuple(actions.shape)}."
            )
        actions = actions.detach().to(device=self.device, dtype=torch.float32)
        ctrl = torch.clamp(actions, self._ctrl_low, self._ctrl_high)
        x_before = self._qpos[:, 0].clone()
        self._ctrl.copy_(ctrl)
        with self._wp.ScopedDevice(self._wp_device):
            for _ in range(self.spec.frame_skip):
                self._mjw.step(self.m, self.d)
        x_after = self._qpos[:, 0]

        forward_vel = (x_after - x_before) / self.frame_dt
        ctrl_cost = self.spec.ctrl_cost_weight * (self._ctrl**2).sum(dim=1)
        rewards = self.spec.healthy_reward + forward_vel - ctrl_cost

        obs = self._obs_from_state()
        terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.spec.healthy_z_range is not None:
            # qpos layout is [rootx, rootz, rooty(angle), ...] for planar tasks
            # (skip=1) and [rootx, rooty, rootz, quat, ...] for 3D tasks
            # (skip=2).
            z = self._qpos[:, 2] if self.spec.obs_qpos_skip == 2 else self._qpos[:, 1]
            min_z, max_z = self.spec.healthy_z_range
            terminations |= (z < min_z) | (z > max_z)
        if self.spec.healthy_angle_range is not None:
            angle = self._qpos[:, 2]
            min_a, max_a = self.spec.healthy_angle_range
            terminations |= (angle < min_a) | (angle > max_a)
        if self.spec.healthy_z_range is not None:
            terminations |= ~torch.isfinite(obs).all(dim=1)

        return obs, rewards, terminations

    # ------------------------------------------------------------------
    # EnvWorker contract
    # ------------------------------------------------------------------
    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset selected worlds and return observations for the full batch."""
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

        partial_obs = self._vector_reset(env_idx_arr, reset_state_ids)
        if len(env_idx_arr) == self.num_envs:
            full_obs = self._obs_from_state().clone()
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
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        obs, rewards, terminations = self._vector_step(actions)

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
        """Step a chunk of actions and return EnvWorker-compatible batch outputs."""
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
        aggregated_final_info = None
        aggregated_final_obs = None
        for i in range(acts.shape[1]):
            (
                extracted_obs,
                step_reward,
                terminations,
                truncations,
                infos,
            ) = self.step(acts[:, i])
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

            step_dones = torch.logical_or(terminations, truncations)
            if step_dones.any() and self.auto_reset:
                aggregated_final_info = self._masked_update(
                    aggregated_final_info, infos["final_info"], step_dones
                )
                aggregated_final_obs = self._masked_update(
                    aggregated_final_obs, infos["final_observation"], step_dones
                )

        chunk_rewards_tensor = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations_tensor = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations_tensor = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations_tensor.any(dim=1)
        past_truncations = raw_chunk_truncations_tensor.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            infos_list[-1] = dict(infos_list[-1])
            infos_list[-1]["final_info"] = aggregated_final_info
            infos_list[-1]["final_observation"] = aggregated_final_obs
            infos_list[-1]["_final_info"] = past_dones
            infos_list[-1]["_final_observation"] = past_dones
            infos_list[-1]["_elapsed_steps"] = past_dones

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
        if self.is_eval and self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
        reset_state_ids = self._sample_reset_state_ids(env_idx)
        obs, infos = self.reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        dones_cpu = dones.cpu()
        infos["_final_info"] = dones_cpu
        infos["_final_observation"] = dones_cpu
        infos["_elapsed_steps"] = dones_cpu
        return obs, infos
