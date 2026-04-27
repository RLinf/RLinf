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

"""Base class for MuJoCo-Warp GPU-batched environments.

All task-specific logic is deferred to subclasses under ``tasks/``.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

_RENDER_CAM_NAME = "env_cam"


def _lookat_quat(eye, lookat) -> list:
    """Compute MuJoCo camera quaternion (wxyz) from eye + lookat positions.

    MuJoCo camera convention: camera looks along its -Z axis, +Y is up.
    Returns wxyz quaternion suitable for ``MjSpec`` camera ``quat`` field.
    """
    eye = np.asarray(eye, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    fwd = lookat - eye
    fwd /= np.linalg.norm(fwd)
    world_up = (
        np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    z_cam = -fwd
    y_cam = np.cross(z_cam, right)
    y_cam /= np.linalg.norm(y_cam)
    R = np.column_stack([right, y_cam, z_cam])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat.tolist()


class MuJoCoWarpEnv(gym.Env, ABC):
    """Base class for MuJoCo-Warp GPU-batched environments.

    Handles all shared infrastructure: GPU sim lifecycle, CUDA graph
    management, rendering, metrics, and auto-reset.  Subclasses under
    ``tasks/`` implement task-specific physics, observation, reward, and
    control.

    The constructor signature follows the standard RLinf env contract.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by each task
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _task_description(self) -> str:
        """Human-readable task description used in observation dict."""

    @property
    @abstractmethod
    def _timestep(self) -> float:
        """MuJoCo simulation timestep (seconds)."""

    @property
    @abstractmethod
    def _n_substeps(self) -> int:
        """Number of physics substeps per control step."""

    @abstractmethod
    def _build_scene(self, spec: mujoco.MjSpec, wb) -> None:
        """Add task-specific bodies, geoms, joints, and lights to the MJCF spec.

        Called during ``_build_sim()`` before compilation.  The spec is
        pre-loaded from the MJCF file returned by ``_mjcf_path``.
        """

    @abstractmethod
    def _configure_model_cpu(self) -> None:
        """Apply post-compile model configuration (actuator gains, solver
        params, etc.).  Called after ``spec.compile()``.
        """

    @abstractmethod
    def _init_task_state(self) -> None:
        """Cache body/joint/dof indices and other derived state after the
        CPU model is compiled and configured.  Called at the end of
        ``_build_sim()``.
        """

    @abstractmethod
    def _compute_ctrl(self, actions: np.ndarray) -> np.ndarray:
        """Convert raw policy actions to ctrl array for GPU write.

        Args:
            actions: ``(num_envs, action_dim)`` numpy float array.

        Returns:
            ``(num_envs, nu)`` numpy float32 ctrl array.
        """

    @abstractmethod
    def _build_obs_dict(self) -> dict[str, Any]:
        """Build the canonical RLinf observation dict from the current GPU state."""

    @abstractmethod
    def _compute_reward_and_done(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-step reward and terminal flag for all envs.

        Returns:
            ``(rewards, done)`` — each shape ``(num_envs,)``, float32.
        """

    @abstractmethod
    def _write_init_reset_states(
        self, indices: list[int], qpos_np: np.ndarray, qvel_np: np.ndarray
    ) -> None:
        """Write initial state for reset envs into the numpy arrays *in-place*.

        The arrays will be copied to GPU after this call (before settle
        physics, if any).  ``ctrl_np`` is available as ``self._mw_data.ctrl.numpy()``
        if the task needs to set initial ctrl.
        """

    # ------------------------------------------------------------------
    # Optional hooks — override to customise
    # ------------------------------------------------------------------

    @property
    def _mjcf_path(self) -> str:
        """Path to the primary MJCF XML file.  Override if not in assets/xml/."""
        raise NotImplementedError(
            "Subclass must provide _mjcf_path or override _load_spec"
        )

    def _load_spec(self) -> mujoco.MjSpec:
        """Load and return the MuJoCo spec.  Override to customise loading."""
        import os

        return mujoco.MjSpec.from_file(os.path.abspath(self._mjcf_path))

    @property
    def _cast_shadows(self) -> bool:
        """Whether environment lights cast shadows."""
        return False

    @property
    def _render_enabled_geom_groups(self) -> list[int]:
        """Geom groups to include in GPU renders."""
        return sorted(set(self._model_cpu.geom_group.tolist()))

    @property
    def _use_textures_in_render(self) -> bool:
        """Whether to use textures in GPU renders."""
        return True

    @property
    def _render_fps(self) -> int:
        """Framerate for metadata."""
        return 50

    @property
    def _use_reset_settle(self) -> bool:
        """Whether to run settle physics after writing reset states."""
        return False

    @property
    def _reset_settle_steps(self) -> int:
        """Number of physics steps for reset settle."""
        return 0

    @property
    def _track_reward_terms(self) -> bool:
        """Whether to track per-term reward metrics."""
        return False

    @property
    def _njmax(self) -> Optional[int]:
        """njmax buffer size for GPU data.  None uses default."""
        return None

    def _build_step_graph(self) -> Optional[wp.Graph]:
        """Capture the step graph.  Returns None if CUDA is unavailable."""
        if not wp.get_device().is_cuda:
            return None
        with wp.ScopedCapture() as capture:
            for _ in range(self._n_substeps):
                mjw.step(self._mw_model, self._mw_data)
        return capture.graph

    def _build_reset_graph(self) -> Optional[wp.Graph]:
        """Capture the reset-settle graph.  Returns None if unavailable."""
        if not self._use_reset_settle or not wp.get_device().is_cuda:
            return None
        with wp.ScopedCapture() as capture:
            for _ in range(self._reset_settle_steps):
                mjw.step(self._mw_model, self._mw_data)
        return capture.graph

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        self.metadata["render_fps"] = self._render_fps
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.group_size: int = cfg.group_size
        self.num_group: int = num_envs // self.group_size
        self.auto_reset: bool = cfg.auto_reset
        self.use_rel_reward: bool = cfg.use_rel_reward
        self.ignore_terminations: bool = cfg.ignore_terminations
        self.max_episode_steps: int = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg
        self.reward_coef: float = float(cfg.get("reward_coef", 1.0))
        self.use_fixed_reset_state_ids: bool = bool(
            cfg.get("use_fixed_reset_state_ids", False)
        )

        self.device = "cuda"
        self._is_start = True

        # Parse camera init_params (common to all tasks)
        init_params = (
            OmegaConf.to_container(cfg.init_params, resolve=True)
            if hasattr(cfg, "init_params") and cfg.init_params is not None
            else {}
        )
        self._cam_height: int = int(init_params.get("camera_height", 128))
        self._cam_width: int = int(init_params.get("camera_width", 128))
        self._cam_eye: list[float] = list(
            init_params.get("camera_eye", [0.0, 4.0, 2.5])
        )
        self._cam_lookat: list[float] = list(
            init_params.get("camera_lookat", [0.0, 0.0, 0.75])
        )
        self._cam_fov: float = float(init_params.get("camera_fov", 40.0))
        self._init_params = init_params

        self._rng = np.random.default_rng(self.seed)

        self._build_sim()

        # Tracking tensors (CPU)
        self.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)

        if record_metrics:
            self._init_metrics()

    # ------------------------------------------------------------------
    # Sim construction
    # ------------------------------------------------------------------

    def _build_sim(self) -> None:
        """Load MJCF, add scene, compile, build GPU model + data + graphs."""
        spec = self._load_spec()
        wb = spec.worldbody

        # Remove default keyframes (may be invalid after subclasses add bodies)
        for k in list(spec.keys):
            spec.delete(k)

        # Task-specific scene elements
        self._build_scene(spec, wb)

        # Common lights
        _l1 = wb.add_light()
        _l1.name = "env_light1"
        _l1.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        _l1.pos = [0.0, 0.0, 3.0]
        _l1.dir = [0.0, 0.0, -1.0]
        _l1.diffuse = [0.8, 0.8, 0.8]
        _l1.specular = [0.5, 0.5, 0.5]
        _l1.castshadow = self._cast_shadows

        _l2 = wb.add_light()
        _l2.name = "env_light2"
        _l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        _l2.pos = [-2.0, 2.0, 2.0]
        _l2.dir = [0.5, -0.5, -1.0]
        _l2.diffuse = [0.4, 0.4, 0.4]
        _l2.specular = [0.0, 0.0, 0.0]
        _l2.castshadow = self._cast_shadows

        # Camera
        cam = wb.add_camera()
        cam.name = _RENDER_CAM_NAME
        cam.pos = list(self._cam_eye)
        cam.quat = _lookat_quat(self._cam_eye, self._cam_lookat)
        cam.fovy = self._cam_fov

        self._model_cpu = spec.compile()
        self._model_cpu.opt.timestep = self._timestep

        # Task-specific post-compile initialisation (cache body/joint IDs
        # before _configure_model_cpu uses them for actuator/contact tuning)
        self._init_task_state()

        # Task-specific post-compile model tuning
        self._configure_model_cpu()

        # Cache camera index for GPU render calls
        self._render_cam_idx = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_CAMERA, _RENDER_CAM_NAME
        )

        # GPU model + batched data
        self._mw_model = mjw.put_model(self._model_cpu)
        njmax = self._njmax
        if njmax is not None:
            self._mw_data = mjw.make_data(
                self._model_cpu, nworld=self.num_envs, njmax=njmax
            )
        else:
            self._mw_data = mjw.make_data(self._model_cpu, nworld=self.num_envs)

        # Pre-write zero ctrl to GPU (fixes buffer addresses for CUDA graphs)
        wp.copy(
            self._mw_data.ctrl,
            wp.array(
                np.zeros((self.num_envs, self._model_cpu.nu), dtype=np.float32),
                dtype=wp.float32,
            ),
        )

        # Capture CUDA graphs
        self._step_graph = self._build_step_graph()
        self._reset_graph = self._build_reset_graph()

        # Lazy GPU render state
        self._render_ctx = None
        self._rgb_buf = None
        self._render_graph = None

        # Compute render fps based on control period
        ctrl_dt = self._timestep * self._n_substeps
        self.metadata["render_fps"] = int(round(1.0 / ctrl_dt))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments (all or a subset)."""
        if env_idx is None and options is not None and "env_idx" in options:
            env_idx = options["env_idx"]

        if env_idx is None:
            indices = list(range(self.num_envs))
            reset_tensor = None
        else:
            if isinstance(env_idx, torch.Tensor):
                indices = env_idx.tolist()
                reset_tensor = env_idx.to(dtype=torch.int64)
            elif isinstance(env_idx, np.ndarray):
                indices = env_idx.tolist()
                reset_tensor = torch.from_numpy(env_idx).to(dtype=torch.int64)
            else:
                indices = list(env_idx)
                reset_tensor = torch.tensor(indices, dtype=torch.int64)

        # Read current GPU state
        qpos_np = self._mw_data.qpos.numpy().copy()
        qvel_np = self._mw_data.qvel.numpy().copy()

        # Save non-reset env states if doing partial reset with settle physics
        partial_reset = len(indices) < self.num_envs
        saved_qpos = saved_qvel = saved_ctrl = None
        non_reset_idx: list[int] = []
        if partial_reset and self._use_reset_settle:
            non_reset_idx = [i for i in range(self.num_envs) if i not in set(indices)]
            saved_qpos = qpos_np[non_reset_idx].copy()
            saved_qvel = qvel_np[non_reset_idx].copy()
            saved_ctrl = self._mw_data.ctrl.numpy()[non_reset_idx].copy()

        # Write initial states for reset envs
        self._write_init_reset_states(indices, qpos_np, qvel_np)

        # Write to GPU
        wp.copy(self._mw_data.qpos, wp.array(qpos_np, dtype=wp.float32))
        wp.copy(self._mw_data.qvel, wp.array(qvel_np, dtype=wp.float32))

        # Reset settle physics
        if self._reset_graph is not None:
            wp.capture_launch(self._reset_graph)
            wp.synchronize()
        elif self._use_reset_settle:
            for _ in range(self._reset_settle_steps):
                mjw.step(self._mw_model, self._mw_data)

        # Restore non-reset envs
        if partial_reset and self._use_reset_settle:
            settled_qpos = self._mw_data.qpos.numpy().copy()
            settled_qvel = self._mw_data.qvel.numpy().copy()
            settled_ctrl = self._mw_data.ctrl.numpy().copy()
            settled_qpos[non_reset_idx] = saved_qpos
            settled_qvel[non_reset_idx] = saved_qvel
            settled_ctrl[non_reset_idx] = saved_ctrl
            wp.copy(self._mw_data.qpos, wp.array(settled_qpos, dtype=wp.float32))
            wp.copy(self._mw_data.qvel, wp.array(settled_qvel, dtype=wp.float32))
            wp.copy(self._mw_data.ctrl, wp.array(settled_ctrl, dtype=wp.float32))

        # Forward kinematics
        mjw.forward(self._mw_model, self._mw_data)
        wp.synchronize()

        self._reset_metrics(reset_tensor)
        self._post_reset_hook()
        obs = self._build_obs_dict()
        return obs, {}

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Execute one environment step."""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions_np = actions.cpu().numpy()
        if actions_np.ndim == 1:
            actions_np = actions_np[np.newaxis, :]

        self._elapsed_steps += 1

        # Compute and write ctrl
        ctrl_np = self._compute_ctrl(actions_np)
        wp.copy(self._mw_data.ctrl, wp.array(ctrl_np, dtype=wp.float32))

        # Replay step graph
        if self._step_graph is not None:
            wp.capture_launch(self._step_graph)
            wp.synchronize()
        else:
            for _ in range(self._n_substeps):
                mjw.step(self._mw_model, self._mw_data)

        # Post-step hook (e.g. forward kinematics, contact force reading)
        self._post_step_hook()

        obs = self._build_obs_dict()
        rewards_np, done_np = self._compute_reward_and_done()

        step_reward = torch.from_numpy(rewards_np).float()
        step_reward = self.reward_coef * step_reward

        reward_diff = step_reward - self.prev_step_reward
        self.prev_step_reward = step_reward.clone()
        if self.use_rel_reward:
            step_reward = reward_diff

        done = torch.from_numpy(done_np).bool()
        terminations = done.clone()
        truncations = self._elapsed_steps >= self.max_episode_steps

        infos: dict[str, Any] = {
            "success": done,
            "fail": torch.zeros(self.num_envs, dtype=torch.bool),
        }
        if self.record_metrics:
            infos = self._record_metrics(step_reward, done, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            terminations = torch.zeros_like(terminations)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def chunk_step(
        self,
        chunk_actions: torch.Tensor,
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute an action chunk and aggregate results."""
        chunk_size = chunk_actions.shape[1]
        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        chunk_rewards: list[torch.Tensor] = []
        raw_chunk_terminations: list[torch.Tensor] = []
        raw_chunk_truncations: list[torch.Tensor] = []

        for i in range(chunk_size):
            obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_term_t = torch.stack(raw_chunk_terminations, dim=1)
        raw_trunc_t = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_term_t.any(dim=1)
        past_truncations = raw_trunc_t.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_term_t)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_trunc_t)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_term_t.clone()
            chunk_truncations = raw_trunc_t.clone()

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _post_step_hook(self) -> None:
        """Called after step graph replay.  Override for forward kinematics,
        contact force reading, etc."""
        mjw.forward(self._mw_model, self._mw_data)
        wp.synchronize()

    def _post_reset_hook(self) -> None:
        """Called after reset completes (post-forward, pre-obs)."""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_images(self) -> Optional[np.ndarray]:
        """Render all parallel envs on GPU; return (B, H, W, 3) uint8 array."""
        if self._render_ctx is None:
            cam_active = [
                i == self._render_cam_idx for i in range(self._model_cpu.ncam)
            ]
            self._render_ctx = mjw.create_render_context(
                self._model_cpu,
                nworld=self.num_envs,
                cam_res=(self._cam_width, self._cam_height),
                render_rgb=True,
                render_depth=False,
                use_textures=self._use_textures_in_render,
                use_shadows=self._cast_shadows,
                cam_active=cam_active,
                enabled_geom_groups=self._render_enabled_geom_groups,
            )
            self._rgb_buf = wp.zeros(
                (self.num_envs, self._cam_height, self._cam_width),
                dtype=wp.vec3,
            )
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    mjw.refit_bvh(self._mw_model, self._mw_data, self._render_ctx)
                    mjw.render(self._mw_model, self._mw_data, self._render_ctx)
                self._render_graph = capture.graph

        if self._render_graph is not None:
            wp.capture_launch(self._render_graph)
            wp.synchronize()
        else:
            mjw.refit_bvh(self._mw_model, self._mw_data, self._render_ctx)
            mjw.render(self._mw_model, self._mw_data, self._render_ctx)

        mjw.get_rgb(self._render_ctx, self._render_cam_idx, self._rgb_buf)
        wp.synchronize()

        rgb_np = self._rgb_buf.numpy()
        return (rgb_np * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        if self._track_reward_terms:
            self.reward_term_returns: dict[str, torch.Tensor] = {}

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            idx = env_idx
            self.prev_step_reward[idx] = 0.0
            if self.record_metrics:
                self.success_once[idx] = False
                self.fail_once[idx] = False
                self.returns[idx] = 0.0
                for value in getattr(self, "reward_term_returns", {}).values():
                    value[idx] = 0.0
            self._elapsed_steps[idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                for value in getattr(self, "reward_term_returns", {}).values():
                    value[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(
        self,
        step_reward: torch.Tensor,
        success: torch.Tensor,
        infos: dict[str, Any],
    ) -> dict[str, Any]:
        self.returns += step_reward
        self.success_once = self.success_once | success.bool()
        episode_info: dict[str, Any] = {
            "success_once": self.success_once.clone(),
            "fail_once": self.fail_once.clone(),
            "return": self.returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self.returns / self._elapsed_steps.clamp(min=1).float(),
        }
        reward_terms = infos.get("reward_terms")
        if reward_terms and self._track_reward_terms:
            safe_len = self._elapsed_steps.clamp(min=1).float()
            for key, value in reward_terms.items():
                metric = value.detach().float()
                if key not in self.reward_term_returns:
                    self.reward_term_returns[key] = torch.zeros(
                        self.num_envs, dtype=torch.float32
                    )
                self.reward_term_returns[key] += metric
                episode_info[f"{key}_return"] = self.reward_term_returns[key].clone()
                episode_info[f"{key}_mean"] = self.reward_term_returns[key] / safe_len
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Auto-reset
    # ------------------------------------------------------------------

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset done environments, stashing the terminal observation."""
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)

        env_idx = torch.arange(self.num_envs)[dones]
        obs, infos = self.reset(env_idx=env_idx)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    # ------------------------------------------------------------------
    # RLinf worker interface
    # ------------------------------------------------------------------

    def update_reset_state_ids(self) -> None:
        """No-op: satisfies the RLinf worker interface."""
        pass

    def close(self) -> None:
        """Free resources."""
        self._render_ctx = None
        self._rgb_buf = None
        self._render_graph = None
