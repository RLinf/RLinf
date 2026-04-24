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

"""Classic CartPole task for MuJoCo-Warp GPU backend.

Mirrors the MuJoCo and Genesis CartPole exactly:
  - Same geometry and masses (cart 1 kg box, pole 0.1 kg cylinder)
  - Same MJCF model from assets/xml/cartpole.xml
  - Same reward function and coefficients
  - Same termination conditions (|angle|>0.2 rad or |cart_pos|>2.4 m)
  - Same action interface (1-D raw force, tanh-squashed to ±3 N)
  - Same control period: timestep=0.01s × n_substeps=2 → 0.02s/step

MuJoCo-Warp difference:
  - All num_envs worlds are stepped in parallel on GPU via mjw.step()
  - State is read back as batched numpy arrays: shape (num_envs, ...)
  - Control is written as a batched warp array: shape (num_envs, nu)

State (4 dims): cart_pos, cart_vel, pole_angle, pole_omega
Action (1 dim): raw policy output → tanh → force in [-3, 3] N on slider
"""

from __future__ import annotations

import copy
import os
from typing import Any, Optional, Union

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# MJCF file — shared with the MuJoCo and Genesis backends
# ---------------------------------------------------------------------------
_MJCF_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "xml", "cartpole.xml",
)

_RENDER_CAM_NAME = "env_cam"


def _lookat_quat(eye, lookat) -> list:
    """Compute MuJoCo camera quaternion (wxyz) from eye + lookat positions.

    MuJoCo camera convention: camera looks along its -Z axis, +Y is up.
    Returns wxyz quaternion suitable for ``MjSpec`` camera ``quat`` field.
    """
    eye    = np.asarray(eye,    dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    fwd = lookat - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0., 0., 1.]) if abs(fwd[2]) < 0.99 else np.array([0., 1., 0.])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    z_cam = -fwd                                # camera looks in -Z
    y_cam = np.cross(z_cam, right)
    y_cam /= np.linalg.norm(y_cam)
    R = np.column_stack([right, y_cam, z_cam])  # columns = camera axes in world
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())      # mujoco expects row-major
    return quat.tolist()                        # wxyz

# ---------------------------------------------------------------------------
# Physics constants (must match MuJoCo / Genesis / ManiSkill setup)
# ---------------------------------------------------------------------------
_TRACK_HALF_LENGTH = 2.4   # m  — position termination threshold
_MAX_ANGLE         = 0.2   # rad — angle termination threshold (~11.5°)
_MAX_FORCE         = 3.0   # N  — maximum applied force (after tanh)

# Reward shaping coefficients (identical to other backends)
_W_CART_POS   = 0.2
_W_POLE_ANGLE = 0.3
_W_CART_VEL   = 0.5
_W_ACTION     = 0.1
_VEL_NORM     = 1.5   # m/s — velocity normalisation factor

_TASK_DESCRIPTION = "Balance the pole on the cart by applying horizontal force."


class MuJoCoWarpCartPoleEnv(gym.Env):
    """CartPole balancing task backed by MuJoCo-Warp (GPU-batched physics).

    Matches the MuJoCo and Genesis CartPole tasks exactly.  All num_envs
    worlds are stepped in a single GPU kernel call via mjw.step(), providing
    significantly higher throughput than the sequential CPU MuJoCo backend.

    The constructor signature follows the standard RLinf env contract.

    Args:
        cfg: Hydra ``DictConfig`` for this env (e.g. ``cfg.env.train``).
        num_envs: Number of parallel environments this worker manages.
        seed_offset: Unique seed offset for this worker process.
        total_num_processes: Total number of env worker processes.
        worker_info: Opaque worker metadata from the scheduler.
        record_metrics: Whether to track success/return metrics.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
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

        self.device = "cuda"  # MuJoCo-Warp runs on GPU
        self._is_start = True

        # Parse init_params
        init_params = (
            OmegaConf.to_container(cfg.init_params, resolve=True)
            if hasattr(cfg, "init_params") and cfg.init_params is not None
            else {}
        )
        self._timestep: float = float(init_params.get("timestep", 0.01))
        self._n_substeps: int = int(init_params.get("n_substeps", 2))
        self._track_half_length: float = float(
            init_params.get("track_half_length", _TRACK_HALF_LENGTH)
        )
        self._max_angle: float = float(init_params.get("max_angle", _MAX_ANGLE))
        self._max_force: float = float(init_params.get("max_force", _MAX_FORCE))
        self._cam_height: int = int(init_params.get("camera_height", 128))
        self._cam_width: int = int(init_params.get("camera_width", 128))
        self._cam_eye: list[float] = list(init_params.get("camera_eye", [0.0, 4.0, 2.5]))
        self._cam_lookat: list[float] = list(
            init_params.get("camera_lookat", [0.0, 0.0, 0.75])
        )
        self._cam_fov: float = float(init_params.get("camera_fov", 40.0))

        self._rng = np.random.default_rng(self.seed)

        self._build_sim()

        # Tracking tensors (CPU)
        self.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._last_norm_action = torch.zeros(num_envs, 1, dtype=torch.float32)

        if record_metrics:
            self._init_metrics()

    # ------------------------------------------------------------------
    # MuJoCo-Warp setup
    # ------------------------------------------------------------------

    def _build_sim(self) -> None:
        """Load MJCF, add camera + lights, build MuJoCo-Warp model + CUDA graph."""
        mjcf_path = os.path.abspath(_MJCF_FILE)

        # Build spec so we can inject render camera and directional lights
        spec = mujoco.MjSpec.from_file(mjcf_path)

        l1 = spec.worldbody.add_light()
        l1.name = "env_light1"
        l1.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l1.pos = [0.0, 0.0, 3.0]; l1.dir = [0.0, 0.0, -1.0]
        l1.diffuse = [0.8, 0.8, 0.8]; l1.specular = [0.5, 0.5, 0.5]
        l1.castshadow = False

        l2 = spec.worldbody.add_light()
        l2.name = "env_light2"
        l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l2.pos = [-2.0, 2.0, 2.0]; l2.dir = [0.5, -0.5, -1.0]
        l2.diffuse = [0.4, 0.4, 0.4]; l2.specular = [0.0, 0.0, 0.0]
        l2.castshadow = False

        cam = spec.worldbody.add_camera()
        cam.name = _RENDER_CAM_NAME
        cam.pos  = list(self._cam_eye)
        cam.quat = _lookat_quat(self._cam_eye, self._cam_lookat)
        cam.fovy = self._cam_fov

        self._model_cpu = spec.compile()
        self._model_cpu.opt.timestep = self._timestep

        # Cache camera index for GPU render calls
        self._render_cam_idx = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_CAMERA, _RENDER_CAM_NAME
        )

        # GPU model
        self._mw_model = mjw.put_model(self._model_cpu)

        # Batched GPU data: one world per parallel env
        self._mw_data = mjw.make_data(self._model_cpu, nworld=self.num_envs)

        # Capture mjw.step() as a CUDA graph for maximum throughput.
        # The graph records all GPU kernel launches for one step; subsequent
        # wp.capture_launch(graph) replays them with zero Python overhead.
        # wp.copy() is used for all ctrl / qpos / qvel updates so the GPU
        # buffer addresses captured in the graph never change.
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                mjw.step(self._mw_model, self._mw_data)
            self._step_graph = capture.graph
        else:
            self._step_graph = None

        # Lazy GPU render state (created on first _render_images call)
        self._render_ctx = None
        self._rgb_buf    = None
        self._render_graph = None

    # ------------------------------------------------------------------
    # Gymnasium interface
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

    def reset(
        self,
        env_idx: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments (all or a subset).

        Args:
            env_idx: Indices of envs to reset. ``None`` → reset all.

        Returns:
            ``(obs_dict, infos)`` following the gymnasium convention.
        """
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

        # Read current GPU state (copy so we can modify specific worlds)
        qpos_np = self._mw_data.qpos.numpy().copy()  # (nworld, nq)
        qvel_np = self._mw_data.qvel.numpy().copy()  # (nworld, nv)

        for i in indices:
            # Random initial perturbation (matches MuJoCo/Genesis/ManiSkill)
            cart_pos = float(np.clip(
                self._rng.normal(0.0, 0.05),
                -self._track_half_length, self._track_half_length,
            ))
            pole_angle = float(np.clip(
                self._rng.normal(0.0, 0.05), -self._max_angle, self._max_angle
            ))
            qpos_np[i, 0] = cart_pos
            qpos_np[i, 1] = pole_angle
            qvel_np[i, :] = 0.0

        # Write back to GPU in-place (wp.copy keeps buffer addresses stable
        # so the captured CUDA graph remains valid)
        wp.copy(self._mw_data.qpos, wp.array(qpos_np, dtype=wp.float32))
        wp.copy(self._mw_data.qvel, wp.array(qvel_np, dtype=wp.float32))

        # Run forward kinematics to update derived quantities (not captured)
        mjw.forward(self._mw_model, self._mw_data)

        self._reset_metrics(reset_tensor)
        qpos_np = self._mw_data.qpos.numpy()
        qvel_np = self._mw_data.qvel.numpy()
        obs = self._wrap_obs(qpos_np, qvel_np)
        return obs, {}

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: Action tensor of shape ``(num_envs, 1)`` (raw policy output).
            auto_reset: Whether to auto-reset terminated/truncated envs.

        Returns:
            ``(obs, step_reward, terminations, truncations, infos)`` tuple.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions = actions.cpu()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)  # ensure (B, 1)

        self._elapsed_steps += 1

        # Tanh squash then scale to force
        norm_action = torch.tanh(actions)           # (B, 1) → (−1, 1)
        forces = (norm_action * self._max_force)    # (B, 1) → (−3, 3) N
        self._last_norm_action = norm_action.detach().clone()

        # Write ctrl to all worlds in-place (preserves GPU buffer address)
        ctrl_np = forces.numpy().astype(np.float32)  # (B, 1)
        wp.copy(self._mw_data.ctrl, wp.array(ctrl_np, dtype=wp.float32))

        # Step all worlds in parallel — replay CUDA graph when available.
        # NOTE: no wp.synchronize() between substeps — CUDA graph launches
        # queue on the GPU stream and run back-to-back.  The first .numpy()
        # call after the loop implicitly syncs the device.
        for _ in range(self._n_substeps):
            if self._step_graph is not None:
                wp.capture_launch(self._step_graph)
            else:
                mjw.step(self._mw_model, self._mw_data)

        # Read GPU state ONCE and pass to helpers (avoids 3 redundant syncs)
        qpos_np = self._mw_data.qpos.numpy()  # (nworld, 2)
        qvel_np = self._mw_data.qvel.numpy()  # (nworld, 2)

        obs = self._wrap_obs(qpos_np, qvel_np)
        fail = self._check_done(qpos_np)
        step_reward = self._calc_step_reward(fail, qpos_np, qvel_np)
        terminations = fail
        truncations = self._elapsed_steps >= self.max_episode_steps

        infos: dict[str, Any] = {"fail": fail, "success": ~fail}
        if self.record_metrics:
            infos = self._record_metrics(step_reward, ~fail, infos)

        if self.ignore_terminations:
            if "episode" in infos:
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
        """Execute an action chunk (multiple steps) and aggregate results.

        Args:
            chunk_actions: Tensor of shape ``(num_envs, chunk_size, action_dim)``.
        """
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

        return (obs_list, chunk_rewards_t, chunk_terminations, chunk_truncations, infos_list)

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _check_done(self, qpos_np: np.ndarray) -> torch.Tensor:
        """Termination conditions (same as MuJoCo/Genesis CartPole)."""
        cart_pos   = torch.from_numpy(qpos_np[:, 0].copy())
        pole_angle = torch.from_numpy(qpos_np[:, 1].copy())
        return (pole_angle.abs() > self._max_angle) | (cart_pos.abs() > self._track_half_length)

    def _calc_step_reward(
        self, fail: torch.Tensor, qpos_np: np.ndarray, qvel_np: np.ndarray
    ) -> torch.Tensor:
        """Compute per-step reward (identical formula to MuJoCo/Genesis)."""
        cart_pos   = torch.from_numpy(qpos_np[:, 0].copy())
        pole_angle = torch.from_numpy(qpos_np[:, 1].copy())
        cart_vel   = torch.from_numpy(qvel_np[:, 0].copy())

        alive = (~fail).float()

        cart_pos_cost   = (cart_pos   / self._track_half_length).pow(2)
        pole_angle_cost = (pole_angle / self._max_angle).pow(2)
        cart_vel_cost   = (cart_vel   / _VEL_NORM).pow(2)
        action_cost     = self._last_norm_action[:, 0].pow(2)

        raw_reward = alive * (
            1.0
            - _W_CART_POS   * cart_pos_cost
            - _W_POLE_ANGLE * pole_angle_cost
            - _W_CART_VEL   * cart_vel_cost
            - _W_ACTION     * action_cost
        )
        reward = self.reward_coef * raw_reward
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward.clone()
        if self.use_rel_reward:
            return reward_diff
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _wrap_obs(self, qpos_np: np.ndarray, qvel_np: np.ndarray) -> dict[str, Any]:
        """Build canonical RLinf observation dict.

        State order: [cart_pos, cart_vel, pole_angle, pole_omega]
        Matches the MuJoCo/Genesis observation layout exactly.
        """
        cart_pos   = torch.from_numpy(qpos_np[:, 0].copy())
        cart_vel   = torch.from_numpy(qvel_np[:, 0].copy())
        pole_angle = torch.from_numpy(qpos_np[:, 1].copy())
        pole_omega = torch.from_numpy(qvel_np[:, 1].copy())
        states = torch.stack([cart_pos, cart_vel, pole_angle, pole_omega], dim=-1)  # (B, 4)

        obs: dict[str, Any] = {
            "states": states,
            "task_descriptions": [_TASK_DESCRIPTION] * self.num_envs,
        }

        if getattr(self.video_cfg, "save_video", False):
            imgs = self._render_images()
            if imgs is not None:
                obs["main_images"] = torch.from_numpy(imgs)

        return obs

    # ------------------------------------------------------------------
    # Rendering (GPU-side via mjw render pipeline)
    # ------------------------------------------------------------------

    def _render_images(self) -> Optional[np.ndarray]:
        """Render all parallel envs on GPU; return (B, H, W, 3) uint8 array."""
        if self._render_ctx is None:
            cam_active = [i == self._render_cam_idx
                          for i in range(self._model_cpu.ncam)]
            enabled_geom_groups = sorted(set(self._model_cpu.geom_group.tolist()))
            self._render_ctx = mjw.create_render_context(
                self._model_cpu,
                nworld=self.num_envs,
                cam_res=(self._cam_width, self._cam_height),
                render_rgb=True,
                render_depth=False,
                use_textures=True,
                use_shadows=False,
                cam_active=cam_active,
                enabled_geom_groups=enabled_geom_groups,
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

        rgb_np = self._rgb_buf.numpy()  # (B, H, W, 3) float32 in [0, 1]
        return (rgb_np * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            idx = env_idx
            self.prev_step_reward[idx] = 0.0
            if self.record_metrics:
                self.success_once[idx] = False
                self.fail_once[idx] = False
                self.returns[idx] = 0.0
            self._elapsed_steps[idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(
        self,
        step_reward: torch.Tensor,
        success: torch.Tensor,
        infos: dict[str, Any],
    ) -> dict[str, Any]:
        self.returns += step_reward
        self.success_once = self.success_once | success.bool()
        fail = infos.get("fail", None)
        if fail is not None:
            self.fail_once = self.fail_once | fail.bool()
        episode_info: dict[str, Any] = {
            "success_once": self.success_once.clone(),
            "fail_once": self.fail_once.clone(),
            "return": self.returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self.returns / self._elapsed_steps.clamp(min=1).float(),
        }
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
        self._render_ctx   = None
        self._rgb_buf      = None
        self._render_graph = None
