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

"""Genesis environment wrapper for RLinf.

``GenesisEnv`` is a ``gymnasium.Env`` subclass that bridges the Genesis
physics simulator with RLinf's embodied RL training loop.  It uses
Genesis's native GPU-batched parallel simulation
(``scene.build(n_envs=N)``) so that all parallel environments run in a
single process on the GPU -- no subprocess overhead.

Usage (via config)::

    env:
      train:
        env_type: genesis
        init_params:
          task_name: cube_pick
          robot_file: xml/franka_emika_panda/panda.xml
          dt: 0.01
          camera_height: 480
          camera_width: 640
        group_size: 1
        max_episode_steps: 200
        auto_reset: true
        seed: 42
        ...
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Union

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.genesis.tasks import get_task_cls
from rlinf.envs.utils import to_tensor

# Number of physics steps executed during reset so the scene can settle.
_RESET_SETTLE_STEPS = 5


class GenesisEnv(gym.Env):
    """RLinf-compatible wrapper around a Genesis GPU-batched scene.

    The constructor signature follows the standard RLinf env contract so
    that it can be instantiated by ``EnvWorker`` without any special-casing.

    Args:
        cfg: Hydra ``DictConfig`` for this env (e.g. ``cfg.env.train``).
        num_envs: Number of parallel environments this worker manages.
        seed_offset: Unique seed offset for this worker process.
        total_num_processes: Total number of env worker processes.
        worker_info: Opaque worker metadata from the scheduler.
        record_metrics: Whether to track success/return metrics.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 30}

    # ------------------------------------------------------------------
    # Construction
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
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        # Standard RLinf env config fields.
        self.group_size: int = cfg.group_size
        self.num_group: int = num_envs // self.group_size
        self.auto_reset: bool = cfg.auto_reset
        self.use_rel_reward: bool = cfg.use_rel_reward
        self.ignore_terminations: bool = cfg.ignore_terminations
        self.use_fixed_reset_state_ids: bool = cfg.use_fixed_reset_state_ids
        self.max_episode_steps: int = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg
        self.reward_coef: float = float(cfg.get("reward_coef", 1.0))

        self._is_start = True
        self._generator = np.random.default_rng(seed=self.seed)

        # ----- Build Genesis scene via the task -----
        init_params = (
            OmegaConf.to_container(cfg.init_params, resolve=True)
            if hasattr(cfg, "init_params") and cfg.init_params is not None
            else {}
        )

        task_name: str = init_params.get("task_name", "cube_pick")
        task_cls = get_task_cls(task_name)
        self.task = task_cls()
        self.task.seed(self.seed)

        self._build_genesis_scene(cfg, init_params)

        # All metric / counter tensors live on the same device as the
        # simulation (gs.device) so that arithmetic with reward / success
        # tensors returned by the task avoids cross-device errors.
        self.device = gs.device

        # ----- Metrics and step counters -----
        self.prev_step_reward = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            num_envs, dtype=torch.int32, device=self.device
        )
        self._init_metrics()

    # ------------------------------------------------------------------
    # Genesis scene construction
    # ------------------------------------------------------------------

    def _build_genesis_scene(self, cfg, init_params: dict) -> None:
        """Initialize Genesis, create a scene, and build it."""
        # Guard: only initialize Genesis once per process.
        if not getattr(gs, "_initialized", False):
            # BatchRenderer requires the CUDA backend.
            backend = init_params.get("backend", "cuda")
            backend_map = {
                "gpu": gs.gpu,
                "cuda": gs.cuda,
                "cpu": gs.cpu,
            }
            gs.init(
                backend=backend_map.get(backend, gs.cuda),
                precision=str(init_params.get("precision", "32")),
                logging_level="warning",
            )

        dt = float(init_params.get("dt", 0.01))
        gravity = tuple(init_params.get("gravity", (0.0, 0.0, -9.81)))

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            renderer=gs.renderers.BatchRenderer(use_rasterizer=True),
            show_viewer=False,
        )

        # Let the task populate the scene with entities.
        self.task.build_scene(self.scene, cfg)

        # Build with GPU-parallel environments.
        env_spacing = tuple(init_params.get("env_spacing", (1.0, 1.0)))
        self.scene.build(n_envs=self.num_envs, env_spacing=env_spacing)

        # Post-build hook (e.g. to resolve link references).
        if hasattr(self.task, "post_build"):
            self.task.post_build()

    # ------------------------------------------------------------------
    # Properties required by the RLinf env worker
    # ------------------------------------------------------------------

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    # ------------------------------------------------------------------
    # Metrics (identical pattern to MetaWorld / FrankaSim / etc.)
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.fail_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            # Ensure the index tensor lives on the same device as the
            # metric tensors so that advanced indexing works correctly.
            idx = env_idx.to(self.device)
            self.prev_step_reward[idx] = 0.0
            self.success_once[idx] = False
            self.fail_once[idx] = False
            self.returns[idx] = 0.0
            self._elapsed_steps[idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
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
        self.success_once = self.success_once | success
        episode_info: dict[str, Any] = {}
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self._elapsed_steps.clone()
        # Avoid division by zero for episode_len == 0.
        safe_len = self._elapsed_steps.clamp(min=1).float()
        episode_info["reward"] = self.returns / safe_len
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Observation wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert raw task observations into the canonical RLinf dict.

        Canonical keys:
        * ``main_images``: ``torch.Tensor`` of shape ``(B, H, W, 3)`` uint8.
        * ``states``:       ``torch.Tensor`` of shape ``(B, state_dim)`` float.
        * ``task_descriptions``: ``list[str]`` of length ``B``.
        """
        obs: dict[str, Any] = {}

        # Images
        images = raw_obs.get("images")
        if images is not None:
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images.copy())
            obs["main_images"] = images

        # States
        states = raw_obs.get("states")
        if states is not None:
            if isinstance(states, np.ndarray):
                states = torch.from_numpy(states).float()
            obs["states"] = states

        # Task description (broadcast to batch)
        desc = self.task.task_description
        obs["task_descriptions"] = [desc] * self.num_envs

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calc_step_reward(
        self, success: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-step reward from success signal.

        Supports ``use_rel_reward`` (differential reward) mode.
        """
        reward = self.reward_coef * success.float()
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward.clone()
        if self.use_rel_reward:
            return reward_diff
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments (all or a subset).

        Args:
            env_idx: Optional array of environment indices to reset.
                ``None`` means reset all environments.
            seed: Optional seed (unused -- seed is set at construction).
            options: Optional dict; ``options["env_idx"]`` is an
                alternative way to specify the reset indices.

        Returns:
            ``(obs_dict, infos)`` tuple following the gymnasium convention.
        """
        # Allow env_idx via options dict (compatibility with gymnasium API).
        if env_idx is None and options is not None and "env_idx" in options:
            env_idx = options["env_idx"]

        # Convert to torch tensor for Genesis API.
        envs_idx_tensor: torch.Tensor | None = None
        if env_idx is not None:
            if isinstance(env_idx, np.ndarray):
                envs_idx_tensor = torch.from_numpy(env_idx).to(dtype=torch.int64, device=gs.device)
            elif isinstance(env_idx, torch.Tensor):
                envs_idx_tensor = env_idx.to(dtype=torch.int64, device=gs.device)
            else:
                envs_idx_tensor = torch.tensor(env_idx, dtype=torch.int64, device=gs.device)

        # Delegate to task.
        self.task.reset(self.scene, self.num_envs, envs_idx=envs_idx_tensor)

        # Settle physics.
        for _ in range(_RESET_SETTLE_STEPS):
            self.scene.step()

        # Observations.
        raw_obs = self.task.get_obs(self.scene, self.num_envs)
        obs = self._wrap_obs(raw_obs)

        # Reset metrics.
        if env_idx is not None:
            self._reset_metrics(envs_idx_tensor)
        else:
            self._reset_metrics()

        infos: dict[str, Any] = {}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: Action tensor of shape ``(num_envs, action_dim)``.
                For the default Franka task this is 9-dim (7 arm + 2 gripper).
            auto_reset: Whether to auto-reset terminated/truncated envs.

        Returns:
            ``(obs, step_reward, terminations, truncations, infos)`` tuple.
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(gs.device)
        elif actions.device != gs.device:
            actions = actions.to(gs.device)

        self._elapsed_steps += 1

        # Apply actions to the robot.
        task = self.task
        if task.motor_dofs is not None:
            task.robot.control_dofs_position(
                actions[:, : len(task.motor_dofs)], task.motor_dofs
            )
        if task.finger_dofs is not None:
            task.robot.control_dofs_position(
                actions[:, len(task.motor_dofs) : len(task.motor_dofs) + len(task.finger_dofs)],
                task.finger_dofs,
            )

        # Advance simulation.
        self.scene.step()

        # Observations.
        raw_obs = task.get_obs(self.scene, self.num_envs)
        obs = self._wrap_obs(raw_obs)

        # Reward / termination / truncation.
        _reward, success = task.compute_reward(self.scene, self.num_envs)
        step_reward = self._calc_step_reward(success)
        terminations = success.bool()
        truncations = self._elapsed_steps >= self.max_episode_steps

        infos: dict[str, Any] = {"success": success}
        infos = self._record_metrics(step_reward, success, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            terminations = torch.zeros_like(terminations)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Chunk step (action-chunking interface for embodied policies)
    # ------------------------------------------------------------------

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

        Returns:
            ``(obs_list, chunk_rewards, chunk_terminations, chunk_truncations,
              infos_list)`` where the reward/termination/truncation tensors
            have shape ``(num_envs, chunk_size)``.
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

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)  # (B, C)
        raw_term_t = torch.stack(raw_chunk_terminations, dim=1)  # (B, C)
        raw_trunc_t = torch.stack(raw_chunk_truncations, dim=1)  # (B, C)

        past_terminations = raw_term_t.any(dim=1)
        past_truncations = raw_trunc_t.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        # Collapse termination/truncation to last step only when auto-reset
        # is enabled, matching the convention used by other RLinf envs.
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

        # Indices of done environments.
        env_idx = torch.arange(self.num_envs, device=self.device)[dones]

        obs, infos = self.reset(env_idx=env_idx)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    # ------------------------------------------------------------------
    # Reset-state management
    # ------------------------------------------------------------------

    def update_reset_state_ids(self) -> None:
        """Advance the internal RNG so the next epoch samples fresh states.

        Called by the env worker at the end of each rollout epoch.
        """
        # Genesis tasks are stateless w.r.t. reset-state IDs; the task's
        # internal RNG is automatically advanced each time ``reset`` is
        # called.  This method exists to satisfy the RLinf worker interface.
        pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Destroy the Genesis scene and free GPU resources."""
        if hasattr(self, "scene") and self.scene is not None:
            try:
                self.scene = None
            except Exception:
                pass
