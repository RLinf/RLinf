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

import os
from typing import Any

import mujoco
import numpy as np
import torch

from rlinf.envs.mujoco_warp.mujoco_warp_env import MuJoCoWarpEnv

# ---------------------------------------------------------------------------
# MJCF file
# ---------------------------------------------------------------------------
_MJCF_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "xml",
    "cartpole.xml",
)

# ---------------------------------------------------------------------------
# Physics constants (must match MuJoCo / Genesis / ManiSkill setup)
# ---------------------------------------------------------------------------
_TRACK_HALF_LENGTH = 2.4
_MAX_ANGLE = 0.2
_MAX_FORCE = 3.0

# Reward shaping coefficients (identical to other backends)
_W_CART_POS = 0.2
_W_POLE_ANGLE = 0.3
_W_CART_VEL = 0.5
_W_ACTION = 0.1
_VEL_NORM = 1.5

_TASK_DESCRIPTION = "Balance the pole on the cart by applying horizontal force."

_DEFAULT_TIMESTEP = 0.01
_DEFAULT_N_SUBSTEPS = 2


class CartPoleTask(MuJoCoWarpEnv):
    """CartPole balancing task backed by MuJoCo-Warp (GPU-batched physics)."""

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def _task_description(self) -> str:
        return _TASK_DESCRIPTION

    @property
    def _timestep(self) -> float:
        return self._init_params.get("timestep", _DEFAULT_TIMESTEP)

    @property
    def _n_substeps(self) -> int:
        return int(self._init_params.get("n_substeps", _DEFAULT_N_SUBSTEPS))

    @property
    def _mjcf_path(self) -> str:
        return _MJCF_FILE

    @property
    def _cast_shadows(self) -> bool:
        return False

    def _build_scene(self, spec: mujoco.MjSpec, wb) -> None:
        """CartPole MJCF already contains all needed elements."""

    def _configure_model_cpu(self) -> None:
        """No extra model tuning needed for CartPole."""

    def _init_task_state(self) -> None:
        """Cache CartPole-specific constants from init_params."""
        self._track_half_length = float(
            self._init_params.get("track_half_length", _TRACK_HALF_LENGTH)
        )
        self._max_angle = float(self._init_params.get("max_angle", _MAX_ANGLE))
        self._max_force = float(self._init_params.get("max_force", _MAX_FORCE))
        self._last_norm_action = torch.zeros(self.num_envs, 1, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _compute_ctrl(self, actions: np.ndarray) -> np.ndarray:
        """Tanh squash then scale to force."""
        norm_action = torch.tanh(torch.from_numpy(actions.copy()).float())
        self._last_norm_action = norm_action.detach().clone()
        forces = (norm_action * self._max_force).numpy().astype(np.float32)
        return forces

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict[str, Any]:
        """Build 4-D state: [cart_pos, cart_vel, pole_angle, pole_omega]."""
        qpos_np = self._mw_data.qpos.numpy()
        qvel_np = self._mw_data.qvel.numpy()

        cart_pos = torch.from_numpy(qpos_np[:, 0].copy())
        cart_vel = torch.from_numpy(qvel_np[:, 0].copy())
        pole_angle = torch.from_numpy(qpos_np[:, 1].copy())
        pole_omega = torch.from_numpy(qvel_np[:, 1].copy())
        states = torch.stack([cart_pos, cart_vel, pole_angle, pole_omega], dim=-1)

        obs: dict[str, Any] = {
            "states": states,
            "task_descriptions": [self._task_description] * self.num_envs,
        }

        if getattr(self.video_cfg, "save_video", False):
            imgs = self._render_images()
            if imgs is not None:
                obs["main_images"] = torch.from_numpy(imgs)

        return obs

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def _compute_reward_and_done(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-step reward and termination."""
        qpos_np = self._mw_data.qpos.numpy()
        qvel_np = self._mw_data.qvel.numpy()

        cart_pos = torch.from_numpy(qpos_np[:, 0].copy())
        pole_angle = torch.from_numpy(qpos_np[:, 1].copy())
        cart_vel = torch.from_numpy(qvel_np[:, 0].copy())

        fail = (pole_angle.abs() > self._max_angle) | (
            cart_pos.abs() > self._track_half_length
        )
        alive = (~fail).float()

        cart_pos_cost = (cart_pos / self._track_half_length).pow(2)
        pole_angle_cost = (pole_angle / self._max_angle).pow(2)
        cart_vel_cost = (cart_vel / _VEL_NORM).pow(2)
        action_cost = self._last_norm_action[:, 0].pow(2)

        raw_reward = (
            (
                alive
                * (
                    1.0
                    - _W_CART_POS * cart_pos_cost
                    - _W_POLE_ANGLE * pole_angle_cost
                    - _W_CART_VEL * cart_vel_cost
                    - _W_ACTION * action_cost
                )
            )
            .numpy()
            .astype(np.float32)
        )

        return raw_reward, fail.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _write_init_reset_states(
        self, indices: list[int], qpos_np: np.ndarray, qvel_np: np.ndarray
    ) -> None:
        """Random initial perturbation (matches MuJoCo/Genesis/ManiSkill)."""
        for i in indices:
            cart_pos = float(
                np.clip(
                    self._rng.normal(0.0, 0.05),
                    -self._track_half_length,
                    self._track_half_length,
                )
            )
            pole_angle = float(
                np.clip(self._rng.normal(0.0, 0.05), -self._max_angle, self._max_angle)
            )
            qpos_np[i, 0] = cart_pos
            qpos_np[i, 1] = pole_angle
            qvel_np[i, :] = 0.0
