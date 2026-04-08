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
#
# SpacemouseSimIntervention — wraps a vectorized GenieSimEnv and injects
# SpaceMouse teleoperation for env_0.
#
# Action layout assumed (ee mode, action_dim=14):
#   [0:3]  left  EE position  (base_link frame, metres)
#   [3:6]  left  EE euler XYZ (base_link frame, radians)
#   [6:9]  right EE position  (base_link frame, metres)
#   [9:12] right EE euler XYZ (base_link frame, radians)
#   [12]   left  gripper command
#   [13]   right gripper command
#
# State layout assumed (state_dim=40):
#   [0:7]   left  arm joint positions
#   [7:14]  right arm joint positions
#   [14:21] left  arm joint velocities
#   [21:28] right arm joint velocities
#   [28:31] left  EE position
#   [31:34] left  EE RPY
#   [34:37] right EE position
#   [37:40] right EE RPY
#
# SpaceMouse button semantics (see ``button_mode`` on ``SpacemouseSimIntervention``):
#
#   ``legacy`` (default):
#     Translation/rotation → right arm EEF delta (accumulated → absolute target)
#     Left  button → right arm gripper close
#     Right button → episode done + success (save demo when CollectEpisode.only_success)
#
#   ``place_workpiece`` (GenieSim collect_sim_data):
#     Translation/rotation → same as above
#     Left  button → episode done + success (save)
#     Right button → episode done + failure (discard when only_success=True)
#     No gripper mapping from buttons
#
#   ``training`` (online RL with human-in-the-loop):
#     Translation/rotation → same as above
#     Left  button → right arm gripper close
#     Right button → right arm gripper open
#     No episode termination from buttons — env terminates naturally

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# SpaceMouse expert interfaces
# ---------------------------------------------------------------------------


class SpaceMouseExpertBase:
    """Abstract base for SpaceMouse-like experts."""

    def get_action(self) -> tuple[np.ndarray, list]:
        """Return (action[6], buttons[2]).

        action: [dx, dy, dz, droll, dpitch, dyaw] — unit-ish values in
                SpaceMouse sensor frame (already remapped to robot base frame).
        buttons: [left_button, right_button] — each 0 or 1.
        """
        raise NotImplementedError

    def on_episode_reset(self):
        """Called when the environment resets so the expert can update state."""


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------


class SpacemouseSimIntervention:
    """Wraps a vectorized GenieSimEnv to inject SpaceMouse teleoperation.

    Only env ``intervention_env_id`` (default: 0) is driven by the SpaceMouse;
    all other envs receive the unmodified policy actions.

    The SpaceMouse outputs a 6-DoF *delta* which is accumulated on top of the
    right arm's current end-effector target.  The underlying environment
    expects an *absolute* EEF target in ee-control mode, so this wrapper
    maintains an internal ``_current_target`` that starts at the EE pose
    read from the reset observation and is updated each step.

    Button semantics (``button_mode``)
    ----------------------------------
    ``legacy`` (default):
        Left  button : close the right arm gripper.
        Right button : episode done and successful (save when CollectEpisode
                       uses ``only_success=True``).

    ``place_workpiece`` (GenieSim demonstration collection):
        Left  button : episode done and successful (save).
        Right button : episode done and unsuccessful (discard when
                       ``only_success=True``).
        No gripper action from buttons.

    ``training`` (online RL with human-in-the-loop):
        Left  button : close the right arm gripper.
        Right button : open the right arm gripper.
        No episode termination from buttons — the environment terminates
        naturally via its own success/failure conditions.

    ``intervene_action`` / ``intervene_flag``
    -----------------------------------------
    When the SpaceMouse is active (non-zero motion or relevant buttons), the
    wrapper writes::

        info["intervene_action"] = actions[intervention_env_id]  # tensor
        info["intervene_flag"] = torch.ones(1, dtype=torch.bool)

    CollectEpisode's LeRobot writer uses these fields to replace the
    policy action with the expert action when saving demos.

    Args:
        env: Vectorized GenieSimEnv (or compatible).
        expert: SpaceMouseExpertBase instance.
        action_scale: Multiplier applied to SpaceMouse translational output
            before adding to the EEF position target (metres per unit).
        rotation_scale: Multiplier applied to SpaceMouse rotational output
            before adding to the EEF orientation target (radians per unit).
        intervention_env_id: Index of the env instance driven by SpaceMouse.
        button_mode: ``"legacy"``, ``"place_workpiece"``, or ``"training"`` — see class docstring.
    """

    # Default action vector indices (ee mode, action_dim=14, both arms)
    _L_POS = slice(0, 3)
    _L_ROT = slice(3, 6)
    _R_POS = slice(6, 9)
    _R_ROT = slice(9, 12)
    _L_GRIP = 12
    _R_GRIP = 13

    # Default state vector indices for EE poses (state_dim=52)
    _S_L_EE_POS = slice(28, 31)
    _S_L_EE_ROT = slice(31, 34)
    _S_R_EE_POS = slice(40, 43)
    _S_R_EE_ROT = slice(43, 46)

    def __init__(
        self,
        env,
        expert: SpaceMouseExpertBase,
        action_scale: float = 0.01,
        rotation_scale: float = 0.05,
        intervention_env_id: int = 0,
        button_mode: str = "legacy",
        *,
        action_dim: int = 14,
        r_pos_action: slice | None = None,
        r_rot_action: slice | None = None,
        r_grip_action: int | None = None,
        s_r_ee_pos: slice | None = None,
        s_r_ee_rot: slice | None = None,
        s_l_ee_pos: slice | None = None,
        s_l_ee_rot: slice | None = None,
        action_rescale: tuple | None = None,
    ):
        self.env = env
        self.expert = expert
        self.action_scale = action_scale
        self.rotation_scale = rotation_scale
        self.intervention_env_id = intervention_env_id
        if button_mode not in ("legacy", "place_workpiece", "training"):
            raise ValueError(
                f"button_mode must be 'legacy', 'place_workpiece', or 'training', got {button_mode!r}"
            )
        self.button_mode = button_mode
        self._action_dim = action_dim

        if r_pos_action is not None:
            self._R_POS = r_pos_action
        if r_rot_action is not None:
            self._R_ROT = r_rot_action
        if r_grip_action is not None:
            self._R_GRIP = r_grip_action
        if s_r_ee_pos is not None:
            self._S_R_EE_POS = s_r_ee_pos
        if s_r_ee_rot is not None:
            self._S_R_EE_ROT = s_r_ee_rot
        if s_l_ee_pos is not None:
            self._S_L_EE_POS = s_l_ee_pos
        if s_l_ee_rot is not None:
            self._S_L_EE_ROT = s_l_ee_rot

        self._gripper_state: float = 0.0
        self._last_intervene_time: float = 0.0
        self._intervene_timeout: float = 0.5
        self._post_reset_cooldown: float = 1.0
        self._cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_target_from_obs(self, obs: dict[str, Any]) -> None:
        self._gripper_state = 0.0

    # ------------------------------------------------------------------
    # gym-compatible interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self._init_target_from_obs(obs)
        self._last_intervene_time = 0.0
        self._btn_cooldown_until = time.time() + 0.5
        self._cooldown_until = time.time() + self._post_reset_cooldown
        self.expert.on_episode_reset()
        return obs, info

    def step(
        self,
        actions,
        **kwargs,
    ) -> tuple[dict, Any, Any, Any, dict]:
        """Inject SpaceMouse action for env_0, pass others through unchanged."""
        if isinstance(actions, torch.Tensor):
            actions = actions.clone().float()
        else:
            actions = torch.tensor(np.asarray(actions, dtype=np.float32))

        sm_delta, buttons = self.expert.get_action()
        left_btn = bool(buttons[0]) if len(buttons) > 0 else False
        right_btn = bool(buttons[1]) if len(buttons) > 1 else False

        if time.time() < getattr(self, "_btn_cooldown_until", 0.0):
            left_btn = False
            right_btn = False

        has_motion = float(np.linalg.norm(sm_delta)) > 0.05

        in_cooldown = time.time() < self._cooldown_until
        if in_cooldown:
            has_motion = False
            left_btn = False
            right_btn = False

        if self.button_mode == "place_workpiece":
            intervened = has_motion or left_btn or right_btn
        elif self.button_mode == "training":
            intervened = has_motion or left_btn or right_btn
        else:
            intervened = has_motion or left_btn

        if intervened:
            self._last_intervene_time = time.time()

        delta_action = np.zeros(self._action_dim, dtype=np.float32)
        if has_motion:
            sm_delta[3], sm_delta[4] = sm_delta[4], -sm_delta[3]
            from rlinf.envs.geniesim.tasks.place_workpiece import (
                _POS_SCALE,
                _RPY_SCALE,
            )

            delta_action[self._R_POS] = np.clip(
                sm_delta[:3] * self.action_scale / _POS_SCALE,
                -1.0,
                1.0,
            )
            delta_action[self._R_ROT] = np.clip(
                sm_delta[3:] * self.rotation_scale / _RPY_SCALE,
                -1.0,
                1.0,
            )

        if self.button_mode == "legacy" and left_btn:
            self._gripper_state = -1.0
        elif self.button_mode == "training":
            if left_btn:
                self._gripper_state = -1.0
            elif right_btn:
                self._gripper_state = 1.0
        delta_action[self._R_GRIP] = self._gripper_state

        eid = self.intervention_env_id

        active = intervened or (
            time.time() - self._last_intervene_time < self._intervene_timeout
        )

        if active or self.button_mode in ("place_workpiece",):
            for i in range(self._action_dim):
                actions[eid, i] = float(delta_action[i])

        if self.button_mode == "training":
            obs, reward, terminated, truncated, info = self.env.step(actions, **kwargs)
        else:
            obs, reward, terminated, truncated, info = self.env.step(
                actions, auto_reset=False, **kwargs
            )

        if self.button_mode == "place_workpiece":
            if left_btn or right_btn:
                terminated = (
                    terminated.clone()
                    if isinstance(terminated, torch.Tensor)
                    else torch.tensor(np.array(terminated))
                )
                terminated[eid] = True
                success = bool(left_btn)
                info["success"] = success
                if "episode" in info and isinstance(info["episode"], dict) and success:
                    sc = info["episode"].get("success_once")
                    if isinstance(sc, torch.Tensor):
                        sc = sc.clone()
                        sc[eid] = True
                        info["episode"]["success_once"] = sc
        elif self.button_mode == "legacy" and right_btn:
            terminated = (
                terminated.clone()
                if isinstance(terminated, torch.Tensor)
                else torch.tensor(np.array(terminated))
            )
            terminated[eid] = True
            info["success"] = True
            if "episode" in info and isinstance(info["episode"], dict):
                sc = info["episode"].get("success_once")
                if isinstance(sc, torch.Tensor):
                    sc = sc.clone()
                    sc[eid] = True
                    info["episode"]["success_once"] = sc

        if self.button_mode == "training":
            num_envs = actions.shape[0]
            ia = actions.clone()
            iflags = torch.zeros(num_envs, dtype=torch.bool)
            if active:
                iflags[eid] = True
            info["intervene_action"] = ia
            info["intervene_flag"] = iflags
        else:
            if active:
                info["intervene_action"] = actions[eid]
                info["intervene_flag"] = torch.ones(1, dtype=torch.bool)

        if self.button_mode == "training":
            dones = (
                (terminated | truncated)
                if isinstance(terminated, torch.Tensor)
                else torch.tensor(np.array(terminated))
                | torch.tensor(np.array(truncated))
            )
            if dones[eid]:
                self._init_target_from_obs(obs)
                self.expert.on_episode_reset()
                self._cooldown_until = time.time() + self._post_reset_cooldown
            elif not active:
                self._init_target_from_obs(obs)

        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions) -> tuple:
        if not isinstance(chunk_actions, torch.Tensor):
            chunk_actions = torch.as_tensor(np.asarray(chunk_actions, dtype=np.float32))

        if chunk_actions.ndim == 2:
            chunk_actions = chunk_actions.unsqueeze(1)

        num_action_chunks = chunk_actions.shape[1]

        obs_list: list = []
        rewards_list: list = []
        terminated_list: list = []
        truncated_list: list = []
        infos_list: list = []

        for i in range(num_action_chunks):
            act = chunk_actions[:, i, :]
            obs, rewards, terminated, truncated, infos = self.step(
                act, auto_reset=getattr(self.env, "auto_reset", True)
            )
            obs_list.append(obs)
            rewards_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos_list.append(infos)

        chunk_rewards_t = torch.stack(rewards_list, dim=1)
        chunk_terminations_t = torch.stack(terminated_list, dim=1)
        chunk_truncations_t = torch.stack(truncated_list, dim=1)

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations_t,
            chunk_truncations_t,
            infos_list,
        )

    def close(self):
        return self.env.close()

    # Delegate attribute lookups so wrappers stacked on top work correctly
    def __getattr__(self, name: str):
        return getattr(self.env, name)
