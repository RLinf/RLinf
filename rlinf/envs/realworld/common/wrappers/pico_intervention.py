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

"""PICO intervention wrappers for Franka environments."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.pico.pico_expert import PicoExpert
from rlinf.utils.rot6d import matrix_to_rot6d


def _match_action_space(
    action: np.ndarray,
    fallback_action: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Pad/truncate an expert action to a target action shape."""
    target_dim = int(np.prod(target_shape))
    action_flat = np.asarray(action, dtype=np.float32).reshape(-1)

    if action_flat.size == target_dim:
        return action_flat.reshape(target_shape)

    matched = np.zeros((target_dim,), dtype=np.float32)
    copy_dim = min(action_flat.size, target_dim)
    matched[:copy_dim] = action_flat[:copy_dim]

    fallback_flat = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
    if action_flat.size < target_dim and fallback_flat.size >= target_dim:
        matched[action_flat.size : target_dim] = fallback_flat[
            action_flat.size : target_dim
        ]

    return matched.reshape(target_shape)


def _dict_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    return dict(config)


def _rotvec_action_to_euler_action(
    expert_action: np.ndarray,
    action_scale: np.ndarray,
) -> np.ndarray:
    """Adapt PicoExpert's rotvec delta action to Euler-delta Franka envs."""
    action = np.asarray(expert_action, dtype=np.float32).reshape(-1).copy()
    if action.size < 6:
        return action

    scale = float(np.asarray(action_scale, dtype=np.float64)[1])
    if scale <= 1e-9:
        action[3:6] = 0.0
        return action

    delta_rot = R.from_rotvec(np.asarray(action[3:6], dtype=np.float64) * scale)
    action[3:6] = np.clip(delta_rot.as_euler("xyz") / scale, -1.0, 1.0)
    return action


def _split_dual_pico_config(pico_config: Mapping[str, Any]) -> tuple[dict, dict]:
    """Build left/right PICO configs from shared or nested config blocks."""
    cfg = _dict_config(pico_config)
    shared = {k: v for k, v in cfg.items() if k not in ("hand", "left", "right")}
    shared_calibration = _dict_config(shared.get("calibration"))

    left_cfg = shared.copy()
    left_override = _dict_config(cfg.get("left"))
    left_cfg.update(left_override)
    if shared_calibration or "calibration" in left_override:
        left_calibration = shared_calibration.copy()
        left_calibration.update(_dict_config(left_override.get("calibration")))
        left_cfg["calibration"] = left_calibration

    right_cfg = shared.copy()
    right_override = _dict_config(cfg.get("right"))
    right_cfg.update(right_override)
    if shared_calibration or "calibration" in right_override:
        right_calibration = shared_calibration.copy()
        right_calibration.update(_dict_config(right_override.get("calibration")))
        right_cfg["calibration"] = right_calibration

    # Dual-arm mode always binds left controller to left arm and right to right.
    left_cfg["hand"] = "left"
    right_cfg["hand"] = "right"
    return left_cfg, right_cfg


class PicoIntervention(gym.ActionWrapper):
    """Override policy actions with PICO controller input.

    ``hand="left"`` or ``hand="right"`` creates one PICO controller.  On a
    single-arm env it controls the whole action; on a dual-arm env it controls
    only that arm's action slice.  ``hand="dual"`` creates both controllers and
    maps left/right controller input to left/right arm respectively.
    """

    def __init__(
        self,
        env,
        gripper_enabled: bool = True,
        **pico_config: Any,
    ):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.hand = str(pico_config.get("hand", "right")).lower()
        if self.hand not in ("left", "right", "dual"):
            raise ValueError(
                "PicoIntervention hand must be 'left', 'right', or 'dual'."
            )

        if self.hand == "dual":
            left_cfg, right_cfg = _split_dual_pico_config(pico_config)
            self.experts = {
                "left": PicoExpert(**left_cfg),
                "right": PicoExpert(**right_cfg),
            }
        else:
            single_cfg = {
                key: value
                for key, value in pico_config.items()
                if key not in ("left", "right")
            }
            single_cfg["hand"] = self.hand
            self.experts = {self.hand: PicoExpert(**single_cfg)}
        self.left = False
        self.right = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.left = False
        self.right = False
        return obs, info

    def _arm_action_slice(self, arm_idx: int, per_arm_dim: int) -> slice:
        start = arm_idx * per_arm_dim
        return slice(start, start + per_arm_dim)

    def _match_arm_action(
        self,
        expert_action: np.ndarray,
        fallback_arm_action: np.ndarray,
        per_arm_dim: int,
    ) -> np.ndarray:
        return _match_action_space(
            expert_action,
            fallback_arm_action,
            (per_arm_dim,),
        )

    def _update_hand_flags(self, side: str, active: bool) -> None:
        if side == "left":
            self.left = active
        else:
            self.right = active

    def _action_shape(self) -> tuple[tuple[int, ...], int]:
        target_shape = self.env.action_space.shape
        target_dim = int(np.prod(target_shape))
        return target_shape, target_dim

    def _single_action(
        self,
        action: np.ndarray,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        side, expert = next(iter(self.experts.items()))
        target_shape, target_dim = self._action_shape()
        action_flat = np.asarray(action, dtype=np.float32).reshape(-1)

        if tcp_pose.size == 7:
            expert_action, replaced, pico_info = expert.get_action(
                tcp_pose,
                action_scale,
                gripper_enabled=self.gripper_enabled,
            )
            active = bool(pico_info.get("pico_active", False))
            self._update_hand_flags(side, active)
            if replaced:
                expert_action = _rotvec_action_to_euler_action(
                    expert_action,
                    action_scale,
                )
                expert_action = _match_action_space(
                    expert_action,
                    action_flat,
                    target_shape,
                )
                return expert_action, True, pico_info
            return action, False, pico_info

        if tcp_pose.size != 14:
            raise ValueError(
                "PicoIntervention expects get_tcp_pose() to return 7 or 14 values, "
                f"got shape {tcp_pose.shape}."
            )
        if target_dim % 2 != 0:
            raise ValueError(
                "PicoIntervention expects an even dual-arm action dimension, "
                f"got action_space.shape={target_shape}."
            )

        per_arm_dim = target_dim // 2
        if action_flat.size != target_dim:
            action_flat = _match_action_space(action_flat, action_flat, (target_dim,))
        new_action = action_flat.copy()

        arm_idx = 0 if side == "left" else 1
        tcp_slice = slice(arm_idx * 7, arm_idx * 7 + 7)
        action_slice = self._arm_action_slice(arm_idx, per_arm_dim)
        expert_action, replaced, pico_info = expert.get_action(
            tcp_pose[tcp_slice],
            action_scale,
            gripper_enabled=self.gripper_enabled,
        )

        active = bool(pico_info.get("pico_active", False))
        self._update_hand_flags(side, active)
        if replaced:
            expert_action = _rotvec_action_to_euler_action(
                expert_action,
                action_scale,
            )
            new_action[action_slice] = self._match_arm_action(
                expert_action,
                action_flat[action_slice],
                per_arm_dim,
            )
            return new_action.reshape(target_shape), True, pico_info
        return action, False, pico_info

    def _dual_action(
        self,
        action: np.ndarray,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        if tcp_pose.size != 14:
            raise ValueError(
                "PicoIntervention hand='dual' expects get_tcp_pose() to return 14 "
                "values "
                f"(two xyz+quat poses), got shape {tcp_pose.shape}."
            )

        target_shape = self.env.action_space.shape
        target_dim = int(np.prod(target_shape))
        if target_dim % 2 != 0:
            raise ValueError(
                "PicoIntervention hand='dual' expects an even dual-arm action "
                "dimension, "
                f"got action_space.shape={target_shape}."
            )

        per_arm_dim = target_dim // 2
        action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_flat.size != target_dim:
            action_flat = _match_action_space(action_flat, action_flat, (target_dim,))
        new_action = action_flat.copy()

        replaced_any = False
        pico_info: dict[str, Any] = {}
        for arm_idx, (side, expert) in enumerate(self.experts.items()):
            tcp_slice = slice(arm_idx * 7, arm_idx * 7 + 7)
            action_slice = self._arm_action_slice(arm_idx, per_arm_dim)
            expert_action, replaced, side_info = expert.get_action(
                tcp_pose[tcp_slice],
                action_scale,
                gripper_enabled=self.gripper_enabled,
            )

            side_active = bool(side_info.get("pico_active", False))
            self._update_hand_flags(side, side_active)

            for key, value in side_info.items():
                pico_info[f"{side}_{key}"] = value

            if replaced:
                expert_action = _rotvec_action_to_euler_action(
                    expert_action,
                    action_scale,
                )
                new_action[action_slice] = self._match_arm_action(
                    expert_action,
                    action_flat[action_slice],
                    per_arm_dim,
                )
                replaced_any = True

        pico_info["pico_active"] = bool(self.left or self.right)
        pico_info["pico_ready"] = bool(
            pico_info.get("left_pico_ready", False)
            or pico_info.get("right_pico_ready", False)
        )
        return new_action.reshape(target_shape), replaced_any, pico_info

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        tcp_pose = np.asarray(self.get_wrapper_attr("get_tcp_pose")(), dtype=np.float32)
        action_scale = self.get_wrapper_attr("get_action_scale")()
        if self.hand == "dual":
            return self._dual_action(action, tcp_pose, action_scale)
        return self._single_action(action, tcp_pose, action_scale)

    def step(self, action):
        new_action, replaced, pico_info = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info.update(pico_info)
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        for expert in self.experts.values():
            expert.stop()
        return super().close()


class DualFrankaTcpPicoIntervention(gym.ActionWrapper):
    """PICO teleoperation for ``DualFrankaTcpEnv`` absolute rot6d actions.

    ``PicoExpert`` emits a 7D delta-TCP action with rotation as a normalized
    rotvec.  This wrapper adapts that output to the dual TCP env layout:
    ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
    """

    def __init__(
        self,
        env,
        gripper_enabled: bool = True,
        hold_current_when_inactive: bool = True,
        **pico_config: Any,
    ):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.hold_current_when_inactive = bool(hold_current_when_inactive)
        self.hand = str(pico_config.get("hand", "dual")).lower()
        if self.hand not in ("left", "right", "dual"):
            raise ValueError(
                "DualFrankaTcpPicoIntervention hand must be 'left', 'right', or 'dual'."
            )

        if self.hand == "dual":
            left_cfg, right_cfg = _split_dual_pico_config(pico_config)
            self.experts = {
                "left": PicoExpert(**left_cfg),
                "right": PicoExpert(**right_cfg),
            }
        else:
            single_cfg = {
                key: value
                for key, value in pico_config.items()
                if key not in ("left", "right")
            }
            single_cfg["hand"] = self.hand
            self.experts = {self.hand: PicoExpert(**single_cfg)}

        self.left = False
        self.right = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.left = False
        self.right = False
        return obs, info

    @staticmethod
    def _arm_index(side: str) -> int:
        return 0 if side == "left" else 1

    @staticmethod
    def _tcp_pose_to_rot6d_action(
        tcp_pose: np.ndarray,
        gripper_action: float = 0.0,
    ) -> np.ndarray:
        rot6d = matrix_to_rot6d(R.from_quat(tcp_pose[3:7]).as_matrix())
        return np.concatenate(
            [
                np.asarray(tcp_pose[:3], dtype=np.float32),
                rot6d.astype(np.float32),
                np.array([gripper_action], dtype=np.float32),
            ],
            axis=0,
        )

    def _current_tcp_action(
        self,
        tcp_pose: np.ndarray,
        fallback_action: np.ndarray,
        arm_idx: int,
        per_arm_dim: int,
    ) -> np.ndarray:
        tcp_slice = slice(arm_idx * 7, arm_idx * 7 + 7)
        action_slice = slice(arm_idx * per_arm_dim, arm_idx * per_arm_dim + per_arm_dim)
        fallback_gripper = 0.0
        if fallback_action.size >= action_slice.stop:
            fallback_gripper = float(fallback_action[action_slice][-1])
        return self._tcp_pose_to_rot6d_action(tcp_pose[tcp_slice], fallback_gripper)

    def _expert_delta_to_tcp_action(
        self,
        expert_action: np.ndarray,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
    ) -> np.ndarray:
        expert_action = np.asarray(expert_action, dtype=np.float32).reshape(-1)
        if expert_action.size < 6:
            raise ValueError(
                "DualFrankaTcpPicoIntervention expects PicoExpert to return at "
                f"least 6 motion dims, got shape {expert_action.shape}."
            )

        action_scale = np.asarray(action_scale, dtype=np.float64)
        current_pos = np.asarray(tcp_pose[:3], dtype=np.float64)
        current_rot = R.from_quat(np.asarray(tcp_pose[3:7], dtype=np.float64))

        target_pos = current_pos + expert_action[:3] * float(action_scale[0])
        rot_action = np.asarray(expert_action[3:6], dtype=np.float64)
        rot_norm = float(np.linalg.norm(rot_action))
        if rot_norm > 1.0:
            rot_action = rot_action / rot_norm
        delta_rot = R.from_rotvec(rot_action * float(action_scale[1]))
        target_rot = delta_rot * current_rot
        gripper_action = float(expert_action[6]) if expert_action.size >= 7 else 0.0

        target_pose = np.concatenate([target_pos, target_rot.as_quat()])
        return self._tcp_pose_to_rot6d_action(target_pose, gripper_action)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        target_shape = self.env.action_space.shape
        target_dim = int(np.prod(target_shape))
        if target_dim != 20:
            raise ValueError(
                "DualFrankaTcpPicoIntervention expects DualFrankaTcpEnv's 20D "
                f"action space, got action_space.shape={target_shape}."
            )

        tcp_pose = np.asarray(self.get_wrapper_attr("get_tcp_pose")(), dtype=np.float32)
        if tcp_pose.size != 14:
            raise ValueError(
                "DualFrankaTcpPicoIntervention expects get_tcp_pose() to return "
                f"14 values, got shape {tcp_pose.shape}."
            )

        action_scale = self.get_wrapper_attr("get_action_scale")()
        action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_flat.size != target_dim:
            action_flat = _match_action_space(action_flat, action_flat, (target_dim,))
        else:
            action_flat = action_flat.copy()

        per_arm_dim = target_dim // 2
        if self.hold_current_when_inactive:
            new_action = np.concatenate(
                [
                    self._current_tcp_action(tcp_pose, action_flat, 0, per_arm_dim),
                    self._current_tcp_action(tcp_pose, action_flat, 1, per_arm_dim),
                ]
            ).astype(np.float32)
        else:
            new_action = action_flat.copy()

        replaced_any = False
        replaced_by_side: dict[str, bool] = {}
        pico_info: dict[str, Any] = {}
        for side, expert in self.experts.items():
            arm_idx = self._arm_index(side)
            tcp_slice = slice(arm_idx * 7, arm_idx * 7 + 7)
            action_slice = slice(
                arm_idx * per_arm_dim,
                arm_idx * per_arm_dim + per_arm_dim,
            )
            expert_action, replaced, side_info = expert.get_action(
                tcp_pose[tcp_slice],
                action_scale,
                gripper_enabled=self.gripper_enabled,
            )

            side_active = bool(side_info.get("pico_active", False))
            if side == "left":
                self.left = side_active
            else:
                self.right = side_active

            for key, value in side_info.items():
                pico_info[f"{side}_{key}"] = value

            replaced_by_side[side] = bool(replaced)
            pico_info[f"{side}_pico_replaced"] = bool(replaced)
            if replaced:
                new_action[action_slice] = self._expert_delta_to_tcp_action(
                    expert_action,
                    tcp_pose[tcp_slice],
                    action_scale,
                )
                replaced_any = True

        pico_info["pico_active"] = bool(self.left or self.right)
        pico_info["pico_ready"] = bool(
            pico_info.get("left_pico_ready", False)
            or pico_info.get("right_pico_ready", False)
        )
        pico_info["pico_replaced"] = replaced_any
        pico_info["pico_dual_replaced"] = bool(
            replaced_by_side.get("left", False) and replaced_by_side.get("right", False)
        )

        new_action = np.clip(
            new_action,
            self.env.action_space.low.reshape(-1),
            self.env.action_space.high.reshape(-1),
        )
        return new_action.reshape(target_shape), replaced_any, pico_info

    def step(self, action):
        new_action, replaced, pico_info = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        should_record_intervention = replaced
        if self.hand == "dual" and not self.hold_current_when_inactive:
            should_record_intervention = bool(pico_info["pico_dual_replaced"])
        if should_record_intervention or self.hold_current_when_inactive:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        info.update(pico_info)
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        for expert in self.experts.values():
            expert.stop()
        return super().close()
