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

"""Wrapper-stack builders shared by realworld task factories."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym

from rlinf.envs.realworld.common.wrappers.dual_euler_obs import (
    DualQuat2EulerWrapper,
)
from rlinf.envs.realworld.common.wrappers.dual_gello_intervention import (
    DualGelloIntervention,
)
from rlinf.envs.realworld.common.wrappers.dual_pose_action import (
    DualAbsolutePoseActionWrapper,
    DualRelativePoseActionWrapper,
)
from rlinf.envs.realworld.common.wrappers.dual_relative_frame import (
    DualRelativeFrame,
)
from rlinf.envs.realworld.common.wrappers.dual_spacemouse_intervention import (
    DualSpacemouseIntervention,
)
from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.gello_intervention import (
    GelloIntervention,
)
from rlinf.envs.realworld.common.wrappers.gripper_close import GripperCloseEnv
from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame
from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
)
from rlinf.envs.realworld.common.wrappers.spacemouse_intervention import (
    SpacemouseIntervention,
)


def _validate_teleop_mode(use_spacemouse: bool, use_gello: bool) -> None:
    if use_spacemouse and use_gello:
        raise ValueError(
            "Only one teleop mode can be active at a time. "
            "Set exactly one of use_spacemouse, use_gello to True."
        )


def _apply_keyboard_reward(env: gym.Env, mode: Optional[str]) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if config.is_dummy or not mode:
        return env
    if mode == "multi_stage":
        return KeyboardRewardDoneMultiStageWrapper(env)
    if mode == "single_stage":
        return KeyboardRewardDoneWrapper(env)
    return env


def apply_single_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for single-arm realworld envs (franka single, xsquare)."""
    no_gripper = cfg.get("no_gripper", True)
    if no_gripper:
        env = GripperCloseEnv(env)

    use_spacemouse = cfg.get("use_spacemouse", True)
    use_gello = cfg.get("use_gello", False)
    _validate_teleop_mode(use_spacemouse, use_gello)

    gripper_enabled = not no_gripper

    if not env.config.is_dummy and use_spacemouse:
        env = SpacemouseIntervention(env, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and use_gello:
        gello_port = cfg.get("gello_port", None)
        if gello_port is None:
            raise ValueError(
                "use_gello=True requires 'gello_port' in the env config "
                "(e.g. env.eval.gello_port)."
            )
        env = GelloIntervention(env, port=gello_port, gripper_enabled=gripper_enabled)

    env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = RelativeFrame(env)
    env = Quat2EulerWrapper(env, keep_gripper=not no_gripper)
    return env


def apply_dual_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for dual-arm realworld envs.

    ``action_mode`` defaults to ``delta_axis_angle`` to preserve the existing
    dual-arm teleop/training stack.  Turtle2 deploy can opt into
    ``relative_pose`` or ``absolute_pose`` action modes.  Those pose-action
    modes route actions through explicit pose-action wrappers, skip teleop and
    gripper-close wrappers, and differ only in frame handling:
    ``relative_pose`` may use ``DualRelativeFrame`` while ``absolute_pose``
    stays in the base frame before converting observations to euler angles.
    """
    action_mode = cfg.get("action_mode", "delta_axis_angle")
    if action_mode not in {"delta_axis_angle", "relative_pose", "absolute_pose"}:
        raise ValueError(
            f"Unsupported action_mode={action_mode!r}. "
            "Expected one of {'delta_axis_angle', 'relative_pose', 'absolute_pose'}."
        )

    if cfg.get("no_gripper", True):
        raise NotImplementedError(
            "Dual-arm realworld wrappers require no_gripper=False. "
            "Pose-action modes use 7D per-arm actions including gripper, "
            "and delta_axis_angle mode does not have DualGripperCloseEnv yet."
        )

    if action_mode == "relative_pose":
        if cfg.get("use_spacemouse", False) or cfg.get("use_gello", False):
            raise ValueError(
                "Dual-arm pose-action modes do not support teleop wrappers. "
                "Set use_spacemouse=False and use_gello=False."
            )
        env = DualRelativePoseActionWrapper(env)
        if cfg.get("use_relative_frame", True):
            env = DualRelativeFrame(env)
        env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))
        env = DualQuat2EulerWrapper(env)
        return env

    if action_mode == "absolute_pose":
        if cfg.get("use_spacemouse", False) or cfg.get("use_gello", False):
            raise ValueError(
                "Dual-arm pose-action modes do not support teleop wrappers. "
                "Set use_spacemouse=False and use_gello=False."
            )
        env = DualAbsolutePoseActionWrapper(env)
        env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))
        env = DualQuat2EulerWrapper(env)
        return env

    use_spacemouse = cfg.get("use_spacemouse", True)
    use_gello = cfg.get("use_gello", False)
    _validate_teleop_mode(use_spacemouse, use_gello)

    gripper_enabled = True

    if not env.config.is_dummy and use_spacemouse:
        env = DualSpacemouseIntervention(env, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and use_gello:
        left_port = cfg.get("left_gello_port", None)
        right_port = cfg.get("right_gello_port", None)
        if left_port is None or right_port is None:
            raise ValueError(
                "use_gello=True on a dual-arm env requires both "
                "'left_gello_port' and 'right_gello_port' in the env config."
            )
        env = DualGelloIntervention(
            env,
            left_port=left_port,
            right_port=right_port,
            gripper_enabled=gripper_enabled,
        )

    env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = DualRelativeFrame(env)
    env = DualQuat2EulerWrapper(env)
    return env
