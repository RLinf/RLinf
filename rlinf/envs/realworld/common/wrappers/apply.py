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
from rlinf.envs.realworld.common.wrappers.master_takeover_intervention import (
    MasterTakeoverIntervention,
)
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
    env = Quat2EulerWrapper(env)
    return env


def apply_dual_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for dual-arm realworld envs (dual-franka today)."""
    if cfg.get("no_gripper", True):
        # No DualGripperCloseEnv yet, so a 12D action would blow up as reshape(2,7).
        raise NotImplementedError(
            "no_gripper=True is not yet supported for dual-arm envs: "
            "DualGripperCloseEnv is not implemented. "
            "Set env.eval.no_gripper=False (or env.train.no_gripper=False)."
        )

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


def apply_dual_pose_action_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for dual-arm pose-action deploy envs.

    This mirrors the dual-arm tail of ``apply_dual_arm_wrappers``:
    ``DualRelativeFrame`` plus ``DualQuat2EulerWrapper``. It skips teleop and
    gripper-close wrappers because deploy envs do not run human interventions.
    """
    action_mode = cfg.get("action_mode", "relative_pose")
    use_master_takeover = cfg.get("use_master_takeover", False)
    if use_master_takeover and action_mode != "absolute_pose":
        raise ValueError(
            "use_master_takeover=True requires action_mode='absolute_pose'. "
            "Master takeover poses are absolute dual-arm 14D commands and cannot "
            "be interpreted as relative_pose actions."
        )

    if action_mode == "absolute_pose":
        env = DualAbsolutePoseActionWrapper(env)
    elif action_mode == "relative_pose":
        env = DualRelativePoseActionWrapper(env)
        if cfg.get("use_relative_frame", True):
            env = DualRelativeFrame(env)
    else:
        raise ValueError(
            f"Unsupported action_mode={action_mode!r}. "
            "Expected one of {'absolute_pose', 'relative_pose'}."
        )

    if use_master_takeover:
        env = MasterTakeoverIntervention(env, config=cfg.get("master_takeover", None))

    env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))
    env = DualQuat2EulerWrapper(env)
    return env
