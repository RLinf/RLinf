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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers import (
    apply_keyboard_reward,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.common.wrappers.dual_euler_obs import (
    DualQuat2EulerWrapper,
)
from rlinf.envs.realworld.common.wrappers.dual_relative_frame import (
    DualRelativeFrame,
)
from rlinf.envs.realworld.common.wrappers.keyboard_running_mode_wrapper import (
    KeyboardRunningModeWrapper,
)
from rlinf.envs.realworld.common.wrappers.master_takeover_intervention import (
    MasterTakeoverIntervention,
)
from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
)
from rlinf.envs.realworld.xsquare.turtle2_deploy_utils import (
    resolve_turtle2_deploy_action_mode,
)
from rlinf.envs.realworld.xsquare.turtle2_env import (
    Turtle2Env,
    Turtle2RobotConfig,
)


def create_button_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = ButtonEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_turtle2_deploy_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    override_cfg = dict(override_cfg)
    action_mode = resolve_turtle2_deploy_action_mode(override_cfg)
    override_cfg["action_mode"] = action_mode
    use_master_takeover = bool(env_cfg.get("use_master_takeover", False))
    use_arm_ids = list(override_cfg.get("use_arm_ids", [0, 1]))
    if use_arm_ids != [0, 1]:
        raise ValueError("Turtle2DeployEnv-v1 only supports use_arm_ids=[0, 1].")
    override_cfg["use_arm_ids"] = use_arm_ids
    override_cfg.setdefault("use_camera_ids", [0, 1, 2])
    override_cfg.setdefault("enforce_gripper_close", False)
    override_cfg.setdefault("enable_task_reward", False)
    override_cfg.setdefault(
        "task_description", env_cfg.get("task_description", "")
    )
    config = Turtle2RobotConfig(**override_cfg)
    env = Turtle2Env(
        config=config,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    if action_mode == "relative_pose" and env_cfg.get("use_relative_frame", True):
        env = DualRelativeFrame(env)

    if use_master_takeover:
        env = MasterTakeoverIntervention(
            env,
            config=env_cfg.get("master_takeover", None),
        )
        keyboard_running_mode_cfg = env_cfg.get("keyboard_running_mode", None)
        if (
            keyboard_running_mode_cfg is not None
            and keyboard_running_mode_cfg.get("enabled", False)
            and not env.get_wrapper_attr("config").is_dummy
        ):
            env = KeyboardRunningModeWrapper(
                env,
                config=keyboard_running_mode_cfg,
            )

    env = apply_keyboard_reward(env, env_cfg.get("keyboard_reward_wrapper", None))
    env = DualQuat2EulerWrapper(env)
    return env


register(
    id="ButtonEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_button_env",
)

register(
    id="Turtle2DeployEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_turtle2_deploy_env",
)
