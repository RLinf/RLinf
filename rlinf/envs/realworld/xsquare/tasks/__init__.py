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
    DualQuat2EulerWrapper,
    DualRelativeFrame,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
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
    env_action_mode = env_cfg.get("action_mode", None)
    override_action_mode = override_cfg.get("action_mode", None)
    if (
        env_action_mode is not None
        and override_action_mode is not None
        and env_action_mode != override_action_mode
    ):
        raise ValueError(
            "Turtle2 deploy action_mode is configured in both env config "
            "and override_cfg with different values: "
            f"{env_action_mode!r} != {override_action_mode!r}."
        )
    action_mode = env_action_mode or override_action_mode or "relative_pose"
    if action_mode not in {"relative_pose", "absolute_pose"}:
        raise ValueError(
            f"Unsupported Turtle2 deploy action_mode={action_mode!r}. "
            "Expected one of {'relative_pose', 'absolute_pose'}."
        )

    override_cfg["action_mode"] = action_mode
    use_arm_ids = list(override_cfg.setdefault("use_arm_ids", [0, 1]))
    if use_arm_ids != [0, 1]:
        raise ValueError("Turtle2DeployEnv-v1 only supports use_arm_ids=[0, 1].")
    override_cfg["use_arm_ids"] = use_arm_ids
    override_cfg.setdefault("use_camera_ids", [0, 1, 2])
    override_cfg.setdefault("enforce_gripper_close", False)
    override_cfg.setdefault("enable_task_reward", False)
    override_cfg.setdefault("task_description", env_cfg.get("task_description", ""))
    config = Turtle2RobotConfig(**override_cfg)
    env = Turtle2Env(
        config=config,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    if config.action_mode == "relative_pose" and env_cfg.get(
        "use_relative_frame", True
    ):
        env = DualRelativeFrame(env)
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
