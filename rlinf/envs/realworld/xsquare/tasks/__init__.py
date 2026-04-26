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
    apply_dual_pose_action_wrappers,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
)
from rlinf.envs.realworld.xsquare.tasks.deploy_env import (
    Turtle2DeployEnv as Turtle2DeployEnv,
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
    override_cfg.setdefault("action_mode", env_cfg.get("action_mode", "relative_pose"))
    env = Turtle2DeployEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_dual_pose_action_wrappers(env, env_cfg)


register(
    id="ButtonEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_button_env",
)

register(
    id="Turtle2DeployEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_turtle2_deploy_env",
)
