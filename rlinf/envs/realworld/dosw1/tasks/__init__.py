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

from rlinf.envs.realworld.common.wrappers import LeaderFollowerKeyboardIntervention
from rlinf.envs.realworld.dosw1.tasks.pick import PickEnv as PickEnv


def _maybe_apply_keyboard_intervention(
    env: gym.Env, env_cfg: Mapping[str, Any]
) -> gym.Env:
    if (
        env_cfg.get("keyboard_intervention_wrapper", False)
        and getattr(env.config, "enable_human_in_loop", False)
        and not getattr(env.config, "is_dummy", False)
    ):
        env = LeaderFollowerKeyboardIntervention(env)
    return env


def create_dosw1_pick_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = PickEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _maybe_apply_keyboard_intervention(env, env_cfg)


register(
    id="DOSW1PickEnv-v1",
    entry_point="rlinf.envs.realworld.dosw1.tasks:create_dosw1_pick_env",
)

__all__ = ["PickEnv", "create_dosw1_pick_env"]
