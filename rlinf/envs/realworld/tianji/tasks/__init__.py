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

"""Gym registration for the Tianji RLT environment."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

from ..tianji_env import TianjiConfig, TianjiEnv


def create_tianji_rlt_env(
    override_cfg: Mapping[str, Any] | None = None,
    worker_info=None,
    hardware_info=None,
    env_idx: int = 0,
    env_cfg=None,
) -> gym.Env:
    """Create one Tianji env; worker/hardware args are accepted for parity."""
    del worker_info, hardware_info, env_idx, env_cfg
    values = dict(override_cfg or {})
    known = {field.name for field in dataclasses.fields(TianjiConfig)}
    return TianjiEnv(
        TianjiConfig(**{key: value for key, value in values.items() if key in known})
    )


register(
    id="TianjiRLTEnv-v1",
    entry_point="rlinf.envs.realworld.tianji.tasks:create_tianji_rlt_env",
)

__all__ = ["create_tianji_rlt_env"]
