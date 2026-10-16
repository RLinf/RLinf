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

from typing import Any

from rlinf.envs.sim.embodichain.embodichain_env import EmbodiChainEnv
from rlinf.envs.sim.embodichain.stack_cube_env import StackCubeEnv

REGISTER_EMBODICHAIN_ENVS = {"RLinf-StackCube-Expert-v1": StackCubeEnv}


def get_embodichain_env_cls(env_cfg: Any = None) -> type[EmbodiChainEnv]:
    """Select an RLinf wrapper from the native Gym task ID."""
    if env_cfg is None:
        return EmbodiChainEnv
    from embodichain.utils.utility import load_config

    from rlinf.envs.sim.embodichain.embodichain_env import (
        _cfg_get,
        _resolve_gym_config_path,
    )

    configured = _cfg_get(env_cfg, "gym_config_path")
    if not configured:
        return EmbodiChainEnv
    path = str(configured)
    if not path.startswith("embodichain_tasks/"):
        path = str(_resolve_gym_config_path(path))
    config = load_config(path)
    return REGISTER_EMBODICHAIN_ENVS.get(config["id"], EmbodiChainEnv)


__all__ = ["EmbodiChainEnv", "StackCubeEnv", "get_embodichain_env_cls"]
