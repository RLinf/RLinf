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

from ..env_manager import EnvManager
from .tasks.stack_cube import IsaaclabStackCubeEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0": IsaaclabStackCubeEnv,
}


@EnvManager.register_env("isaaclab")
def get_env_cls(env_cfg):
    assert env_cfg.init_params.id in REGISTER_ISAACLAB_ENVS, (
        f"Task type {env_cfg.init_params.id} have not been registered!"
    )
    return REGISTER_ISAACLAB_ENVS[env_cfg.init_params.id]


__all__ = [list(REGISTER_ISAACLAB_ENVS.keys())]
