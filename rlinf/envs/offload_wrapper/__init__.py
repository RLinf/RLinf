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


def get_offload_env(env_type: str):
    """Get offload environment by name.

    Args:
        env_type (str): The name of the environment.

    Returns:
        type: The offload environment class.
    """
    from rlinf.envs.offload_wrapper.maniskill_wrapper import (
        ManiskillEnv as ManiskillEnv,
    )

    offload_envs = {
        "maniskill": ManiskillEnv,
    }
    if env_type in offload_envs:
        return offload_envs[env_type]
    else:
        raise ValueError(f"Environment {env_type} does not support offloading.")
