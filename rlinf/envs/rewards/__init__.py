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

"""Customizable step-reward functions for embodied environments.

See :mod:`rlinf.envs.rewards.registry` for the function contract and
:mod:`rlinf.envs.rewards.builtin` for the built-in dense/hierarchical
``weighted_components`` reward.
"""

from rlinf.envs.rewards import builtin  # noqa: F401  (registers built-ins)
from rlinf.envs.rewards.registry import (
    ENV_REWARD_REGISTRY,
    get_env_reward_fn,
    list_env_rewards,
    register_env_reward,
)

__all__ = [
    "ENV_REWARD_REGISTRY",
    "get_env_reward_fn",
    "list_env_rewards",
    "register_env_reward",
]
