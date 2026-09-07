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

"""Registry for embodied environment step-reward functions.

A step-reward function converts the raw simulator reward and the per-step
``info`` dict emitted by an environment into the scalar reward used for
training (e.g. GRPO/PPO rollouts). Registering a function here lets users
select it from YAML via ``env.<split>.reward_mode`` without modifying the
environment code.

A reward function must accept keyword arguments::

    fn(*, raw_reward: torch.Tensor, info: dict, cfg) -> torch.Tensor

where ``raw_reward`` is the ``[num_envs]`` reward returned by the underlying
simulator, ``info`` is the step info dict (containing tensors such as
``success`` or task-specific signals like ``is_src_obj_grasped``), and
``cfg`` is the environment config node (so functions can read options such as
``reward_components``). It must return a float tensor of shape
``[num_envs]``.
"""

import importlib
from typing import Callable, Optional

ENV_REWARD_REGISTRY: dict[str, Callable] = {}


def register_env_reward(name: str) -> Callable:
    """Decorator registering a step-reward function under ``name``.

    The name is case-normalized to lowercase and is the value users put in
    ``env.<split>.reward_mode``.
    """

    def decorator(fn: Callable) -> Callable:
        key = name.lower()
        assert key not in ENV_REWARD_REGISTRY, f"Env reward '{key}' already registered"
        ENV_REWARD_REGISTRY[key] = fn
        return fn

    return decorator


def get_env_reward_fn(name: str, module: Optional[str] = None) -> Callable:
    """Retrieve a registered step-reward function by name.

    Args:
        name: Registered reward name (matched case-insensitively).
        module: Optional dotted module path (e.g. ``my_pkg.my_rewards``) that
            is imported first so its ``@register_env_reward`` decorators run.
            This lets users ship custom rewards outside the RLinf tree and
            point to them from YAML via ``env.<split>.reward_fn_module``.

    Returns:
        The registered reward callable.
    """
    if module:
        importlib.import_module(module)
    key = name.lower()
    if key not in ENV_REWARD_REGISTRY:
        raise ValueError(
            f"Env reward '{name}' not registered. "
            f"Available: {sorted(ENV_REWARD_REGISTRY.keys())}. "
            "Register your own with rlinf.envs.rewards.register_env_reward, "
            "and set env.<split>.reward_fn_module to its module path if it "
            "lives outside RLinf."
        )
    return ENV_REWARD_REGISTRY[key]


def list_env_rewards() -> list[str]:
    """Return the names of all registered step-reward functions."""
    return sorted(ENV_REWARD_REGISTRY.keys())
