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

"""Registry for embodied action policies, mirroring ``rlinf/algorithms/registry``.

A policy is a class (subclass of :class:`EmbodiedActionPolicy`) registered under
its ``model_type``. The general sglang rollout worker looks up the policy by
``rollout.model.model_type`` and delegates action inference to it.
"""

from typing import Callable, TypeVar

T = TypeVar("T")

ACTION_POLICY_REGISTRY: dict[str, Callable[..., object]] = {}


def register_action_policy(name: str):
    """Decorator registering an action policy class under a model type name.

    Args:
        name: model_type string (case-insensitive); must match
            ``rollout.model.model_type`` for the worker to select this policy.
    """

    def decorator(cls):
        ACTION_POLICY_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_action_policy_cls(name: str):
    """Look up a registered action policy class by model type name."""
    key = name.lower()
    if key not in ACTION_POLICY_REGISTRY:
        raise ValueError(
            f"Action policy '{name}' not registered. "
            f"Available: {list(ACTION_POLICY_REGISTRY.keys())}"
        )
    return ACTION_POLICY_REGISTRY[key]
