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

from typing import Callable, Dict, Optional, Tuple

import torch

ADV_REGISTRY: Dict[str, Callable] = {}


def register_advantage(name: str):
    """Decorator to register advantage & returns function."""

    def decorator(func: Callable):
        ADV_REGISTRY[name.lower()] = func
        return func

    return decorator


def get_adv_and_returns(name: str) -> Callable:
    """Retrieve registered advantage function by name."""
    if name.lower() not in ADV_REGISTRY:
        raise ValueError(
            f"Advantage '{name}' not registered. Available: {list(ADV_REGISTRY.keys())}"
        )
    return ADV_REGISTRY[name.lower()]


LOSS_REGISTRY: Dict[str, Callable] = {}


def register_policy_loss(name: str):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def get_policy_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not registered")
    return LOSS_REGISTRY[name]


def actor_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Unified actor loss entry.
    """
    loss_type = kwargs["loss_type"]
    loss_fn = get_policy_loss(loss_type)
    return loss_fn(**kwargs)


def calculate_adv_and_returns(**kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified entry for advantage + return computation.
    Accepts variable keyword arguments, preprocesses them, then dispatches
    to specific algorithm via registry.
    """
    adv_type = kwargs["adv_type"]
    fn = get_adv_and_returns(adv_type)
    return fn(**kwargs)
