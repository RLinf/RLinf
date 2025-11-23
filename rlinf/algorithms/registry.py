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

from functools import wraps
from typing import Callable, Optional

import torch

from rlinf.algorithms.utils import (
    calculate_scores,
    postprocess_embodied_advantages_outputs,
    postprocess_loss_metric,
    postprocess_reasoning_advantages_outputs,
    preprocess_embodied_advantages_inputs,
    preprocess_loss_inputs,
    preprocess_reasoning_advantages_inputs,
)

ADV_REGISTRY: dict[str, Callable] = {}


def register_advantage(name: str):
    """Decorator to register advantage & returns function."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        ADV_REGISTRY[name.lower()] = wrapper
        return wrapper

    return decorator


def get_adv_and_returns(name: str) -> Callable:
    """Retrieve registered advantage function by name."""
    if name.lower() not in ADV_REGISTRY:
        raise ValueError(
            f"Advantage '{name}' not registered. Available: {list(ADV_REGISTRY.keys())}"
        )
    return ADV_REGISTRY[name.lower()]


LOSS_REGISTRY: dict[str, Callable] = {}


def register_policy_loss(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        LOSS_REGISTRY[name.lower()] = wrapper
        return wrapper

    return decorator


def get_policy_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not registered")
    return LOSS_REGISTRY[name]


def policy_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Unified actor loss entry.
    """
    loss_type = kwargs["loss_type"]
    loss_fn = get_policy_loss(loss_type)

    task_type = kwargs["task_type"]
    if task_type == "embodied":
        kwargs = preprocess_loss_inputs(**kwargs)

    loss, metrics_data = loss_fn(**kwargs)

    if task_type == "embodied":
        metrics_data = postprocess_loss_metric(metrics_data)
    return loss, metrics_data


def calculate_adv_and_returns(**kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified entry for advantage + return computation.
    Accepts variable keyword arguments, preprocesses them, then dispatches
    to specific algorithm via registry.
    """
    adv_type = kwargs["adv_type"]
    fn = get_adv_and_returns(adv_type)

    task_type = kwargs["task_type"]
    if task_type == "embodied":
        kwargs = preprocess_embodied_advantages_inputs(**kwargs)
        if adv_type != "gae":
            kwargs = calculate_scores(**kwargs)
        advantages, returns = fn(**kwargs)
        res = postprocess_embodied_advantages_outputs(
            advantages=advantages, returns=returns, **kwargs
        )
    else:
        # reasoning tasks
        kwargs = preprocess_reasoning_advantages_inputs(**kwargs)
        advantages, returns = fn(**kwargs)
        res = postprocess_reasoning_advantages_outputs(advantages, returns)
    return res
