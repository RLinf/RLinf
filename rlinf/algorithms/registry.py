from typing import Callable, Dict, Optional, Tuple
import torch
from rlinf.algorithms.utils import preprocess_loss_inputs, preprocess_advantages_inputs

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
        raise ValueError(f"Advantage '{name}' not registered. Available: {list(ADV_REGISTRY.keys())}")
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
    # Step 1: preprocess
    kwargs = preprocess_loss_inputs(**kwargs)

    # Step 2: dispatch
    loss_type = kwargs["loss_type"]
    loss_fn = get_policy_loss(loss_type)
    return loss_fn(**kwargs)


def calculate_adv_and_returns(**kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified entry for advantage + return computation.
    Accepts variable keyword arguments, preprocesses them, then dispatches
    to specific algorithm via registry.
    """
    # Preprocess common cases:
    kwargs = preprocess_advantages_inputs(**kwargs)

    adv_type = kwargs["adv_type"]
    fn = get_adv_and_returns(adv_type)

    return fn(**kwargs)