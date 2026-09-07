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

"""Built-in step-reward functions for embodied environments.

Besides the trivial ``raw`` and ``only_success`` modes, this module provides
``weighted_components``: a config-driven dense/hierarchical reward that sums
weighted signals from the environment's ``info`` dict, e.g.::

    env:
      train:
        reward_mode: weighted_components
        reward_components:
          is_src_obj_grasped: 0.3          # contact / grasp detection
          gripper_carrot_dist:             # dense pose-alignment shaping
            weight: 0.5
            transform: one_minus_tanh
            scale: 5.0
          success:                         # task completion, gated on grasp
            weight: 1.0
            requires: [is_src_obj_grasped]

Each component is either ``name: weight`` or ``name: {weight, transform,
scale, requires}``. ``name`` is looked up in ``info`` (the special name
``raw`` refers to the simulator's raw reward). ``transform`` reshapes the
signal before weighting — useful for turning distances into dense rewards —
and ``requires`` gates the component on boolean stage signals, enabling
hierarchical (staged) reward structures.
"""

from typing import Any

import torch

from rlinf.envs.rewards.registry import register_env_reward

_TRANSFORMS = {
    # Use the signal as-is (booleans become 0/1).
    "none": lambda v, scale: v,
    # Negated signal: reward decreasing quantities such as distances.
    "neg": lambda v, scale: -scale * v,
    # exp(-scale * v): dense shaping in (0, 1] that peaks at v == 0.
    "neg_exp": lambda v, scale: torch.exp(-scale * v),
    # 1 - tanh(scale * v): dense shaping in (0, 1] that peaks at v == 0.
    "one_minus_tanh": lambda v, scale: 1.0 - torch.tanh(scale * v),
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _component_value(name: str, raw_reward: torch.Tensor, info: dict) -> torch.Tensor:
    if name == "raw":
        value = raw_reward
    elif name in info:
        value = info[name]
    else:
        tensor_keys = sorted(k for k, v in info.items() if isinstance(v, torch.Tensor))
        raise KeyError(
            f"Reward component '{name}' not found in env info. "
            f"Available info tensors: {tensor_keys} (plus the special "
            "component 'raw' for the simulator's raw reward)."
        )
    return torch.as_tensor(value, device=raw_reward.device).to(torch.float32)


@register_env_reward("raw")
def raw_reward_fn(
    *, raw_reward: torch.Tensor, info: dict, cfg: Any = None
) -> torch.Tensor:
    """Pass through the simulator's raw (typically dense) reward."""
    return torch.as_tensor(raw_reward).to(torch.float32)


@register_env_reward("only_success")
def only_success_reward_fn(
    *, raw_reward: torch.Tensor, info: dict, cfg: Any = None
) -> torch.Tensor:
    """Sparse reward: 1.0 on task success, 0.0 otherwise."""
    return _component_value("success", raw_reward, info)


@register_env_reward("weighted_components")
def weighted_components_reward_fn(
    *, raw_reward: torch.Tensor, info: dict, cfg: Any = None
) -> torch.Tensor:
    """Config-driven weighted sum of ``info`` signals (see module docstring).

    Reads ``reward_components`` from ``cfg``: a mapping from component name to
    either a scalar weight or a dict with keys ``weight`` (required),
    ``transform`` (one of ``none``/``neg``/``neg_exp``/``one_minus_tanh``,
    default ``none``), ``scale`` (float, default 1.0), and ``requires`` (list
    of boolean ``info`` keys that must all hold for the component to count).
    """
    components = _cfg_get(cfg, "reward_components")
    if not components:
        raise ValueError(
            "reward_mode 'weighted_components' requires a non-empty "
            "env.<split>.reward_components mapping in the config."
        )

    reward = torch.zeros_like(torch.as_tensor(raw_reward), dtype=torch.float32)
    for name, spec in components.items():
        if isinstance(spec, (int, float)):
            weight, transform, scale, requires = float(spec), "none", 1.0, []
        else:
            weight = float(_cfg_get(spec, "weight"))
            transform = str(_cfg_get(spec, "transform", "none"))
            scale = float(_cfg_get(spec, "scale", 1.0))
            requires = list(_cfg_get(spec, "requires", []) or [])

        if transform not in _TRANSFORMS:
            raise ValueError(
                f"Unknown transform '{transform}' for reward component "
                f"'{name}'. Available: {sorted(_TRANSFORMS.keys())}"
            )

        value = _TRANSFORMS[transform](_component_value(name, raw_reward, info), scale)
        for req in requires:
            value = value * _component_value(req, raw_reward, info)
        reward = reward + weight * value
    return reward
