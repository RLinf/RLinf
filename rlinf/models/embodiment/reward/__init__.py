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

"""Reward models for embodied RL."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        BufferedVLMRewardModel,
        ShapedVLMRewardModel,
        VLMRewardModel,
    )

__all__ = [
    "BaseRewardModel",
    "ResNetRewardModel",
    "VLMRewardModel",
    "BufferedVLMRewardModel",
    "ShapedVLMRewardModel",
]

_REWARD_MODEL_IMPORTS = {
    "resnet": (
        "rlinf.models.embodiment.reward.resnet_reward_model",
        "ResNetRewardModel",
    ),
    "vlm": ("rlinf.models.embodiment.reward.vlm_reward_model", "VLMRewardModel"),
    "buffered_vlm": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "BufferedVLMRewardModel",
    ),
    "shaped_vlm": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "ShapedVLMRewardModel",
    ),
}

_EXPORT_IMPORTS = {
    class_name: (module_name, class_name)
    for module_name, class_name in _REWARD_MODEL_IMPORTS.values()
}


def _load_class(module_name: str, class_name: str) -> Any:
    module = import_module(module_name)
    return getattr(module, class_name)


class _RewardModelRegistry(dict):
    def __getitem__(self, reward_model_type: str) -> Any:
        module_name, class_name = super().__getitem__(reward_model_type)
        return _load_class(module_name, class_name)

    def get(self, reward_model_type: str, default: Any = None) -> Any:
        if reward_model_type not in self:
            return default
        return self[reward_model_type]


reward_model_registry = _RewardModelRegistry(_REWARD_MODEL_IMPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORT_IMPORTS:
        module_name, class_name = _EXPORT_IMPORTS[name]
        model_class = _load_class(module_name, class_name)
        globals()[name] = model_class
        return model_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_reward_model_class(reward_model_type: str) -> Any:
    if reward_model_type not in reward_model_registry:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    return reward_model_registry[reward_model_type]
