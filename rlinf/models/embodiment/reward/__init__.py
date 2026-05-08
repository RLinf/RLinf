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

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [  # noqa: F822
    "BaseRewardModel",
    "BaseImageRewardModel",
    "ResNetRewardModel",
    "VLMRewardModel",
    "HistoryVLMRewardModel",
    "HistoryVLMSGLangRewardModel",
    "get_reward_model_class",
]

_LAZY_CLASS_PATHS = {
    "BaseRewardModel": (
        "rlinf.models.embodiment.reward.base_reward_model",
        "BaseRewardModel",
    ),
    "BaseImageRewardModel": (
        "rlinf.models.embodiment.reward.base_image_reward_model",
        "BaseImageRewardModel",
    ),
    "ResNetRewardModel": (
        "rlinf.models.embodiment.reward.resnet_reward_model",
        "ResNetRewardModel",
    ),
    "VLMRewardModel": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "VLMRewardModel",
    ),
    "HistoryVLMRewardModel": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "HistoryVLMRewardModel",
    ),
    "HistoryVLMSGLangRewardModel": (
        "rlinf.models.embodiment.reward.vlm_sglang_reward_model",
        "HistoryVLMSGLangRewardModel",
    ),
}

reward_model_registry = {
    "resnet": _LAZY_CLASS_PATHS["ResNetRewardModel"],
    "vlm": _LAZY_CLASS_PATHS["VLMRewardModel"],
    "history_vlm": _LAZY_CLASS_PATHS["HistoryVLMRewardModel"],
    "history_vlm_sglang": _LAZY_CLASS_PATHS["HistoryVLMSGLangRewardModel"],
}


def _load_class(class_path: tuple[str, str]) -> type:
    module_name, class_name = class_path
    module = import_module(module_name)
    return getattr(module, class_name)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_CLASS_PATHS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    loaded_class = _load_class(_LAZY_CLASS_PATHS[name])
    globals()[name] = loaded_class
    return loaded_class


def get_reward_model_class(reward_model_type: str):
    if reward_model_type not in reward_model_registry:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    return _load_class(reward_model_registry[reward_model_type])
