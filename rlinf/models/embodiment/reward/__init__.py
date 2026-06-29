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

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

__all__ = [
    "BaseRewardModel",
    "ResNetRewardModel",
    "VLMRewardModel",
    "HistoryVLMRewardModel",
    "HistoryVLMSGLangRewardModel",
    "get_reward_model_class",
    "resolve_reward_model_backend",
]

_REWARD_MODEL_CLASS_SPECS = {
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
    "resnet": "ResNetRewardModel",
    "vlm": "VLMRewardModel",
    "history_vlm": "HistoryVLMRewardModel",
}

_HISTORY_VLM_MODEL_TYPE = "history_vlm"
_HISTORY_VLM_TRANSFORMERS_BACKEND = "hf"
_HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES = {"hf", "transformers"}
_HISTORY_VLM_SUPPORTED_BACKENDS = _HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES | {"sglang"}


def _load_reward_model_class(class_name: str):
    module_name, attr_name = _REWARD_MODEL_CLASS_SPECS[class_name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __getattr__(name: str):
    if name in _REWARD_MODEL_CLASS_SPECS:
        return _load_reward_model_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_backend(inference_backend: str | None) -> str | None:
    if inference_backend is None or inference_backend == "":
        return None
    backend = str(inference_backend).lower()
    if backend in _HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES:
        return _HISTORY_VLM_TRANSFORMERS_BACKEND
    return backend


def resolve_reward_model_backend(
    reward_model_type: str,
    inference_backend: str | None = None,
) -> tuple[str, str | None]:
    """Resolve reward model type and inference backend."""
    backend = _normalize_backend(inference_backend)

    if reward_model_type not in reward_model_registry:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    if reward_model_type != _HISTORY_VLM_MODEL_TYPE:
        if backend is not None:
            raise ValueError(
                "reward.model.inference_backend is only supported for "
                "reward.model.model_type='history_vlm'."
            )
        return reward_model_type, None

    if backend is not None and backend not in _HISTORY_VLM_SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported reward.model.inference_backend for history_vlm: "
            f"{inference_backend!r}. Supported backend values are 'hf' "
            "(alias 'transformers'), 'sglang', or unset."
        )
    return reward_model_type, backend


def get_reward_model_class(
    reward_model_type: str,
    inference_backend: str | None = None,
):
    reward_model_type, inference_backend = resolve_reward_model_backend(
        reward_model_type,
        inference_backend,
    )

    if reward_model_type == _HISTORY_VLM_MODEL_TYPE and inference_backend == "sglang":
        return _load_reward_model_class("HistoryVLMSGLangRewardModel")

    return _load_reward_model_class(reward_model_registry[reward_model_type])
