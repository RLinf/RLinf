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

"""Reward parsers that turn VLM text outputs into scalar reward tensors.

Parsers are registered by name in ``REWARD_PARSER_REGISTRY`` and looked up via
``get_reward_parser``. Each parser maps a batch of generated strings to a float
reward tensor according to a task-specific labeling contract.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import torch

REWARD_PARSER_REGISTRY: dict[str, type] = {}


def register_reward_parser(name: str) -> Callable[[type], type]:
    """Register a reward-parser class under ``name`` (case-insensitive).

    Args:
        name: Registry key used to look the parser up later.

    Returns:
        A class decorator that records the class and returns it unchanged.
    """

    def decorator(cls: type) -> type:
        REWARD_PARSER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_reward_parser(name: str) -> type:
    """Return the registered reward-parser class for ``name``.

    Args:
        name: Registry key (case-insensitive).

    Returns:
        The registered parser class.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    name_lower = name.lower()
    if name_lower not in REWARD_PARSER_REGISTRY:
        raise ValueError(f"RewardParser '{name}' not registered")
    return REWARD_PARSER_REGISTRY[name_lower]


@register_reward_parser("base_reward_parser")
class BaseRewardParser:
    """Abstract base parser; subclasses map text outputs to a reward tensor."""

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        """Parse a batch of generated strings into a float reward tensor."""
        raise NotImplementedError


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text_chunk = text[start : end + 1]
        try:
            obj = json.loads(json_text_chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _parse_qwentrend_output(text: str) -> str | None:
    valid_labels = {"positive", "negative", "unclear"}
    obj = _extract_json_object(text)
    if obj is not None:
        trend_label = str(obj.get("trend", "")).strip().lower()
        if trend_label in valid_labels:
            return trend_label

    matches = re.findall(r"\b(positive|negative|unclear)\b", str(text).strip().lower())
    return matches[-1] if matches else None


@register_reward_parser("qwentrend_reward_parser")
class QwentrendRewardParser(BaseRewardParser):
    """Map ``positive`` / ``negative`` / ``unclear`` trend labels to rewards."""

    def __init__(
        self,
        positive_reward: float = 1.0,
        negative_reward: float = -0.2,
        unclear_reward: float = 0.0,
        invalid_reward: float = 0.0,
    ) -> None:
        self.positive_reward = float(positive_reward)
        self.negative_reward = float(negative_reward)
        self.unclear_reward = float(unclear_reward)
        self.invalid_reward = float(invalid_reward)

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        """Map each output's trend label to its configured scalar reward."""
        rewards: list[float] = []
        for output in outputs:
            label = _parse_qwentrend_output(output)
            if label == "positive":
                rewards.append(self.positive_reward)
            elif label == "negative":
                rewards.append(self.negative_reward)
            elif label == "unclear":
                rewards.append(self.unclear_reward)
            else:
                rewards.append(self.invalid_reward)
        return torch.tensor(rewards, dtype=torch.float32)


@register_reward_parser("qwentrend_binary_digit_reward_parser")
class QwentrendBinaryDigitRewardParser(BaseRewardParser):
    """Map generated binary success labels to sparse rewards."""

    def __init__(
        self,
        positive_reward: float = 1.0,
        negative_reward: float = 0.0,
        invalid_reward: float = 0.0,
        **ignored_params: object,
    ) -> None:
        # Configs share one ``reward_parser_params`` shape across parsers, so
        # tolerate (and ignore) keys such as ``unclear_reward`` that only apply
        # to the trend parser.
        del ignored_params
        self.positive_reward = float(positive_reward)
        self.negative_reward = float(negative_reward)
        self.invalid_reward = float(invalid_reward)

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        """Map a trailing ``1`` to the success reward and ``0`` to non-success."""
        rewards = []
        for output in outputs:
            labels = re.findall(r"(?<!\d)([01])(?!\d)", str(output).strip())
            if labels and labels[-1] == "1":
                rewards.append(self.positive_reward)
            elif labels and labels[-1] == "0":
                rewards.append(self.negative_reward)
            else:
                rewards.append(self.invalid_reward)
        return torch.tensor(rewards, dtype=torch.float32)
