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

import json
import re
from typing import Any

import torch

REWARD_PARSER_REGISTRY: dict[str, type] = {}


def register_reward_parser(name: str):
    def decorator(cls: type):
        REWARD_PARSER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_reward_parser(name: str) -> type:
    name_lower = name.lower()
    if name_lower not in REWARD_PARSER_REGISTRY:
        raise ValueError(f"RewardParser '{name}' not registered")
    return REWARD_PARSER_REGISTRY[name_lower]


@register_reward_parser("base_reward_parser")
class BaseRewardParser:
    def parse_rewards(
        self, outputs: list[str]
    ) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        raise NotImplementedError


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text_chunk = text[start : end + 1]
        try:
            obj = json.loads(json_text_chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _parse_robochallenge_output(
    text: str, use_confidence: bool, reward_weights: dict[str, float] | None = None
) -> float | None:
    obj = _extract_json_object(text)
    if obj is not None:
        judgement = obj.get("judgement", None)
        if judgement is None:
            for key in ("answer", "label"):
                judgement = obj.get(key, None)
                if judgement is not None:
                    break

        if judgement is not None:
            if reward_weights is None:
                reward_weights = {}

            j = str(judgement).strip().lower()
            if j in ("positive", "negative", "unchanged"):
                if not use_confidence:
                    if j == "positive":
                        return reward_weights.get("positive", 1.0)
                    elif j == "negative":
                        return reward_weights.get("negative", -1.0)
                    elif j == "unchanged":
                        return reward_weights.get("unchanged", 0.0)

                confidence = obj.get("confidence", None)
                if confidence is None:
                    return None
                try:
                    conf = float(confidence)
                except Exception:
                    return None
                return conf if j == "positive" else 1.0 - conf

    if use_confidence:
        return None

    matches = re.findall(
        r"\b(positive|negative|unchanged)\b", str(text).strip().lower()
    )
    if not matches:
        return None
    judgement = matches[-1]

    if reward_weights is None:
        reward_weights = {}

    if judgement == "positive":
        return reward_weights.get("positive", 1.0)
    if judgement == "negative":
        return reward_weights.get("negative", -1.0)
    if judgement == "unchanged":
        return reward_weights.get("unchanged", 0.0)

    return None


def _parse_ternary_output(text: str) -> tuple[float | None, str | None]:
    obj = _extract_json_object(text)
    if obj is not None:
        judgement = obj.get("judgement", None)
        if judgement is None:
            for key in ("answer", "label"):
                judgement = obj.get(key, None)
                if judgement is not None:
                    break
        if judgement is not None:
            label = str(judgement).strip().lower()
            if label == "positive":
                return 1.0, "positive"
            if label == "unchanged":
                return 0.0, "unchanged"
            if label == "negative":
                return -1.0, "negative"

    matches = re.findall(
        r"\b(positive|negative|unchanged)\b", str(text).strip().lower()
    )
    if not matches:
        return None, None
    label = matches[-1]
    if label == "positive":
        return 1.0, "positive"
    if label == "unchanged":
        return 0.0, "unchanged"
    if label == "negative":
        return -1.0, "negative"
    return None, None


@register_reward_parser("robochallenge_reward_parser")
class RobochallengeRewardParser(BaseRewardParser):
    def __init__(
        self,
        positive_reward_weight: float = 1.0,
        unchanged_reward_weight: float = 0.0,
        negative_reward_weight: float = -0.5,
        reward_weights: dict[str, float] | None = None,
        positive_reward: float | None = None,
        unchanged_reward: float | None = None,
        negative_reward: float | None = None,
    ) -> None:
        if reward_weights is not None:
            positive_reward_weight = reward_weights.get(
                "positive_reward_weight",
                reward_weights.get("positive_reward", positive_reward_weight),
            )
            unchanged_reward_weight = reward_weights.get(
                "unchanged_reward_weight",
                reward_weights.get("unchanged_reward", unchanged_reward_weight),
            )
            negative_reward_weight = reward_weights.get(
                "negative_reward_weight",
                reward_weights.get("negative_reward", negative_reward_weight),
            )
        if positive_reward is not None:
            positive_reward_weight = positive_reward
        if unchanged_reward is not None:
            unchanged_reward_weight = unchanged_reward
        if negative_reward is not None:
            negative_reward_weight = negative_reward
        self.reward_weights = {
            "positive": positive_reward_weight,
            "unchanged": unchanged_reward_weight,
            "negative": negative_reward_weight,
        }

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(
                output, use_confidence=False, reward_weights=self.reward_weights
            )
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(-1.0, 1.0)
        return rewards


@register_reward_parser("confidence_robochallenge_reward_parser")
class ConfidenceRoboChallengeRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(output, use_confidence=True)
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
        return rewards


@register_reward_parser("simple_ternary_reward_parser")
class SimpleTernaryRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward, _ = _parse_ternary_output(output)
            rewards.append(0.0 if reward is None else float(reward))
        return torch.tensor(rewards, dtype=torch.float32).clamp(-1.0, 1.0)


@register_reward_parser("weighted_ternary_reward_parser")
class WeightedTernaryRewardParser(BaseRewardParser):
    def __init__(
        self,
        positive_reward: float = 1.0,
        unchanged_reward: float = 0.0,
        negative_reward: float = -0.5,
    ) -> None:
        self.positive_reward = float(positive_reward)
        self.unchanged_reward = float(unchanged_reward)
        self.negative_reward = float(negative_reward)

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            _, label = _parse_ternary_output(output)
            if label == "positive":
                rewards.append(self.positive_reward)
            elif label == "negative":
                rewards.append(self.negative_reward)
            elif label == "unchanged":
                rewards.append(self.unchanged_reward)
            else:
                rewards.append(self.unchanged_reward)
        return torch.tensor(rewards, dtype=torch.float32)
