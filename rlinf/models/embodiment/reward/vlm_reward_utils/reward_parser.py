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
import logging
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)

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
    def __init__(
        self,
        positive_reward: float = 1.0,
        negative_reward: float = -0.2,
        unclear_reward: float = 0.0,
        invalid_reward: float = 0.0,
        debug_print: bool = False,
        debug_print_every: int = 10,
        debug_sample_texts: int = 2,
    ) -> None:
        self.positive_reward = float(positive_reward)
        self.negative_reward = float(negative_reward)
        self.unclear_reward = float(unclear_reward)
        self.invalid_reward = float(invalid_reward)
        self.debug_print = bool(debug_print)
        self.debug_print_every = max(1, int(debug_print_every))
        self.debug_sample_texts = max(0, int(debug_sample_texts))
        self._call_idx = 0

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        pos_count, neg_count, unclear_count, invalid_count = 0, 0, 0, 0
        invalid_examples: list[str] = []
        for output in outputs:
            label = _parse_qwentrend_output(output)
            if label == "positive":
                rewards.append(self.positive_reward)
                pos_count += 1
            elif label == "negative":
                rewards.append(self.negative_reward)
                neg_count += 1
            elif label == "unclear":
                rewards.append(self.unclear_reward)
                unclear_count += 1
            else:
                rewards.append(self.invalid_reward)
                invalid_count += 1
                if len(invalid_examples) < self.debug_sample_texts:
                    invalid_examples.append(str(output).replace("\n", "\\n")[:220])

        self._call_idx += 1
        if self.debug_print and (
            self._call_idx <= 10 or self._call_idx % self.debug_print_every == 0
        ):
            logger.info(
                "[RMDBG_PARSE] parser=pnu call=%d batch=%d labels={positive:%d, negative:%d, unclear:%d, invalid:%d}",
                self._call_idx,
                len(outputs),
                pos_count,
                neg_count,
                unclear_count,
                invalid_count,
            )
            if invalid_examples:
                logger.info(
                    "[RMDBG_PARSE] invalid_samples=%s",
                    invalid_examples,
                )
        return torch.tensor(rewards, dtype=torch.float32)
