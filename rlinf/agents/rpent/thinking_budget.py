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

"""Bound one RPent thinking span and identify its deterministic closing token."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class ThinkingBudget:
    """A generated-token budget for an already open thinking span."""

    limit: int
    end_token_id: int
    prompt_tokens: int

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError(
                f"thinking_token_budget must be a nonnegative int: {self.limit!r}"
            )
        if type(self.end_token_id) is not int or self.end_token_id < 0:
            raise ValueError(
                f"Expected a valid thinking end token: {self.end_token_id!r}"
            )
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 1:
            raise ValueError(
                f"Expected a nonempty rendered prompt: {self.prompt_tokens!r}"
            )

    @classmethod
    def from_prompt(
        cls,
        prompt_ids: list[int],
        limit: int,
        start_token_id: int,
        end_token_id: int,
    ) -> ThinkingBudget:
        """Check the current span without counting preserved earlier turns."""
        if start_token_id == end_token_id or start_token_id is None:
            raise ValueError("Thinking start and end must be distinct tokenizer IDs")
        last_marker = next(
            (
                token
                for token in reversed(prompt_ids)
                if token in (start_token_id, end_token_id)
            ),
            None,
        )
        if last_marker != start_token_id:
            raise ValueError(
                "Thinking budget requires a prompt with an open thinking span"
            )
        return cls(limit, end_token_id, len(prompt_ids))

    def custom_params(self) -> dict[str, int]:
        return {
            "rpent_thinking_budget": self.limit,
            "rpent_thinking_end": self.end_token_id,
            "rpent_prompt_tokens": self.prompt_tokens,
        }

    def forced_next_token(self, output_ids: list[int]) -> int | None:
        # The prompt already opens this span; earlier turns never count here.
        if self.end_token_id in output_ids[: self.limit + 1]:
            return None
        return self.end_token_id if len(output_ids) >= self.limit else None

    def response_mask(
        self, output_ids: list[int], forced_positions: list[int]
    ) -> list[int]:
        """Exclude only the engine-recorded deterministic token from training."""
        mask = [1] * len(output_ids)
        expected = []
        if (
            len(output_ids) <= self.limit
            or self.end_token_id in output_ids[: self.limit]
        ):
            if forced_positions != expected:
                raise ValueError(
                    "Engine reported a forced token outside the thinking boundary"
                )
            return mask
        if output_ids[self.limit] != self.end_token_id:
            raise ValueError(
                "Engine exceeded the thinking budget without its closing token"
            )
        expected = [self.limit]
        if forced_positions != expected:
            raise ValueError(
                "Engine did not record the deterministic thinking closing token"
            )
        mask[self.limit] = 0
        return mask


def apply_thinking_budget(
    logits: Tensor, custom_param_list: list[dict[str, Any] | None] | None
) -> Tensor:
    """Apply the request-local budget through SGLang's native processor hook."""
    for index, params in enumerate(custom_param_list or []):
        if not params or "rpent_thinking_budget" not in params:
            continue
        budget = ThinkingBudget(
            params["rpent_thinking_budget"],
            params["rpent_thinking_end"],
            params["rpent_prompt_tokens"],
        )
        request = params["__req__"]
        if len(request.origin_input_ids) != budget.prompt_tokens:
            raise ValueError(
                "Engine prompt length differs from the budget's rendered prompt"
            )
        if request.customized_info is None:
            request.customized_info = {}
        # Native SGLang slices customized_info by output-token send offsets.
        # A sparse position list is therefore truncated during intermediate sends.
        flags = request.customized_info.setdefault("forced_token_mask", [])
        position = len(request.output_ids)
        if len(flags) < position + 1:
            flags.extend([False] * (position + 1 - len(flags)))
        forced = budget.forced_next_token(request.output_ids)
        flags[position] = forced is not None
        if forced is not None:
            logits[index].fill_(float("-inf"))
            logits[index, forced] = 0.0
    return logits


@lru_cache(maxsize=1)
def serialized_thinking_processor() -> str:
    """Use SGLang's serializer only when the experiment enables budgeting."""
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class RPentThinkingBudgetProcessor(CustomLogitProcessor):
        def __call__(self, logits, custom_param_list=None):
            return apply_thinking_budget(logits, custom_param_list)

    return RPentThinkingBudgetProcessor.to_str()
