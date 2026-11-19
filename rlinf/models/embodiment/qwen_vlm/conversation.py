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

"""Pack one VLM episode with exact incremental token/mask tracking.

PEAR builds multi-turn episodes incrementally (``ContinuousTokenBuilder``) so
earlier assistant/thinking spans are never rewritten by a chat template and
never lost to subsequence re-matching, and so the episode response budget counts
every token after the initial prompt (generated + injected observations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Return the next index of ``needle`` in ``haystack``, or -1."""
    if not needle or start < 0:
        return -1
    limit = len(haystack) - len(needle) + 1
    for index in range(start, limit):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def assistant_positions(sequence: list[int], responses: list[list[int]]) -> list[int]:
    """Return token positions of each assistant response inside ``sequence``.

    Kept for backwards compatibility with legacy subsequence-based callers and
    unit tests. Prefer :class:`IncrementalEpisodeTokens` for live multi-turn
    rollouts.
    """
    positions: list[int] = []
    cursor = 0
    for response in responses:
        found = find_subsequence(sequence, response, cursor)
        if found < 0:
            continue
        positions.extend(range(found, found + len(response)))
        cursor = found + len(response)
    return positions


def render_incremental_user_delta(
    processor: Any,
    user_message: dict[str, Any],
    template_kwargs: dict[str, Any],
) -> tuple[str, str]:
    """Render the exact chat-template suffix for appending one user turn.

    Returns ``(full_delta, without_leading_im_end)`` so callers can avoid
    duplicating the assistant end-of-turn token if the sampler already emitted
    it as part of ``output_ids``.
    """
    dummy_prefix = [
        {"role": "system", "content": "__RLINF_SYS__"},
        {"role": "user", "content": "__RLINF_U0__"},
        {"role": "assistant", "content": "__RLINF_A0__"},
    ]
    prefix_text = processor.apply_chat_template(
        dummy_prefix,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    extended_text = processor.apply_chat_template(
        [*dummy_prefix, user_message],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if extended_text.startswith(prefix_text):
        delta = extended_text[len(prefix_text) :]
    else:
        # Fallback if template alters whitespace before the trailing turn.
        marker = "__RLINF_A0__"
        pos = extended_text.find(marker)
        delta = extended_text[pos + len(marker) :] if pos >= 0 else extended_text

    # Determine whether the assistant stop token sits at the end of prefix_text.
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token = getattr(tokenizer, "eos_token", None) or ""
    if eos_token and prefix_text.rstrip().endswith(eos_token):
        # The stop token already closed the previous assistant turn inside prefix_text.
        return delta, delta
    stripped = delta
    if eos_token and delta.startswith(eos_token):
        stripped = delta[len(eos_token) :]
    return delta, stripped


def _extract_1d_tokens(tensor: Any) -> list[int]:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().reshape(-1).to(dtype=torch.long).tolist()
    return [int(x) for x in tensor]


def _squeeze_leading_batch(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


@dataclass
class IncrementalEpisodeTokens:
    """Exact incremental token, mask, multimodal, and top-K state for one episode.

    Matches PEAR's ``PearAgentLoop`` semantics:
    - ``response_length`` counts every token after the initial prompt
      (generated assistant tokens with mask 1 + injected user/image tokens with
      mask 0).
    - ``response_positions`` records the exact global indices of generated
      assistant tokens without re-tokenizing or stripping earlier ``<think>``
      blocks.
    - When a new observation or generation reaches ``response_budget``, the
      episode marks ``budget_exhausted = True`` cleanly.
    """

    response_budget: int
    input_ids: list[int] = field(default_factory=list)
    initial_prompt_length: int = 0
    response_mask: list[int] = field(default_factory=list)
    response_positions: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    response_topk_ids: list[list[int]] = field(default_factory=list)
    response_topk_logprobs: list[list[float]] = field(default_factory=list)
    pixel_chunks: list[torch.Tensor] = field(default_factory=list)
    grid_chunks: list[torch.Tensor] = field(default_factory=list)
    mm_token_type_ids: list[int] = field(default_factory=list)
    has_mm_token_types: bool = False
    rendered_text: str = ""
    turn_count: int = 0
    budget_exhausted: bool = False

    @property
    def initialized(self) -> bool:
        return self.initial_prompt_length > 0

    @property
    def response_length(self) -> int:
        """Total tokens after the initial prompt (generated + injected obs)."""
        return len(self.response_mask)

    def remaining_budget(self) -> int:
        """Remaining tokens in the episode budget."""
        return max(int(self.response_budget) - self.response_length, 0)

    def start_episode(
        self,
        *,
        prompt_text: str,
        inputs: dict[str, Any],
    ) -> None:
        """Initialize the episode with the tokenized system + first user prompt."""
        ids = _extract_1d_tokens(inputs["input_ids"])
        self.input_ids = list(ids)
        self.initial_prompt_length = len(ids)
        self.response_mask = []
        self.response_positions = []
        self.response_logprobs = []
        self.response_topk_ids = []
        self.response_topk_logprobs = []
        self.pixel_chunks = [_squeeze_leading_batch(inputs["pixel_values"])]
        self.grid_chunks = [_squeeze_leading_batch(inputs["image_grid_thw"])]
        token_types = inputs.get("mm_token_type_ids")
        if torch.is_tensor(token_types):
            self.has_mm_token_types = True
            self.mm_token_type_ids = _extract_1d_tokens(token_types)
        else:
            self.has_mm_token_types = False
            self.mm_token_type_ids = []
        self.rendered_text = prompt_text
        self.turn_count = 0
        self.budget_exhausted = self.remaining_budget() <= 0

    def append_user_observation(
        self,
        *,
        delta_text: str,
        delta_inputs: dict[str, Any],
    ) -> bool:
        """Append one tokenized user observation turn (mask=0).

        Returns ``True`` if the observation fits within ``response_budget`` and
        was committed, or ``False`` if adding it would exhaust the budget (in
        which case nothing is committed and ``budget_exhausted`` is set).
        """
        delta_ids = _extract_1d_tokens(delta_inputs["input_ids"])
        if self.response_length + len(delta_ids) >= self.response_budget:
            self.budget_exhausted = True
            return False
        self.input_ids.extend(delta_ids)
        self.response_mask.extend([0] * len(delta_ids))
        self.pixel_chunks.append(_squeeze_leading_batch(delta_inputs["pixel_values"]))
        self.grid_chunks.append(_squeeze_leading_batch(delta_inputs["image_grid_thw"]))
        if self.has_mm_token_types:
            delta_types = delta_inputs.get("mm_token_type_ids")
            if torch.is_tensor(delta_types):
                types_list = _extract_1d_tokens(delta_types)
                if len(types_list) < len(delta_ids):
                    types_list.extend([0] * (len(delta_ids) - len(types_list)))
                self.mm_token_type_ids.extend(types_list[: len(delta_ids)])
            else:
                self.mm_token_type_ids.extend([0] * len(delta_ids))
        self.rendered_text += delta_text
        return True

    def append_assistant_response(
        self,
        *,
        response_ids: list[int],
        response_text: str,
        token_logprobs: list[float] | None = None,
        topk_ids: list[list[int]] | None = None,
        topk_logprobs: list[list[float]] | None = None,
    ) -> None:
        """Append one generated assistant turn (mask=1) with exact positions."""
        start = len(self.input_ids)
        count = len(response_ids)
        self.input_ids.extend(int(t) for t in response_ids)
        self.response_mask.extend([1] * count)
        self.response_positions.extend(range(start, start + count))
        if token_logprobs is not None and len(token_logprobs) == count:
            self.response_logprobs.extend(float(lp) for lp in token_logprobs)
        else:
            self.response_logprobs.extend([0.0] * count)
        if topk_ids is not None and topk_logprobs is not None:
            if len(topk_ids) != count or len(topk_logprobs) != count:
                raise ValueError(
                    "Top-K head length must match response_ids length, got "
                    f"{len(topk_ids)}/{len(topk_logprobs)} vs {count}."
                )
            self.response_topk_ids.extend(topk_ids)
            self.response_topk_logprobs.extend(topk_logprobs)
        if self.has_mm_token_types:
            self.mm_token_type_ids.extend([0] * count)
        self.rendered_text += response_text
        self.turn_count += 1
        if self.response_length >= self.response_budget:
            self.budget_exhausted = True

    def build_sample(self) -> dict[str, torch.Tensor]:
        """Materialize the packed training sample for the actor forward pass."""
        sequence = torch.tensor(self.input_ids, dtype=torch.long)
        pixels = (
            torch.cat(self.pixel_chunks, dim=0)
            if len(self.pixel_chunks) > 1
            else self.pixel_chunks[0].clone()
        )
        grids = (
            torch.cat(self.grid_chunks, dim=0)
            if len(self.grid_chunks) > 1
            else self.grid_chunks[0].clone()
        )
        if self.response_positions:
            response_index = torch.tensor(self.response_positions, dtype=torch.long)
            response_logprobs = torch.tensor(
                self.response_logprobs, dtype=torch.float32
            )
        else:
            pos = max(int(sequence.shape[0]) - 1, 1)
            response_index = torch.tensor([pos], dtype=torch.long)
            response_logprobs = torch.zeros(1, dtype=torch.float32)

        sample: dict[str, torch.Tensor] = {
            "input_ids": sequence,
            "attention_mask": torch.ones_like(sequence, dtype=torch.long),
            "pixel_values": pixels,
            "image_grid_thw": grids,
            "response_logprobs": response_logprobs,
            "response_index": response_index,
        }
        if self.has_mm_token_types:
            types = list(self.mm_token_type_ids)
            if len(types) < len(self.input_ids):
                types.extend([0] * (len(self.input_ids) - len(types)))
            sample["mm_token_type_ids"] = torch.tensor(
                types[: len(self.input_ids)], dtype=torch.long
            )
        if self.response_topk_ids and len(self.response_topk_ids) == len(
            self.response_positions
        ):
            sample["rollout_topk_ids"] = torch.tensor(
                self.response_topk_ids, dtype=torch.long
            )
            sample["sampler_head_log_probs"] = torch.tensor(
                self.response_topk_logprobs, dtype=torch.float32
            )
        return sample
