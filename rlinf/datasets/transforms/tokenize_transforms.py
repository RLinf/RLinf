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

"""
Tokenization transforms for SFT datasets.

This module provides transforms for tokenizing prompts with guidance tokens
for Classifier-Free Guidance (CFG) models.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGuidance:
    """Custom transform: tokenize both original prompt and guidance prompts.

    In SFT mode, generates positive and negative guidance prompts for CFG_MODEL:
    - positive_guidance_prompt: "[POSITIVE][POSITIVE]\\nTask: {prompt}"
    - negative_guidance_prompt: "[NEGATIVE][NEGATIVE]\\nTask: {prompt}"
    """

    tokenizer: Any  # openpi.models.tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        # Tokenize original prompt
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        # Generate guidance prompts
        positive_prompt = f"[POSITIVE][POSITIVE]\nTask: {prompt}"
        negative_prompt = f"[NEGATIVE][NEGATIVE]\nTask: {prompt}"

        # Tokenize guidance prompts
        positive_tokens, positive_masks = self.tokenizer.tokenize(
            positive_prompt, state
        )
        negative_tokens, negative_masks = self.tokenizer.tokenize(
            negative_prompt, state
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "tokenized_positive_guidance_prompt": positive_tokens,
            "tokenized_positive_guidance_prompt_mask": positive_masks,
            "tokenized_negative_guidance_prompt": negative_tokens,
            "tokenized_negative_guidance_prompt_mask": negative_masks,
        }
