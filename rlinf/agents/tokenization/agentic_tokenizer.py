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

"""Agentic tokenizer: model-level tokenization for agent loops."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rlinf.models.tokenization.hf import hf_tokenizer


class AgenticTokenizer:
    """Base class: model-level tokenization for agent loops.

    Subclasses customize model-specific behavior (reasoning parser name,
    native tool-call boundary merge, reasoning offset). The base provides the
    plain-text tool-response path (mode="text") used by search-r1 for both
    qwen and glm, bit-identical to the pre-refactor behavior.
    """

    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __init__(
        self,
        model_path: str,
        chat_template_kwargs: dict[str, Any] | None = None,
        trust_remote_code: bool = False,
    ):
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.chat_template_kwargs: dict[str, Any] = chat_template_kwargs or {}

    def require_token_id(self, token_str: str) -> int:
        """Convert token string to id, raising if not in vocab."""
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        if tid is None or tid == self.tokenizer.unk_token_id:
            raise ValueError(f"{token_str!r} not in {type(self).__name__} vocab")
        return tid

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict], **kwargs: Any):
        merged = {**self.chat_template_kwargs, **kwargs}
        return self.tokenizer.apply_chat_template(messages, **merged)

    def get_tool_response_ids(
        self,
        prefix_ids: list[int],
        tool_messages: list[dict],
        mode: str = "text",
    ) -> list[int]:
        """Return the full merged prompt ids for the next turn.

        mode="text" encodes the tool response as plain text (search-r1 style,
        model-agnostic). mode="native" uses the model-native chat-template
        boundary merge (subclass-specific).
        """
        if mode == "native":
            return self.build_native_tool_response(prefix_ids, tool_messages)
        return list(prefix_ids) + self.build_text_tool_response(tool_messages)

    def build_text_tool_response(self, tool_messages: list[dict]) -> list[int]:
        content = "".join(m["content"] for m in tool_messages)
        return self.encode(content, add_special_tokens=False)

    def build_native_tool_response(
        self, prefix_ids: list[int], tool_messages: list[dict]
    ) -> list[int]:
        raise NotImplementedError(
            "Native tool-call boundary merge requires a model-specific subclass"
        )

    def content_start_offset(self, response_ids: list[int]) -> int:
        """Index in response_ids where model content begins (after reasoning).

        Base has no reasoning, returns 0.
        """
        return 0

    def truncate_after_stop(
        self, response_ids: list[int], stop_str: str, start: int
    ) -> tuple[list[int], bool]:
        """Truncate response_ids at the first token where stop_str appears in
        the content region (response_ids[start:]).

        Returns (truncated_ids, matched). Decoded text length is monotonic in
        token count, so binary search finds the smallest k such that
        decode(response_ids[start:k]) contains stop_str. No re-encode, so
        response_ids stays a prefix of the original output_ids.
        """
        content_text = self.decode(response_ids[start:])
        pos = content_text.find(stop_str)
        if pos == -1:
            return list(response_ids), False
        target_end = pos + len(stop_str)
        lo, hi = start + 1, len(response_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.decode(response_ids[start:mid])) >= target_end:
                hi = mid
            else:
                lo = mid + 1
        result = response_ids[:lo]
        assert stop_str in self.decode(result), (
            f"truncate_after_stop landed before {stop_str!r}"
        )
        return result, True


class QwenAgenticTokenizer(AgenticTokenizer):
    """Qwen2.5/3 reference subclass. Inherits base text-mode path."""

    reasoning_parser = "qwen3"
    tool_call_parser = "qwen25"


class GLMAgenticTokenizer(AgenticTokenizer):
    """GLM-4.7-Flash subclass.

    Text mode (search-r1): plain-text encode, identical to qwen.
    Native mode: observation-token boundary merge via incremental chat-template
    render.
    """

    reasoning_parser = "glm45"
    tool_call_parser = "glm47"
    stop_token_ids_extra: tuple[str, ...] = (
        "<|observation|>",
        "<|user|>",
    )

    def __init__(
        self,
        model_path: str,
        chat_template_kwargs: dict[str, Any] | None = None,
        trust_remote_code: bool = True,
    ):
        # clear_thinking=False keeps reasoning blocks in the token stream.
        merged = {"clear_thinking": False, **(chat_template_kwargs or {})}
        super().__init__(model_path, merged, trust_remote_code=trust_remote_code)
        # Resolve special-token ids at load time; fail loudly if missing.
        self.observation_token_id = self.require_token_id("<|observation|>")
        self.user_token_id = self.require_token_id("<|user|>")
        self.ambiguous_token_ids = {self.observation_token_id, self.user_token_id}
        self.reasoning_end_token_id = self.require_token_id("</think>")

    def render_incremental(self, tool_messages: list[dict]) -> list[int]:
        """Incremental chat-template render of tool messages + next assistant
        prompt, returned as token-id diff."""
        without_tool_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "dummy"},
                {"role": "assistant", "content": " "},
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        with_tool_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "dummy"},
                {"role": "assistant", "content": " "},
                *tool_messages,
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        return list(with_tool_ids[len(without_tool_ids) :])

    def build_native_tool_response(
        self, prefix_ids: list[int], tool_messages: list[dict]
    ) -> list[int]:
        incremental = self.render_incremental(tool_messages)
        prefix = list(prefix_ids)
        # Strip trailing ambiguous stop==start marker to avoid duplication.
        if prefix and prefix[-1] in self.ambiguous_token_ids:
            prefix = prefix[:-1]
        return prefix + incremental

    def content_start_offset(self, response_ids: list[int]) -> int:
        """Content begins right after the last reasoning-end token; 0 if
        absent. Uses last occurrence because a thinking model may emit the
        closing think-token as a literal string inside reasoning."""
        try:
            idx = len(response_ids) - response_ids[::-1].index(
                self.reasoning_end_token_id
            )
            return idx
        except ValueError:
            return 0


class AgenticTokenizerType(str, Enum):
    """Registry keys for get_agentic_tokenizer."""

    DEFAULT = "default"
    QWEN = "qwen"
    GLM47 = "glm47"


AGENTIC_TOKENIZER_REGISTRY: dict[AgenticTokenizerType, type[AgenticTokenizer]] = {
    AgenticTokenizerType.DEFAULT: AgenticTokenizer,
    AgenticTokenizerType.QWEN: QwenAgenticTokenizer,
    AgenticTokenizerType.GLM47: GLMAgenticTokenizer,
}


def get_agentic_tokenizer(
    model_path: str,
    tokenizer_type: str = "default",
    chat_template_kwargs: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> AgenticTokenizer:
    """Build an AgenticTokenizer for the given model."""
    try:
        key = AgenticTokenizerType(tokenizer_type)
    except ValueError as e:
        raise ValueError(
            f"Unknown agentic_tokenizer type: {tokenizer_type!r}. "
            f"Valid: {[t.value for t in AgenticTokenizerType]}"
        ) from e
    cls = AGENTIC_TOKENIZER_REGISTRY[key]
    return cls(model_path, chat_template_kwargs, trust_remote_code)


__all__ = [
    "AgenticTokenizer",
    "QwenAgenticTokenizer",
    "GLMAgenticTokenizer",
    "AgenticTokenizerType",
    "get_agentic_tokenizer",
]
