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

"""Agentic tokenizer: model-family-specific tokenization for agent loops."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()


class AgenticTokenizer:
    """Model-family-specific tokenization on top of a caller-loaded HF tokenizer.

    Wraps the given tokenizer rather than loading another one, and covers only
    what varies across model families: reasoning markers, the bound parser
    names, and tool-response ids (plain text here, native boundary handling in
    subclasses). Generic encode / decode goes to the HF tokenizer directly.
    Agent loops that need this build it themselves (currently search-r1).
    """

    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __init__(
        self,
        tokenizer,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs: dict[str, Any] = chat_template_kwargs or {}
        # None when the model has no single-token reasoning markers.
        self.reasoning_start_token_id, self.reasoning_end_token_id = (
            self._resolve_reasoning_tokens()
        )

    def maybe_token_id(self, token_str: str) -> int | None:
        """Token id for token_str, or None when the model has no such token."""
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        if tid is None or tid == self.tokenizer.unk_token_id:
            return None
        return tid

    def require_token_id(self, token_str: str) -> int:
        """Token id for token_str, raising when the model has no such token.

        Used for tokens a subclass cannot work without, e.g. GLM's
        observation boundary markers.
        """
        tid = self.maybe_token_id(token_str)
        if tid is None:
            raise ValueError(f"{token_str!r} not in {type(self).__name__} vocab")
        return tid

    def _resolve_reasoning_tokens(self) -> tuple[int | None, int | None]:
        """Resolve this family's reasoning markers to token ids.

        Markers differ across families (kimi does not use <think>), so they
        are looked up in sglang's reasoning-parser registry by
        reasoning_parser. A marker that is not a single token resolves to
        None. sglang is imported lazily so paths that do not use it never do.
        """
        if self.reasoning_parser is None:
            return None, None
        try:
            from sglang.srt.parser.reasoning_parser import ReasoningParser
        except ImportError as e:
            raise ImportError(
                "AgenticTokenizer resolves reasoning markers from sglang's "
                "reasoning-parser registry, which moved under sglang.srt.parser "
                "in 0.5.5. Install sglang >= 0.5.12 (the version agentic is "
                "validated with, see requirements/install.sh)."
            ) from e
        detector_cls = ReasoningParser.DetectorMap.get(self.reasoning_parser)
        if detector_cls is None:
            raise ValueError(
                f"reasoning_parser={self.reasoning_parser!r} is not in sglang's "
                f"DetectorMap; valid: {sorted(ReasoningParser.DetectorMap)}"
            )
        detector = detector_cls()
        think_start_id = self.maybe_token_id(detector.think_start_token)
        think_end_id = self.maybe_token_id(detector.think_end_token)
        if think_start_id is None or think_end_id is None:
            missing = [
                name
                for name, tid in (
                    ("think_start", think_start_id),
                    ("think_end", think_end_id),
                )
                if tid is None
            ]
            logger.warning(
                "%s declares reasoning_parser=%r, but its %s marker(s) do not "
                "resolve to a single token in this model's vocab; "
                "content_start_offset will always return 0 (reasoning-content "
                "skip disabled, not an error).",
                type(self).__name__,
                self.reasoning_parser,
                ", ".join(missing),
            )
        return think_start_id, think_end_id

    def apply_chat_template(self, messages: list[dict], **kwargs: Any):
        merged = {**self.chat_template_kwargs, **kwargs}
        return self.tokenizer.apply_chat_template(messages, **merged)

    def append_tool_response(self, tool_messages: list[dict]) -> list[int]:
        """Encode tool responses as plain text (search-r1 style, model-agnostic).

        Returns only the appended ids, the same contract as
        AgentLoopWorker.get_tool_response_ids.
        """
        content = "".join(m["content"] for m in tool_messages)
        return self.tokenizer.encode(content, add_special_tokens=False)

    def render_incremental(self, tool_messages: list[dict]) -> list[int]:
        """Chat-template render of tool_messages plus the next assistant
        generation prompt, as the token-id diff against a dummy context.

        Generic across models whose template renders appended messages as a
        pure suffix; a model whose template re-renders earlier messages must
        override this in its subclass. Goes through self.apply_chat_template
        so the class's registered chat_template_kwargs apply.
        """
        base = [
            {"role": "system", "content": "dummy"},
            {"role": "assistant", "content": " "},
        ]
        without_tool_ids = self.apply_chat_template(
            base, add_generation_prompt=False, tokenize=True, return_dict=False
        )
        with_tool_ids = self.apply_chat_template(
            [*base, *tool_messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        if list(with_tool_ids[: len(without_tool_ids)]) != list(without_tool_ids):
            raise ValueError(
                f"{type(self).__name__}: the chat template re-renders earlier "
                "messages when tool messages are appended, so the incremental "
                "diff is not a clean suffix. Override render_incremental in the "
                "model's subclass."
            )
        return list(with_tool_ids[len(without_tool_ids) :])

    def build_native_tool_response(
        self, prefix_ids: list[int], tool_messages: list[dict]
    ) -> list[int]:
        """Return prefix_ids merged with the chat-template render of
        tool_messages. Returns the full sequence rather than an increment
        because a model-specific boundary fix may rewrite the prefix tail
        (GLM drops a trailing observation/user boundary token)."""
        return list(prefix_ids) + self.render_incremental(tool_messages)

    def content_start_offset(
        self, response_ids: list[int], prompt_ids: list[int] | None = None
    ) -> int:
        """Index where this turn's content begins, after its reasoning block.

        The block is opened either by the generation prompt (GLM, Qwen3.5) or
        by the model at the start of the response (Qwen3); the FIRST closer
        after the opener ends it. A later closer means the model ran past its
        turn, and taking it would move the offset past the real stop string
        and skip truncation. With thinking disabled the offset is 0.
        """
        if self.reasoning_end_token_id is None:
            return 0
        if self._prompt_ends_in_reasoning(prompt_ids):
            search_from = 0
        else:
            if self.reasoning_start_token_id is None:
                return 0
            try:
                open_idx = response_ids.index(self.reasoning_start_token_id)
            except ValueError:
                return 0
            # Only an opener at the very start of the response is this turn's
            # reasoning block; one further in is content, not a block start.
            if self.tokenizer.decode(response_ids[:open_idx]).strip():
                return 0
            search_from = open_idx + 1
        try:
            return response_ids.index(self.reasoning_end_token_id, search_from) + 1
        except ValueError:
            return 0

    def _prompt_ends_in_reasoning(self, prompt_ids: list[int] | None) -> bool:
        """True when the last reasoning marker in the prompt is an opener."""
        if not prompt_ids or self.reasoning_start_token_id is None:
            return False
        for tid in reversed(prompt_ids):
            if tid == self.reasoning_end_token_id:
                return False
            if tid == self.reasoning_start_token_id:
                return True
        return False

    def truncate_after_stop(
        self, response_ids: list[int], stop_str: str, start: int
    ) -> tuple[list[int], bool]:
        """Truncate response_ids right after the first stop_str in the content
        region response_ids[start:]. Returns (truncated_ids, matched).

        Binary-searches the smallest k with stop_str in
        decode(response_ids[start:k]). The result is a prefix of the input, so
        rollout logprobs stay aligned by slicing; it ends on a token boundary,
        so its last token may carry characters after stop_str (e.g. "><").
        """
        if stop_str not in self.tokenizer.decode(response_ids[start:]):
            return list(response_ids), False
        lo, hi = start + 1, len(response_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if stop_str in self.tokenizer.decode(response_ids[start:mid]):
                hi = mid
            else:
                lo = mid + 1
        return response_ids[:lo], True


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

    def __init__(
        self,
        tokenizer,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        # clear_thinking=False keeps reasoning blocks in the token stream.
        merged = {"clear_thinking": False, **(chat_template_kwargs or {})}
        super().__init__(tokenizer, merged)
        # Resolve special-token ids at load time; fail loudly if missing.
        self.observation_token_id = self.require_token_id("<|observation|>")
        self.user_token_id = self.require_token_id("<|user|>")
        self.ambiguous_token_ids = {self.observation_token_id, self.user_token_id}

    def build_native_tool_response(
        self, prefix_ids: list[int], tool_messages: list[dict]
    ) -> list[int]:
        incremental = self.render_incremental(tool_messages)
        prefix = list(prefix_ids)
        # Strip trailing ambiguous stop==start marker to avoid duplication.
        if prefix and prefix[-1] in self.ambiguous_token_ids:
            prefix = prefix[:-1]
        return prefix + incremental


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


def _agentic_tokenizer_type(tokenizer_type: str) -> AgenticTokenizerType:
    try:
        return AgenticTokenizerType(tokenizer_type)
    except ValueError as e:
        raise ValueError(
            f"Unknown agentic_tokenizer type: {tokenizer_type!r}. "
            f"Valid: {[t.value for t in AgenticTokenizerType]}"
        ) from e


def get_agentic_tokenizer(
    tokenizer,
    tokenizer_type: str = "default",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> AgenticTokenizer:
    """Build an AgenticTokenizer wrapping the given HF tokenizer."""
    cls = AGENTIC_TOKENIZER_REGISTRY[_agentic_tokenizer_type(tokenizer_type)]
    return cls(tokenizer, chat_template_kwargs)


def agentic_tokenizer_tool_call_parser(tokenizer_type: str = "default") -> str | None:
    """sglang --tool-call-parser bound to an agentic tokenizer family."""
    return AGENTIC_TOKENIZER_REGISTRY[
        _agentic_tokenizer_type(tokenizer_type)
    ].tool_call_parser


__all__ = [
    "AgenticTokenizer",
    "QwenAgenticTokenizer",
    "GLMAgenticTokenizer",
    "AgenticTokenizerType",
    "get_agentic_tokenizer",
    "agentic_tokenizer_tool_call_parser",
]
