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

"""Aligned multimodal preprocessing and span helpers for :class:`SessionServerWorker`."""

import base64
import binascii
import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from rlinf.agents.tokenization.agentic_tokenizer import Qwen35AgenticTokenizer
from rlinf.hybrid_engines.sglang.multimodal import (
    IMAGE_SPANS_PARAM,
    validate_image_spans,
)
from rlinf.utils.multimodal import process_single_image
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.workers.rollout.session_server.chat_completions import (
    ChatCompletions,
    ChatConversation,
    RenderedInput,
    _match_prefix,
    _template_messages,
    _tools_equal,
)

VISION_MARKER_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
IMAGE_PLACEHOLDER_MARKER = "".join(VISION_MARKER_TOKENS)


def attach_multimodal_forward_params(
    forwarded: dict[str, Any],
    image_spans: list[list[int]],
    raw_images: list[bytes],
) -> None:
    """Attach base64 ``image_data`` and ``IMAGE_SPANS_PARAM`` to an SGLang payload."""
    sampling_params = copy.deepcopy(forwarded["sampling_params"])
    custom_params = dict(sampling_params.get("custom_params") or {})
    custom_params[IMAGE_SPANS_PARAM] = [list(span) for span in image_spans]
    sampling_params["custom_params"] = custom_params
    forwarded["sampling_params"] = sampling_params
    forwarded["image_data"] = [
        base64.b64encode(raw).decode("ascii") for raw in raw_images
    ]


def _decode_image_data_url(url: str) -> bytes:
    """Decode a ``data:image/...;base64,...`` URL into raw image bytes."""
    header, sep, payload = url.partition(",")
    if not sep or ";base64" not in header.lower():
        raise ValueError("Chat image inputs must be base64 data:image URLs")
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 payload in chat image URL") from exc


def extract_chat_images(messages: Optional[list[dict]]) -> list[bytes]:
    """Extract and decode embedded ``data:image/...`` URLs from OpenAI messages."""
    images: list[bytes] = []
    for message in messages or ():
        content = message.get("content")
        if not isinstance(content, list):
            continue
        is_system = message.get("role") == "system"
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            if is_system:
                raise ValueError("System messages cannot contain image inputs")
            url = (part.get("image_url") or {}).get("url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("Chat image inputs must be embedded image data URLs")
            images.append(_decode_image_data_url(url))
    return images


def decode_generate_images(image_data: Any) -> list[bytes]:
    """Normalize raw bytes or base64 strings supplied on ``/generate``."""
    if not isinstance(image_data, (list, tuple)):
        raise ValueError("image_data must be a list of images")
    decoded: list[bytes] = []
    for item in image_data:
        if isinstance(item, (bytes, bytearray)):
            decoded.append(bytes(item))
        elif isinstance(item, str):
            if item.startswith("data:image/"):
                decoded.append(_decode_image_data_url(item))
            else:
                try:
                    decoded.append(base64.b64decode(item, validate=True))
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("Invalid base64 image_data item") from exc
        else:
            raise ValueError(
                f"Expected raw bytes or base64 image string, got {type(item).__name__}"
            )
    return decoded


def resolve_vision_marker_ids(tokenizer: Any) -> tuple[int, int, int]:
    """Resolve and validate ``(vision_start_id, image_pad_id, vision_end_id)``."""
    marker_ids = tokenizer.convert_tokens_to_ids(list(VISION_MARKER_TOKENS))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        not isinstance(marker_ids, (list, tuple))
        or len(marker_ids) != 3
        or any(
            type(token_id) is not int or (unk_id is not None and token_id == unk_id)
            for token_id in marker_ids
        )
        or len(set(marker_ids)) != 3
    ):
        raise ValueError(
            f"Tokenizer is missing distinct vision markers {VISION_MARKER_TOKENS}: "
            f"got {marker_ids!r}."
        )
    return int(marker_ids[0]), int(marker_ids[1]), int(marker_ids[2])


def _choose_literal_sentinel(
    messages: Sequence[dict], tools: Optional[list] = None
) -> str:
    """Choose a temporary placeholder absent from serialized messages and tools."""
    haystack = json.dumps((messages, tools), ensure_ascii=False, default=str)
    suffix = 0
    while True:
        candidate = f"__rlinf_lit_imgpad_{suffix}__"
        if candidate not in haystack:
            return candidate
        suffix += 1


def _escape_literal_image_pad(value: Any, sentinel: str) -> Any:
    """Temporarily replace literal ``<|image_pad|>`` in nested text values."""
    token = VISION_MARKER_TOKENS[1]

    def _scrub(node: Any) -> Any:
        if isinstance(node, str):
            return node.replace(token, sentinel)
        if isinstance(node, list):
            return [_scrub(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_scrub(item) for item in node)
        if isinstance(node, dict):
            if node.get("type") == "image_url":
                return copy.deepcopy(node)
            return {
                (_scrub(key) if isinstance(key, str) else key): _scrub(val)
                for key, val in node.items()
            }
        return copy.deepcopy(node)

    return _scrub(value)


@dataclass
class MultimodalSessionContext:
    """Shared HuggingFace processor and image cache for one :class:`SessionServerWorker`."""

    model_path: str
    trust_remote_code: bool = False
    precision: Optional[str] = None
    image_cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        (
            self.vision_start_id,
            self.image_pad_id,
            self.vision_end_id,
        ) = resolve_vision_marker_ids(self.tokenizer)
        self.pixel_dtype = (
            torch_dtype_from_precision(self.precision)
            if self.precision is not None
            else None
        )

    def _process_image_bytes(self, raw_bytes: bytes) -> tuple[Any, Any, int]:
        """Preprocess raw image bytes and convert PIL decode failures to ValueError."""
        try:
            return process_single_image(
                raw_bytes,
                self.processor,
                dtype=self.pixel_dtype,
                cache=self.image_cache,
            )
        except OSError as exc:
            raise ValueError(f"Invalid image payload: {exc}") from exc

    def _expand_rendered_text(
        self,
        rendered_text: str,
        raw_images: list[bytes],
        sentinel: str,
        *,
        prefix_ids: Optional[list[int]] = None,
    ) -> tuple[list[int], list[list[int]], list[str], dict[str, tuple[Any, Any]]]:
        """Split rendered text on real image placeholders and expand aligned spans."""
        segments = rendered_text.split(IMAGE_PLACEHOLDER_MARKER)
        if len(segments) != len(raw_images) + 1:
            raise ValueError(
                f"Chat template emitted {len(segments) - 1} image markers for "
                f"{len(raw_images)} images"
            )
        token_ids: list[int] = list(prefix_ids or ())
        spans: list[list[int]] = []
        digests: list[str] = []
        tensors: dict[str, tuple[Any, Any]] = {}
        token_pad = VISION_MARKER_TOKENS[1]

        for idx, segment in enumerate(segments):
            plain = segment.replace(sentinel, token_pad)
            if plain:
                token_ids.extend(self.tokenizer.encode(plain, add_special_tokens=False))
            if idx >= len(raw_images):
                continue
            raw_bytes = raw_images[idx]
            digest = hashlib.sha256(raw_bytes).hexdigest()
            pixels, grid_thw, num_tokens = self._process_image_bytes(raw_bytes)
            token_ids.append(self.vision_start_id)
            start = len(token_ids)
            token_ids.extend([self.image_pad_id] * num_tokens)
            end = len(token_ids)
            token_ids.append(self.vision_end_id)
            spans.append([start, end])
            digests.append(digest)
            tensors[digest] = (pixels, grid_thw)
        return token_ids, spans, digests, tensors

    def render_chat(
        self,
        chat_entry: ChatCompletions,
        conversation: Optional[ChatConversation],
        last_tokens: list[int],
        messages: list[dict],
        tools: Optional[list],
        image_history: Optional[dict[str, Any]],
    ) -> tuple[RenderedInput, Optional[dict[str, Any]]]:
        """Render OpenAI chat messages, expanding aligned multimodal image spans."""
        raw_images = extract_chat_images(messages)
        if not raw_images:
            rendered = chat_entry.render(conversation, last_tokens, messages, tools)
            return rendered, None

        stored = conversation.messages if conversation is not None else []
        stored_tools = conversation.tools if conversation is not None else None
        rollback = False
        history = image_history

        if (
            _tools_equal(tools, stored_tools)
            and conversation is not None
            and conversation.checkpoint is not None
            and stored
            and len(messages) == len(stored) - 1
            and _match_prefix(messages, stored[:-1]) is not None
        ):
            checkpoint = conversation.checkpoint
            rollback = True
            stored = list(checkpoint.messages)
            stored_tools = checkpoint.tools
            last_tokens = list(checkpoint.last_input or []) + list(
                checkpoint.last_output or []
            )
            history = checkpoint.image_history

        agentic_tokenizer = chat_entry._tokenizer
        if (
            _tools_equal(tools, stored_tools)
            and stored
            and len(messages) > len(stored)
            and _match_prefix(messages, stored) is not None
            and isinstance(agentic_tokenizer, Qwen35AgenticTokenizer)
        ):
            suffix = messages[len(stored) :]
            if not agentic_tokenizer.clears_thinking_for(suffix):
                sentinel = _choose_literal_sentinel(suffix)
                templated_suffix = _escape_literal_image_pad(
                    _template_messages(suffix), sentinel
                )
                inc_text = agentic_tokenizer.render_incremental_text(templated_suffix)
                suffix_images = extract_chat_images(suffix)
                prefix_image_count = len(raw_images) - len(suffix_images)
                base_ids = agentic_tokenizer.join_increment(list(last_tokens), [])
                expanded_ids, new_spans, new_digests, new_tensors = (
                    self._expand_rendered_text(
                        inc_text, suffix_images, sentinel, prefix_ids=base_ids
                    )
                )
                prev_spans = (history or {}).get("spans", [])
                prev_digests = (history or {}).get("digests", [])
                prev_tensors = (history or {}).get("tensors", {})
                if len(prev_spans) == prefix_image_count:
                    all_spans = [list(span) for span in prev_spans] + new_spans
                    all_digests = list(prev_digests) + new_digests
                    tensors = {**prev_tensors, **new_tensors}
                    validate_image_spans(
                        expanded_ids,
                        all_spans,
                        vision_start_id=self.vision_start_id,
                        image_pad_id=self.image_pad_id,
                        vision_end_id=self.vision_end_id,
                        image_count=len(raw_images),
                    )
                    return (
                        RenderedInput(
                            expanded_ids,
                            messages=[*stored, *suffix],
                            rollback=rollback,
                        ),
                        {
                            "raw_images": raw_images,
                            "spans": all_spans,
                            "digests": all_digests,
                            "tensors": tensors,
                        },
                    )

        sentinel = _choose_literal_sentinel(messages, tools)
        templated = _escape_literal_image_pad(_template_messages(messages), sentinel)
        scrubbed_tools = (
            _escape_literal_image_pad(tools, sentinel) if tools is not None else None
        )
        full_text = agentic_tokenizer.apply_chat_template(
            templated,
            tools=scrubbed_tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        expanded_ids, all_spans, all_digests, tensors = self._expand_rendered_text(
            full_text, raw_images, sentinel
        )
        validate_image_spans(
            expanded_ids,
            all_spans,
            vision_start_id=self.vision_start_id,
            image_pad_id=self.image_pad_id,
            vision_end_id=self.vision_end_id,
            image_count=len(raw_images),
        )
        return (
            RenderedInput(expanded_ids, messages=list(messages), rollback=rollback),
            {
                "raw_images": raw_images,
                "spans": all_spans,
                "digests": all_digests,
                "tensors": tensors,
            },
        )

    def prepare_generate_turn(
        self, forwarded: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Validate token-ID image arguments and prepare engine forward + payload data."""
        raw_image_data = forwarded.get("image_data")
        raw_spans = forwarded.pop("image_spans", None)
        if not raw_image_data:
            if raw_spans is not None:
                raise ValueError(
                    "image_spans requires non-empty image_data under aligned multimodal rollouts"
                )
            forwarded.pop("image_data", None)
            return None

        raw_images = decode_generate_images(raw_image_data)
        if raw_spans is None:
            raise ValueError(
                "Image requests require explicit image_spans under aligned multimodal rollouts"
            )
        prompt_ids = forwarded["input_ids"]
        validated = validate_image_spans(
            prompt_ids,
            raw_spans,
            vision_start_id=self.vision_start_id,
            image_pad_id=self.image_pad_id,
            vision_end_id=self.vision_end_id,
            image_count=len(raw_images),
        )
        spans: list[list[int]] = []
        digests: list[str] = []
        tensors: dict[str, tuple[Any, Any]] = {}
        for raw_bytes, (start, end) in zip(raw_images, validated, strict=True):
            pixels, grid_thw, num_tokens = self._process_image_bytes(raw_bytes)
            if num_tokens != end - start:
                raise ValueError(
                    f"Processed image patch width ({num_tokens}) differs from prompt span "
                    f"[{start}, {end}) width ({end - start})"
                )
            digest = hashlib.sha256(raw_bytes).hexdigest()
            spans.append([start, end])
            digests.append(digest)
            tensors[digest] = (pixels, grid_thw)

        attach_multimodal_forward_params(forwarded, spans, raw_images)
        return {
            "raw_images": raw_images,
            "spans": spans,
            "digests": digests,
            "tensors": tensors,
        }
