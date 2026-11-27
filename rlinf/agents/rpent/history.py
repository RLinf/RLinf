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

"""Pydantic-AI-faithful Qwen conversation, sliding-window compaction, and token-prefix history."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from rlinf.agents.tokenization import AgenticTokenizer
from rlinf.workers.agent.agent_loop import AgentLoopOutput

IMAGE_DATA_PREFIX = "data:"
IMAGE_PLACEHOLDER_PREFIX = "rpent-image://"
TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"
_PATCH_BLOCK = 32  # Qwen3.5 16px patch x 2x2 spatial merge
_DEFAULT_IMAGE_TOKENS = 1024


def _image_tokens(png_bytes: bytes) -> int:
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            w, h = img.size
        return math.ceil(w / _PATCH_BLOCK) * math.ceil(h / _PATCH_BLOCK)
    except Exception:
        return _DEFAULT_IMAGE_TOKENS


def _argument_hint(name: str, environment: str) -> str:
    if environment == "metaworld" and name in {
        "execute_actions",
        "move_to",
        "set_gripper",
    }:
        return "Use move_to with xyz only (fixed wrist orientation) or set_gripper with opening 0..1 in execute_actions."
    if environment == "robotwin" and name in {"move_to", "set_gripper"}:
        return "Follow the declared native action fields and orientation convention; include the selected arm where required."
    if environment == "robocasa" and name in {"move_to", "set_gripper"}:
        return (
            "Follow the declared move_to/set_gripper fields and orientation convention."
        )
    return {
        "segment": "Supply exactly one non-empty prompt or point [row,col]; follow the tool's camera/step requirements.",
        "back_project": "Supply exactly one pixel, box or mask_id; pixel/box must match the displayed camera and frame.",
        "move_to": "Supply xyz, orientation_rpy, or both as three finite numbers.",
        "move_base": "Supply bounded local velocities vx/vy/yaw_rate and duration_s within the declared limits.",
        "move_lift": "Supply delta_m within +/-0.25 and duration_s within 0.1..5.",
        "execute_actions": "Supply a non-empty actions list within K; each item must be move_to with a target or set_gripper with opening 0..1.",
    }.get(name, "Follow the declared required fields and value ranges.")


@dataclass(frozen=True)
class LogicalMessage:
    """One Pydantic-AI-equivalent logical message for sliding-window compaction.

    ``wire`` stores lightweight template-ready dicts with ``rpent-image://<fid>``
    placeholders; raw PNG bytes are stored directly in ``images`` without Base64.
    """

    kind: str  # "opening", "assistant", "tool", "retry"
    wire: tuple[dict[str, Any], ...]
    tool_call_ids: tuple[str, ...] = ()
    images: tuple[bytes, ...] = ()
    image_tokens: int = 0


@dataclass(frozen=True)
class CompactionConfig:
    """Sliding-window compaction settings matching RPent RequestAwareSlidingWindow."""

    max_messages: int | None = 80
    keep_messages: int = 10
    context_window: int | None = 262144
    max_fraction: float | None = 0.7
    max_tokens: int | None = None
    preserve_first_user_message: bool = True

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> CompactionConfig | None:
        if cfg is None:
            return None
        return cls(
            max_messages=cfg.get("max_messages", 80),
            keep_messages=int(cfg.get("keep_messages", 10)),
            context_window=cfg.get("context_window", 262144),
            max_fraction=cfg.get("max_fraction", 0.7),
            max_tokens=cfg.get("max_tokens"),
            preserve_first_user_message=bool(
                cfg.get("preserve_first_user_message", True)
            ),
        )

    @property
    def token_trigger(self) -> int | None:
        if self.max_tokens is not None:
            return int(self.max_tokens)
        if self.max_fraction is not None and self.context_window is not None:
            return int(self.context_window * self.max_fraction)
        return None


def find_safe_cutoff(messages: Sequence[LogicalMessage], keep_messages: int) -> int:
    """Walk backward from len - keep_messages so no assistant/tool pair is split."""
    if keep_messages <= 0 or len(messages) <= keep_messages:
        return 0
    cutoff = len(messages) - keep_messages
    call_owner: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        if msg.kind == "assistant":
            for cid in msg.tool_call_ids:
                call_owner[cid] = idx
    changed = True
    while changed and cutoff > 0:
        changed = False
        for msg in messages[cutoff:]:
            if msg.kind == "tool":
                for cid in msg.tool_call_ids:
                    owner = call_owner.get(cid)
                    if owner is not None and owner < cutoff:
                        cutoff = owner
                        changed = True
    return cutoff


@dataclass
class EpisodeConversation:
    """In-process conversation state matching Pydantic AI + QwenChatModel wire format."""

    system_prompt: str
    user_message: str
    initial_observation_text: str
    tools: list[dict[str, Any]]
    environment: str
    compaction: CompactionConfig | None = None
    max_output_retries: int = 1
    messages: list[LogicalMessage] = field(default_factory=list)
    output_retries: int = 0

    def __post_init__(self) -> None:
        self._tool_schemas = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in self.tools
            if t.get("type") == "function"
        }
        if not self.messages:
            self.messages = [
                LogicalMessage(
                    kind="opening",
                    wire=(
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"{self.user_message}\n\n{self.initial_observation_text}",
                        },
                    ),
                )
            ]

    def wire_messages(self) -> list[dict[str, Any]]:
        """Return lightweight template message dicts (no Base64 payloads)."""
        return [item for msg in self.messages for item in msg.wire]

    def images(self) -> list[bytes]:
        """Return ordered raw binary PNG payloads across active messages."""
        return [img for msg in self.messages for img in msg.images]

    def total_image_tokens(self) -> int:
        return sum(msg.image_tokens for msg in self.messages)

    def estimate_tokens(self) -> int:
        chars = len(json.dumps(self.tools, ensure_ascii=False))
        for msg in self.messages:
            for wire in msg.wire:
                content = wire.get("content")
                if isinstance(content, str):
                    chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            chars += len(part.get("text", ""))
                chars += len(wire.get("reasoning_content") or "")
                if wire.get("tool_calls"):
                    chars += len(json.dumps(wire["tool_calls"], ensure_ascii=False))
        return math.ceil(chars / 4) + self.total_image_tokens()

    def maybe_compact(self, current_tokens: int | None = None) -> bool:
        cfg = self.compaction
        if cfg is None:
            return False
        over_msgs = (
            cfg.max_messages is not None and len(self.messages) > cfg.max_messages
        )
        limit = cfg.token_trigger
        if not over_msgs and limit is not None:
            tokens = (
                self.estimate_tokens() if current_tokens is None else current_tokens
            )
            over_toks = tokens > limit
        else:
            over_toks = False
        if not (over_msgs or over_toks):
            return False
        cutoff = find_safe_cutoff(self.messages, cfg.keep_messages)
        if cutoff <= 0:
            return False
        trimmed = list(self.messages[cutoff:])
        if cfg.preserve_first_user_message and (
            not trimmed or trimmed[0] is not self.messages[0]
        ):
            trimmed = [self.messages[0], *trimmed]
        self.messages = trimmed
        return True

    def append_assistant(
        self,
        message: dict[str, Any] | None = None,
        *,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        if message is None:
            message = {"role": "assistant", "content": content}
            if reasoning_content is not None:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
        call_ids = tuple(tc["id"] for tc in message.get("tool_calls") or ())
        self.messages.append(
            LogicalMessage(kind="assistant", wire=(message,), tool_call_ids=call_ids)
        )

    def append_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        text: str,
        images: Sequence[bytes] = (),
    ) -> None:
        self.output_retries = 0
        if not images:
            payload = json.dumps([text], separators=(",", ":"), ensure_ascii=False)
            self.messages.append(
                LogicalMessage(
                    kind="tool",
                    wire=(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": payload,
                        },
                    ),
                    tool_call_ids=(tool_call_id,),
                )
            )
            return
        raw_images = tuple(bytes(img) for img in images)
        fids = [hashlib.sha1(img).hexdigest()[:6] for img in raw_images]
        tool_payload = json.dumps(
            [text, *(f"See file {fid}." for fid in fids)],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": TOOL_RESPONSE_OPEN}]
        img_toks = 0
        for img, fid in zip(raw_images, fids, strict=True):
            img_toks += _image_tokens(img)
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f'<tool_result tool_name="{tool_name}" '
                            f'tool_call_id="{tool_call_id}" file_id="{fid}">'
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{IMAGE_PLACEHOLDER_PREFIX}{fid}"},
                    },
                    {"type": "text", "text": "</tool_result>"},
                ]
            )
        parts.append({"type": "text", "text": TOOL_RESPONSE_CLOSE})
        self.messages.append(
            LogicalMessage(
                kind="tool",
                wire=(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_payload,
                    },
                    {"role": "user", "content": parts},
                ),
                tool_call_ids=(tool_call_id,),
                images=raw_images,
                image_tokens=img_toks,
            )
        )

    def validate_or_retry(
        self, calls: Sequence[dict[str, Any]]
    ) -> tuple[str | None, str | None, dict[str, Any] | None, bool]:
        """Validate parsed tool calls; append retry feedback if within retry budget."""
        if not calls:
            return None, None, None, False
        err_kind: str | None = None
        err_msg: str | None = None
        parsed: dict[str, Any] | None = None
        if len(calls) > 1:
            err_kind = "tool_arguments"
            err_msg = "Use one tool call per response. No tool executed."
        else:
            name = calls[0]["name"]
            if name not in self._tool_schemas:
                err_kind = "tool_name"
                err_msg = f"Unknown tool name: {name}. No tool executed."
            else:
                raw = calls[0]["arguments"]
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    errors = list(
                        Draft202012Validator(self._tool_schemas[name]).iter_errors(args)
                    )
                    if errors:
                        err = errors[0]
                        loc = ".".join(str(x) for x in err.absolute_path)
                        raise ValueError(f"{loc or name}: {err.message}")
                    box = args.get("box")
                    if box is not None and (box[0] >= box[2] or box[1] >= box[3]):
                        raise ValueError(
                            "box must have row_min < row_max and col_min < col_max"
                        )
                    z = args.get("z_range_m", args.get("z_range_cm"))
                    if z is not None and z[0] > z[1]:
                        raise ValueError(
                            "z_range_m must be [low,high] with low <= high"
                        )
                    parsed = args
                except (AssertionError, ValueError, TypeError) as exc:
                    err_kind = "tool_arguments"
                    err_msg = (
                        f"Invalid tool arguments: {exc}. "
                        f"{_argument_hint(name, self.environment)} No tool executed."
                    )
        if err_kind is None:
            self.output_retries = 0
            return None, None, parsed, False
        self.output_retries += 1
        if self.output_retries <= self.max_output_retries:
            self.messages.append(
                LogicalMessage(
                    kind="retry",
                    wire=(
                        {
                            "role": "user",
                            "content": f"Validation feedback:\n{err_msg}\n\nFix the errors and try again.",
                        },
                    ),
                )
            )
            return err_kind, err_msg, None, True
        return err_kind, err_msg, None, False


# ---------------------------------------------------------------------------
# Token-preserving append-only history & turn merging
# ---------------------------------------------------------------------------


def _image_bytes(part: dict[str, Any]) -> bytes | None:
    if part.get("type") != "image_url":
        return None
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str) or not url.startswith(IMAGE_DATA_PREFIX):
        raise ValueError(f"image parts must carry a base64 data URI, got {url!r}")
    _, _, payload = url.partition(",")
    return base64.b64decode(payload, validate=True)


def message_images(messages: Sequence[dict[str, Any]]) -> list[bytes]:
    images: list[bytes] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            images.extend(
                img for part in content if (img := _image_bytes(part)) is not None
            )
    return images


def _normalize_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Shallow-normalize one message dict for chat templating without deep-copying blobs."""
    msg = dict(raw)
    if msg.get("role") == "assistant" and msg.get("reasoning_content") == "":
        msg.pop("reasoning_content", None)
    if msg.get("content") is None:
        msg["content"] = ""
    if isinstance(msg["content"], list) or not msg.get("tool_calls"):
        return msg
    calls, malformed = [], []
    for call in msg["tool_calls"]:
        fn = call["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if isinstance(args, dict):
            calls.append({**call, "function": {**fn, "arguments": args}})
        else:
            malformed.append(json.dumps(call, ensure_ascii=False))
    if malformed:
        msg["content"] = msg["content"] + "\n" + "\n".join(malformed)
    msg["tool_calls"] = calls
    return msg


def template_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_message(msg) for msg in messages]


@dataclass
class RenderedTurn:
    """Compact-marker prompt tokens and ordered image bytes for one turn."""

    input_ids: list[int]
    images: list[bytes] = field(default_factory=list)
    image_spans: list[tuple[int, int]] = field(default_factory=list)
    prefix_length: int = 0


class AppendOnlyHistory:
    """Append newly added messages to previous sampled token IDs to preserve prefixes."""

    def __init__(self, tokenizer: AgenticTokenizer) -> None:
        self.tokenizer = tokenizer
        self._image_pad_id = tokenizer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._messages: list[dict[str, Any]] | None = None
        self._tools: list[dict[str, Any]] | None = None
        self._assistant_message: dict[str, Any] | None = None
        self._ids: list[int] | None = None
        self._rendered: RenderedTurn | None = None
        self.image_history: dict[str, Any] | None = None
        self._chain_id = 0

    @property
    def chain_id(self) -> int:
        return self._chain_id

    @property
    def token_length(self) -> int:
        return len(self._ids) if self._ids is not None else 0

    @contextmanager
    def request(self):
        prev = (self._rendered, self.image_history, self._chain_id)
        try:
            yield
        except BaseException:
            self._rendered, self.image_history, self._chain_id = prev
            raise

    def _begin_chain(self) -> None:
        self._chain_id += 1
        self._rendered = None
        self.image_history = None

    def _continues(
        self, messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> bool:
        count = 0 if self._messages is None else len(self._messages)
        return (
            self._messages is not None
            and tools == self._tools
            and len(messages) > count + 1
            and list(messages[:count]) == self._messages
            and _normalize_message(messages[count]) == self._assistant_message
            and all(m.get("role") in {"user", "tool"} for m in messages[count + 1 :])
        )

    def render_turn(
        self,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: Sequence[bytes] | None = None,
    ) -> RenderedTurn:
        """Render prompt IDs with compact <|image_pad|> spans for SGLang expansion.

        When ``images`` is provided, raw PNG bytes are reused directly by reference
        without Base64 encoding/decoding or deep copying.
        """
        continues = self._continues(messages, tools)
        delta_msgs = template_messages(
            messages[len(self._messages) + 1 :] if continues else messages
        )
        if continues:
            increment = self.tokenizer.render_incremental(delta_msgs)
            ids = self.tokenizer.join_increment(self._ids, increment)
            offset = len(ids) - len(increment)
            old_images = self._rendered.images if self._rendered is not None else []
            old_spans = self._rendered.image_spans if self._rendered is not None else []
            prefix_length = len(self._ids)
            delta_images = (
                list(images[len(old_images) :])
                if images is not None
                else message_images(delta_msgs)
            )
        else:
            self._begin_chain()
            text = self.tokenizer.apply_chat_template(
                delta_msgs, tools=tools, tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]
            increment, offset = ids, 0
            old_images, old_spans, prefix_length = [], [], 0
            delta_images = (
                list(images) if images is not None else message_images(delta_msgs)
            )

        spans = [
            (offset + i, offset + i + 1)
            for i, tok in enumerate(increment)
            if tok == self._image_pad_id
        ]
        if len(spans) != len(delta_images):
            raise ValueError(
                f"Structured images ({len(delta_images)}) and <|image_pad|> markers ({len(spans)}) disagree"
            )
        self._rendered = RenderedTurn(
            input_ids=ids,
            images=old_images + delta_images,
            image_spans=[*old_spans, *spans],
            prefix_length=prefix_length,
        )
        return self._rendered

    def accept_capture(self, captured: dict[str, Any]) -> list[int]:
        """Adopt SGLang's expanded image token IDs and spans while asserting prefix invariance."""
        assert self._rendered is not None
        ids = captured["prompt_token_ids"]
        if (
            ids[: self._rendered.prefix_length]
            != self._rendered.input_ids[: self._rendered.prefix_length]
        ):
            raise ValueError("SGLang image expansion altered the sampled token prefix")
        self._rendered = replace(
            self._rendered,
            input_ids=ids,
            image_spans=[tuple(s) for s in captured["image_spans"]],
        )
        return ids

    def record(
        self,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_ids: list[int],
        response_ids: list[int],
        assistant_message: dict[str, Any],
    ) -> None:
        # ``messages`` holds lightweight placeholder dicts (no Base64 blobs);
        # storing shallow list references avoids quadratic deepcopy overhead.
        self._messages = list(messages)
        self._tools = list(tools) if tools is not None else None
        self._assistant_message = _normalize_message(assistant_message)
        self._ids = [*prompt_ids, *response_ids]


def merge_prefix_turns(
    turns: list[AgentLoopOutput], max_scored_tokens: int
) -> list[AgentLoopOutput]:
    """Fold consecutive prefix-compatible turns in the same chain into one training row."""
    merged: list[AgentLoopOutput] = []
    for turn in turns:
        mask = list(turn.response_mask or [1] * len(turn.response_ids))
        last = merged[-1] if merged else None
        history = last.prompt_ids + last.response_ids if last else None
        if (
            last is not None
            and turn.extra_fields.get("chain_id") == last.extra_fields.get("chain_id")
            and turn.prompt_ids[: len(history)] == history
            and sum(last.response_mask) + sum(mask) <= max_scored_tokens
        ):
            gap = len(turn.prompt_ids) - len(history)
            last.response_ids = (turn.prompt_ids + turn.response_ids)[
                len(last.prompt_ids) :
            ]
            last.response_mask += [0] * gap + mask
            if last.response_logprobs is not None:
                last.response_logprobs += [0.0] * gap + list(turn.response_logprobs)
            if last.response_weight_versions is not None:
                if turn.response_weight_versions is None:
                    last.response_weight_versions = None
                else:
                    carried = last.response_weight_versions[-1]
                    last.response_weight_versions += [carried] * gap + list(
                        turn.response_weight_versions
                    )
            last.response_images = turn.response_images
            last.version_end = turn.version_end
            last.response_text, last.is_end = turn.response_text, turn.is_end
            last.reward_score = turn.reward_score
            last.extra_fields = {
                **turn.extra_fields,
                "turn_ids": last.extra_fields["turn_ids"]
                + [turn.extra_fields.get("turn_id")],
            }
            continue
        merged.append(
            replace(
                turn,
                response_mask=mask,
                response_logprobs=None
                if turn.response_logprobs is None
                else list(turn.response_logprobs),
                response_weight_versions=None
                if turn.response_weight_versions is None
                else list(turn.response_weight_versions),
                extra_fields={
                    **turn.extra_fields,
                    "turn_ids": [turn.extra_fields.get("turn_id")],
                },
            )
        )
    return merged
