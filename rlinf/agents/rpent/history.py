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

"""Render RPent planner chat requests into prompt token IDs and images."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from rlinf.agents.tokenization import AgenticTokenizer

IMAGE_DATA_PREFIX = "data:"
"""A data URI carries the bytes of one image inside its `image_url` part."""


class TemplateMessageError(ValueError):
    """A planner message the renderer cannot turn into prompt tokens."""


def _image_bytes(part: dict) -> bytes | None:
    """Decode one `image_url` content part, or return None if it is not one."""
    if part.get("type") != "image_url":
        return None
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str) or not url.startswith(IMAGE_DATA_PREFIX):
        raise TemplateMessageError(
            f"image parts must carry a base64 data URI, got {url!r}"
        )
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        raise TemplateMessageError(f"image data URI is not base64: {header!r}")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TemplateMessageError(
            f"image data URI does not decode: {error}"
        ) from error


def message_images(messages: list[dict]) -> list[bytes]:
    """Return the images embedded in a request's messages, in order."""
    images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            images.extend(
                image for part in content if (image := _image_bytes(part)) is not None
            )
    return images


def template_messages(messages: list[dict]) -> list[dict]:
    """Return messages in the form the chat template accepts.

    Tool-call arguments become mappings; calls whose arguments are not a JSON
    object are kept as text in the message content. Image parts keep their
    content list so the chat template can place a vision marker for each one.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        # The planner omits empty reasoning when it echoes an assistant reply.
        if (
            message.get("role") == "assistant"
            and message.get("reasoning_content") == ""
        ):
            del message["reasoning_content"]
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            if not isinstance(message["content"], list):
                raise TemplateMessageError(
                    f"RPent prompts need string or content-list messages, got "
                    f"{type(message['content']).__name__}"
                )
            for part in message["content"]:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind not in ("text", "image_url"):
                    raise TemplateMessageError(
                        f"unsupported content part {part!r}; RPent prompts carry "
                        "text and base64 image_url parts"
                    )
                _image_bytes(part) if kind == "image_url" else None
            continue
        calls, malformed = [], []
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if isinstance(arguments, dict):
                call["function"]["arguments"] = arguments
                calls.append(call)
            else:
                malformed.append(json.dumps(call, ensure_ascii=False))
        if malformed:
            message["content"] += "\n" + "\n".join(malformed)
            message["tool_calls"] = calls
    return messages


@dataclass
class ImagePrompt:
    """Prompt tokens of one image-bearing planner request and its images."""

    input_ids: list[int]
    images: list[bytes] = field(default_factory=list)
    """PNG bytes of each image, in the order the template placed its marker."""
    image_grid_thw: Any = None
    """Processor token grid of the images, kept for the training forward."""
    pixel_values: Any = None
    """Processor pixels, used directly when retaining an in-memory row."""
    image_spans: list[tuple[int, int]] = field(default_factory=list)
    """Half-open image spans; new images initially occupy one placeholder."""
    prefix_length: int = 0
    """Number of already expanded tokens preserved from the previous turn."""

    def store_images(self, directory: Path) -> dict:
        """Store immutable image bytes once per episode for lazy training reads.

        The directory must remain visible to learners and checkpoint recovery.
        A turn retains its own ordered references even when history is compacted.
        """
        directory.mkdir(parents=True, exist_ok=True)
        references = []
        for image in self.images:
            digest = hashlib.sha256(image).hexdigest()
            path = directory / digest
            if not path.exists():
                # Concurrent writers have identical bytes. Publish only complete files.
                with NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        stream.write(image)
                        stream.flush()
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            references.append({"path": str(path.resolve()), "sha256": digest})
        return {
            "image_refs": references,
            "image_grid_thw": self.image_grid_thw,
            "num_images": len(references),
        }


def render_prompt(tokenizer: AgenticTokenizer, request: dict) -> list[int]:
    """Render a whole chat request with the model's chat template."""
    text = tokenizer.apply_chat_template(
        template_messages(request["messages"]),
        tools=request.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]


def request_has_images(request: dict) -> bool:
    """Whether a planner request carries image content parts."""
    return any(
        isinstance(message.get("content"), list) for message in request["messages"]
    )


def render_image_prompt(
    tokenizer: AgenticTokenizer,
    request: dict,
    processor: Any,
) -> ImagePrompt:
    """Render an image-bearing request into tokens the image processor expanded.

    The chat template writes one vision marker per image, so the plain render
    holds a single placeholder for each. Expanding it with the model's
    processor turns every placeholder into its grid of pad tokens, which is
    what both the served engine and the training forward score.
    """
    from PIL import Image

    messages = template_messages(request["messages"])
    images = message_images(messages)
    if not images:
        raise TemplateMessageError("render_image_prompt needs at least one image")
    text = tokenizer.apply_chat_template(
        messages, tools=request.get("tools"), tokenize=False, add_generation_prompt=True
    )
    decoded = [Image.open(io.BytesIO(image)).convert("RGB") for image in images]
    inputs = processor(text=[text], images=decoded, return_tensors="pt")
    return ImagePrompt(
        input_ids=inputs["input_ids"][0].tolist(),
        images=images,
        image_grid_thw=inputs["image_grid_thw"],
        pixel_values=inputs["pixel_values"],
    )


def render_image_increment(
    tokenizer: AgenticTokenizer,
    new_messages: list[dict],
    processor: Any,
) -> ImagePrompt:
    """Render only the appended messages of a continuing turn, images expanded.

    ``new_messages`` are the user and tool messages added since the recorded
    assistant reply. The increment text keeps the recorded tokens a prefix of
    this turn, and the processor expands any image markers in it just as a full
    render would, so the appended pad tokens match the engine and the forward.
    """
    from PIL import Image

    text = tokenizer.render_incremental_text(new_messages)
    images = message_images(new_messages)
    if not images:
        ids = tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ImagePrompt(input_ids=ids)
    decoded = [Image.open(io.BytesIO(image)).convert("RGB") for image in images]
    inputs = processor(text=[text], images=decoded, return_tensors="pt")
    return ImagePrompt(
        input_ids=inputs["input_ids"][0].tolist(),
        images=images,
        image_grid_thw=inputs["image_grid_thw"],
        pixel_values=inputs["pixel_values"],
    )


def _extend_image_prompt(
    prior: ImagePrompt | None, increment: ImagePrompt, input_ids: list[int]
) -> ImagePrompt:
    """Return the cumulative images of a prefix chain extended by one turn.

    The appended turn owns ``input_ids``; its images follow the earlier ones in
    the order the template placed their markers, so the row carries every image
    the forward must expand, once each occurrence.
    """
    if prior is None:
        return replace(increment, input_ids=input_ids)
    if not increment.images:
        return replace(prior, input_ids=input_ids)
    import torch

    return ImagePrompt(
        input_ids=input_ids,
        images=prior.images + increment.images,
        image_grid_thw=torch.cat([prior.image_grid_thw, increment.image_grid_thw]),
        pixel_values=None
        if prior.pixel_values is None or increment.pixel_values is None
        else torch.cat([prior.pixel_values, increment.pixel_values]),
    )


class AppendOnlyHistory:
    """Build each turn's prompt by appending to the previous turn's sampled tokens.

    A request that continues the recorded conversation (same earlier messages
    and tools, the recorded assistant reply, then only user or tool messages)
    becomes the previous prompt plus the sampled response plus the new
    messages, so every turn is a token prefix of the next. Any other request,
    such as a compacted history, is rendered in full and starts a new chain.
    """

    def __init__(self, tokenizer: AgenticTokenizer) -> None:
        self.tokenizer = tokenizer
        self._messages = None
        self._tools = None
        self._assistant_message = None
        self._ids = None
        # Cumulative images of the current prefix chain, and the id of that
        # chain. Both reset together whenever the token history is rewritten,
        # so a later turn never inherits a dropped chain's images and only
        # turns of one chain merge. The id advances on every new chain.
        self._image_prompt: ImagePrompt | None = None
        self.image_history: dict | None = None
        self._chain_id = 0

    @property
    def chain_id(self) -> int:
        """Id of the current prefix chain; a rewrite starts a new one."""
        return self._chain_id

    def _begin_chain(self) -> None:
        """Start a fresh prefix chain: new id, no inherited images."""
        self._chain_id += 1
        self._image_prompt = None
        self.image_history = None

    def _continues(self, request: dict) -> bool:
        messages = request["messages"]
        count = 0 if self._messages is None else len(self._messages)
        return (
            self._messages is not None
            and request.get("tools") == self._tools
            and len(messages) > count + 1
            and messages[:count] == self._messages
            and template_messages([messages[count]])[0] == self._assistant_message
            and all(m.get("role") in {"user", "tool"} for m in messages[count + 1 :])
        )

    def render(self, request: dict) -> list[int]:
        """Return the prompt token IDs for a chat request."""
        if not self._continues(request):
            # A text rewrite starts a new chain and drops any earlier images.
            self._begin_chain()
            return render_prompt(self.tokenizer, request)
        new_messages = template_messages(request["messages"][len(self._messages) + 1 :])
        return self.tokenizer.build_native_tool_response(self._ids, new_messages)

    def render_multimodal(self, request: dict, processor: Any) -> ImagePrompt:
        """Return the prompt and the cumulative images for an image-bearing turn.

        A continuing turn expands only its new messages and appends them to the
        recorded tokens, so the turn stays a token prefix of the next and its
        images extend the chain's image list. A rewritten history renders in
        full and starts a new chain. The returned ``image_grid_thw`` and
        ``pixel_values`` cover every image the prompt expands, in order.

        Called once per turn before ``record``; a turn that fails before
        ``record`` abandons the episode, so the chain never reuses stale images.
        """
        if not self._continues(request):
            self._begin_chain()
            self._image_prompt = render_image_prompt(self.tokenizer, request, processor)
            return self._image_prompt
        new_messages = template_messages(request["messages"][len(self._messages) + 1 :])
        increment = render_image_increment(self.tokenizer, new_messages, processor)
        prompt_ids = self.tokenizer.join_increment(self._ids, increment.input_ids)
        self._image_prompt = _extend_image_prompt(
            self._image_prompt, increment, prompt_ids
        )
        return self._image_prompt

    def render_token_space(self, request: dict) -> ImagePrompt:
        """Append compact image markers without processing images on the agent."""
        continues = self._continues(request)
        messages = template_messages(
            request["messages"][len(self._messages) + 1 :]
            if continues
            else request["messages"]
        )
        # The serving API locates media by special tokens. Literal markers must
        # not acquire image semantics, including in tool schemas or responses.
        text_parts = []
        for message in messages:
            message = dict(message)
            if isinstance(message["content"], list):
                message["content"] = [
                    part for part in message["content"] if part["type"] != "image_url"
                ]
            text_parts.append(message)
        literals = json.dumps([text_parts, request.get("tools")], ensure_ascii=False)
        if any(
            marker in literals
            for marker in (
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            )
        ):
            raise TemplateMessageError(
                "Literal vision markers are unsupported in token-space chats"
            )
        images = message_images(messages)
        if continues:
            increment = self.tokenizer.render_incremental(messages)
            ids = self.tokenizer.join_increment(self._ids, increment)
            offset = len(ids) - len(increment)
            prior = self._image_prompt
            old_images = prior.images if prior is not None else []
            old_spans = prior.image_spans if prior is not None else []
            prefix_length = len(self._ids)
        else:
            self._begin_chain()
            ids = render_prompt(self.tokenizer, request)
            increment, offset = ids, 0
            old_images, old_spans, prefix_length = [], [], 0
        image_id = self.tokenizer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        spans = [
            (offset + i, offset + i + 1)
            for i, token in enumerate(increment)
            if token == image_id
        ]
        if len(spans) != len(images):
            raise TemplateMessageError(
                "Structured images and compact image markers disagree"
            )
        self._image_prompt = ImagePrompt(
            ids,
            old_images + images,
            image_spans=[*old_spans, *spans],
            prefix_length=prefix_length,
        )
        return self._image_prompt

    def accept_token_space(self, prepared: dict) -> ImagePrompt:
        """Adopt rollout expansion while preserving the sampled history exactly."""
        prompt = self._image_prompt
        ids = prepared["prompt_token_ids"]
        if ids[: prompt.prefix_length] != prompt.input_ids[: prompt.prefix_length]:
            raise ValueError("Token-space expansion changed the sampled prefix")
        self._image_prompt = replace(
            prompt,
            input_ids=ids,
            image_spans=[tuple(span) for span in prepared["image_spans"]],
            image_grid_thw=prepared["image_grid_thw"],
        )
        return self._image_prompt

    def record(
        self,
        request: dict,
        prompt_ids: list[int],
        response_ids: list[int],
        assistant_message: dict,
    ) -> None:
        """Make a sampled turn the base of the next request."""
        self._messages = copy.deepcopy(request["messages"])
        self._tools = copy.deepcopy(request.get("tools"))
        self._assistant_message = template_messages([assistant_message])[0]
        self._ids = [*prompt_ids, *response_ids]
