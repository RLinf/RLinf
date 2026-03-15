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

import asyncio
import base64
import io
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from rlinf.scheduler import Channel


def encode_image(image: np.ndarray) -> str:
    """Encode a numpy image array into a base64 JPEG string."""
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class LLMWrapper:
    """Client-side helper for multimodal (image + text) generation via Channels."""

    def __init__(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
    ) -> None:
        """Initialize wrapper with rollout input/output channels."""
        self._generate_input_channel = generate_input_channel
        self._generate_output_channel = generate_output_channel

    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, Optional[bool], Any]:
        """Run a multimodal query and return decoded text and raw result."""
        t_encode = time.perf_counter()
        messages = self._build_messages(text_prompt, images)
        image_encode_s = time.perf_counter() - t_encode

        result = self._vl_generate(messages)

        if isinstance(result, dict):
            result["image_encode_s"] = image_encode_s

        return self._parse_result(result)

    def _build_messages(
        self,
        text_prompt: str,
        images: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Build OpenAI-style chat messages with embedded images."""
        content: list[dict[str, str]] = []

        for image in images:
            content.append(
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{encode_image(image)}",
                }
            )
        content.append(
            {
                "type": "text",
                "text": text_prompt,
            }
        )
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _vl_generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send a VL generation request and wait for the response."""
        channel_key = uuid4().hex
        request: dict[str, Any] = {
            "channel_key": channel_key,
            "messages": messages,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_new_tokens": 2000,
            },
        }

        async def _generate() -> dict[str, Any]:
            t0 = time.perf_counter()
            await self._generate_input_channel.put(
                request,
                async_op=True,
            ).async_wait()
            agent_put_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            result = await self._generate_output_channel.get(
                channel_key,
                async_op=True,
            ).async_wait()
            agent_get_s = time.perf_counter() - t1
            if isinstance(result, dict):
                result["agent_put_s"] = agent_put_s
                result["agent_get_s"] = agent_get_s
            return result

        try:
            return asyncio.run(_generate())
        except RuntimeError as exc:
            raise RuntimeError(
                "EngineLLMWrapper: asyncio.run failed (maybe already in async context)."
            ) from exc

    def _parse_result(
        self,
        result: dict[str, Any],
    ) -> tuple[str, Optional[bool], Any]:
        """Parse rollout result into text and metadata."""
        text = result.get("text", "")
        return text, None, result
