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

"""Preserve expanded Qwen image prompts through SGLang preprocessing."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from inspect import isawaitable
from typing import Any

import torch
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from .replay import CAPTURE_KEY, capture_file, save_capture


@dataclass(kw_only=True)
class TokenizedImageInput(BaseMultiModalProcessorOutput):
    """Loaded images with the caller's already expanded prompt IDs."""

    input_ids: list[int]


class TokenizedQwenVLProcessor(QwenVLImageProcessor):
    """Keep sampled token boundaries when processing Qwen3.5 image requests.

    Token-ID requests must contain expanded image pads and raw image data.
    Text requests retain SGLang's normal processor behavior. Register through
    ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``; the shared installation is untouched.
    """

    models = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        params = getattr(request_obj, "sampling_params", None) or {}
        capture = (params.get("custom_params") or {}).get(CAPTURE_KEY)
        if capture is None:
            return await super().process_mm_data_async(
                image_data, input_text, request_obj, **kwargs
            )
        if (
            not isinstance(input_text, list)
            or request_obj.video_data
            or request_obj.audio_data
            or any(isinstance(image, dict) for image in image_data or [])
        ):
            raise ValueError("Image capture requires a token-ID prompt and raw images")
        path = capture_file(capture["ticket"], request_obj.rid)
        spans = capture["image_spans"]
        if not spans or len(spans) != len(image_data) or not path.parent.is_dir():
            raise ValueError(
                "Image capture requires a local request and one span per image"
            )
        # Only real media enter the native image processor. Generated text may
        # literally contain the same special-token IDs as an image delimiter.
        markers = [
            self.vision_start_token_id,
            self.IM_TOKEN_ID,
            self.vision_end_token_id,
        ]
        native = await super().process_mm_data_async(
            image_data,
            self._tokenizer.decode(markers * len(spans)),
            request_obj,
            **kwargs,
        )
        output = self._place_images(input_text, spans, capture["prefix_length"], native)
        if capture.get("token_budget") is not None:
            context_length, reserve = capture["token_budget"]
            remaining = context_length - len(output.input_ids)
            output_limit = min(params["max_new_tokens"], remaining)
            if output_limit < reserve:
                raise ValueError(
                    f"Expanded image prompt leaves {output_limit} output tokens; "
                    f"at least {reserve} are required"
                )
            params["max_new_tokens"] = output_limit
        await asyncio.to_thread(save_capture, path, output)
        return output

    def _place_images(self, ids, spans, prefix, native):
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        if not 0 <= prefix <= len(ids) or len(native.mm_items) != len(spans):
            raise ValueError("Invalid image history or native image count")
        expanded, rope_ids, items = [], [], []
        cursor = 0
        for (start, end), item in zip(spans, native.mm_items, strict=True):
            if not (cursor <= start - 1 < start < end < len(ids)) or (
                ids[start - 1] != self.vision_start_token_id
                or ids[end] != self.vision_end_token_id
                or ids[start:end] != [self.IM_TOKEN_ID] * (end - start)
            ):
                raise ValueError("Image span does not match the structured prompt")
            if len(item.offsets) != 1 or not isinstance(item.feature, torch.Tensor):
                raise ValueError("Capture requires native per-image pixel tensors")
            left, right = item.offsets[0]
            width = right - left + 1
            if (start < prefix and (end > prefix or width != end - start)) or (
                start >= prefix and end - start not in (1, width)
            ):
                raise ValueError("Native image width differs from the recorded span")
            expanded.extend(ids[cursor:start])
            left = len(expanded)
            expanded.extend([self.IM_TOKEN_ID] * width)
            items.append(replace(item, offsets=[(left, len(expanded) - 1)]))
            rope_ids.extend([0] * (start - cursor - 1))
            rope_ids.extend([self.vision_start_token_id] + [self.IM_TOKEN_ID] * width)
            cursor = end
        expanded.extend(ids[cursor:])
        rope_ids.extend([0] * (len(ids) - cursor))
        grid = torch.cat([item.image_grid_thw.reshape(-1, 3) for item in items])
        positions, delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=torch.tensor([rope_ids]),
            image_grid_thw=grid.cpu(),
        )
        return replace(
            native,
            input_ids=expanded,
            mm_items=items,
            mrope_positions=positions.squeeze(1),
            mrope_position_delta=delta,
        )

    def load_mm_data(
        self, prompt: str | list[int], **kwargs: Any
    ) -> BaseMultiModalProcessorOutput | Awaitable[BaseMultiModalProcessorOutput]:
        """Carry token IDs alongside the upstream image-loading result."""
        if not isinstance(prompt, list):
            return super().load_mm_data(prompt=prompt, **kwargs)
        if (
            kwargs.get("video_data")
            or kwargs.get("audio_data")
            or any(isinstance(image, dict) for image in kwargs.get("image_data") or [])
        ):
            raise ValueError(
                "Token-preserving Qwen prompts require raw images; "
                "video, audio and precomputed image inputs are unsupported."
            )
        image_text, image_count = self.mm_tokens.image_token_regex.subn(
            self.mm_tokens.image_token, self._tokenizer.decode(prompt)
        )
        if image_count != len(kwargs.get("image_data") or []):
            raise ValueError(
                f"Prompt contains {image_count} image spans, but received "
                f"{len(kwargs.get('image_data') or [])} images."
            )
        loaded = super().load_mm_data(prompt=image_text, **kwargs)

        def with_prompt_ids(result):
            return TokenizedImageInput(
                input_text=result.input_text,
                images=result.images,
                videos=result.videos,
                audios=result.audios,
                input_ids=prompt,
            )

        # SGLang's asynchronous loader must finish before its fields are read.
        if isawaitable(loaded):

            async def load_with_prompt_ids():
                return with_prompt_ids(await loaded)

            return load_with_prompt_ids()
        return with_prompt_ids(loaded)

    def process_and_combine_mm_data(
        self, base_output: BaseMultiModalProcessorOutput, mm_tokens: Any, **kwargs: Any
    ) -> tuple[list[Any], torch.Tensor, Any]:
        """Pass request-local IDs to image processing before offsets are built."""
        if isinstance(base_output, TokenizedImageInput):
            kwargs["prompt_ids"] = base_output.input_ids
        return super().process_and_combine_mm_data(base_output, mm_tokens, **kwargs)

    def process_mm_data(
        self, input_text: str, prompt_ids: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Use processor pixels while retaining the caller's text/action tokens."""
        processed = super().process_mm_data(input_text=input_text, **kwargs)
        if prompt_ids is None:
            return processed
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
        offsets = self.get_mm_items_offset(input_ids, self.IM_TOKEN_ID)
        actual = [end - start + 1 for start, end in offsets]
        expected = (
            processed["image_grid_thw"].prod(dim=-1) // self.spatial_merge_size**2
        ).tolist()
        if actual != expected or any(
            start == 0
            or end + 1 == len(prompt_ids)
            or prompt_ids[start - 1] != self.vision_start_token_id
            or prompt_ids[end + 1] != self.vision_end_token_id
            for start, end in offsets
        ):
            raise ValueError(
                "Expected expanded image pads enclosed by vision markers: "
                f"processor grid needs {expected}, prompt contains {actual}."
            )
        # Re-tokenizing decoded sampled text can change even equal-length IDs.
        # Offsets and M-RoPE must be derived from this exact sequence instead.
        processed["input_ids"] = input_ids.unsqueeze(0)
        if "attention_mask" in processed:
            processed["attention_mask"] = torch.ones_like(processed["input_ids"])
        if "mm_token_type_ids" in processed:
            processed["mm_token_type_ids"] = (
                processed["input_ids"] == self.IM_TOKEN_ID
            ).long()
        return processed
