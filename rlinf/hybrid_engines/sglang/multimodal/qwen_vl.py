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

"""Preserve Qwen3.5 sampled token boundaries with O(1) per-image LRU caching."""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import replace
from typing import Any

import torch
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from .replay import CAPTURE_KEY, capture_file, save_capture

_IMAGE_CACHE_MAX_ITEMS = 512


def _image_cache_key(item: Any) -> bytes | None:
    if isinstance(item, (bytes, bytearray, memoryview)):
        return hashlib.blake2b(item, digest_size=16).digest()
    if isinstance(item, str):
        return hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
    return None


class TokenizedQwenVLProcessor(QwenVLImageProcessor):
    """Expand Qwen3.5 image markers in-place with an LRU cache per image.

    In multi-turn embodied episodes, only newly observed images enter the native
    image processor; historical images in the conversation are reused directly
    from the per-process LRU cache (reducing preprocessing from O(T) to O(1)).
    """

    models = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._item_cache: OrderedDict[bytes, Any] = OrderedDict()
        self._template_output: Any = None

    async def _process_images_cached(
        self, image_data: list[Any], markers: list[int], request_obj: Any, **kwargs
    ):
        keys = [_image_cache_key(img) for img in image_data]
        resolved: list[Any] = [None] * len(image_data)
        missed_indices: list[int] = []
        for idx, key in enumerate(keys):
            if key is not None and key in self._item_cache:
                self._item_cache.move_to_end(key)
                resolved[idx] = self._item_cache[key]
            else:
                missed_indices.append(idx)

        if missed_indices or self._template_output is None:
            batch_indices = missed_indices if missed_indices else [0]
            batch_images = [image_data[i] for i in batch_indices]
            native = await super().process_mm_data_async(
                batch_images,
                self._tokenizer.decode(markers * len(batch_images)),
                request_obj,
                **kwargs,
            )
            self._template_output = native
            for idx, mm_item in zip(batch_indices, native.mm_items, strict=True):
                resolved[idx] = mm_item
                key = keys[idx]
                if key is not None:
                    self._item_cache[key] = mm_item
                    self._item_cache.move_to_end(key)
                    while len(self._item_cache) > _IMAGE_CACHE_MAX_ITEMS:
                        self._item_cache.popitem(last=False)
            if len(batch_indices) == len(image_data):
                return native
            return replace(native, mm_items=resolved)

        return replace(self._template_output, mm_items=resolved)

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
        markers = [
            self.vision_start_token_id,
            self.IM_TOKEN_ID,
            self.vision_end_token_id,
        ]
        native = await self._process_images_cached(
            list(image_data), markers, request_obj, **kwargs
        )
        output = self._place_images(input_text, spans, capture["prefix_length"], native)
        if capture.get("token_budget") is not None:
            context_length, reserve = capture["token_budget"]
            remaining = context_length - len(output.input_ids)
            max_new_tokens = params.get("max_new_tokens")
            output_limit = (
                min(max_new_tokens, remaining)
                if max_new_tokens is not None
                else remaining
            )
            if output_limit < reserve:
                raise ValueError(
                    f"context_length_exceeded: expanded image prompt leaves {output_limit} output tokens; "
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
            new_left = len(expanded)
            expanded.extend([self.IM_TOKEN_ID] * width)
            items.append(replace(item, offsets=[(new_left, len(expanded) - 1)]))
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
