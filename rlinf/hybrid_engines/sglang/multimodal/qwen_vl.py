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

"""Isolate literal visual special tokens inside SGLang using explicit spans."""

import os
from collections.abc import Sequence
from dataclasses import is_dataclass, replace

import torch

IMAGE_SPANS_PARAM = "rlinf_image_spans"
EXTERNAL_MM_PROCESSOR_PACKAGE = "rlinf.hybrid_engines.sglang.multimodal"
MIN_MULTIMODAL_SGLANG_VERSION = "0.5.12.post1"


def ensure_multimodal_processor_env(sglang_version: object) -> None:
    """Validate SGLang version and register the RLinf multimodal processor."""
    from packaging.version import parse

    if parse(str(sglang_version)) < parse(MIN_MULTIMODAL_SGLANG_VERSION):
        raise ValueError(
            f"Aligned multimodal rollouts require SGLang >= {MIN_MULTIMODAL_SGLANG_VERSION}, "
            f"got {sglang_version}."
        )
    current = os.environ.get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE")
    if not current:
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = (
            EXTERNAL_MM_PROCESSOR_PACKAGE
        )
    elif current != EXTERNAL_MM_PROCESSOR_PACKAGE:
        raise ValueError(
            f"SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE is set to {current!r}, "
            f"but aligned multimodal rollouts require {EXTERNAL_MM_PROCESSOR_PACKAGE!r}."
        )


def validate_image_spans(
    tokens: Sequence[int],
    spans: Sequence[Sequence[int]],
    *,
    vision_start_id: int,
    image_pad_id: int,
    vision_end_id: int,
    image_count: int,
) -> list[tuple[int, int]]:
    """Validate ordered, disjoint half-open ``[start, end)`` image pad spans."""
    if not spans or len(spans) != image_count:
        raise ValueError(
            "Aligned multimodal rollouts require one explicit image span per image: "
            f"got {len(spans) if spans else 0} spans for {image_count} images."
        )
    validated: list[tuple[int, int]] = []
    cursor = 0
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"Expected [start, end) image span, got {span!r}.")
        start, end = span
        if (
            type(start) is not int
            or type(end) is not int
            or not (cursor <= start - 1 < start < end < len(tokens))
        ):
            raise ValueError(
                f"Image span [{start}, {end}) must be ordered, disjoint, and bounded "
                f"inside prompt length {len(tokens)} (cursor={cursor})."
            )
        if (
            tokens[start - 1] != vision_start_id
            or tokens[end] != vision_end_id
            or list(tokens[start:end]) != [image_pad_id] * (end - start)
        ):
            raise ValueError(
                f"Image span [{start}, {end}) does not match the structured prompt markers."
            )
        validated.append((start, end))
        cursor = end + 1
    return validated


def _replace_fields(obj, **changes):
    """Copy a dataclass or msgspec.Struct (SGLang >= 0.5.19) with updated fields."""
    if is_dataclass(obj):
        return replace(obj, **changes)
    import msgspec

    if isinstance(obj, msgspec.Struct):
        return msgspec.structs.replace(obj, **changes)
    raise TypeError(
        f"Unsupported multimodal container type for replacement: {type(obj).__name__}"
    )


try:
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
except ImportError:
    _HAS_QWEN_VL_PROCESSOR = False
else:
    _HAS_QWEN_VL_PROCESSOR = True
    _SUPPORTED_QWEN_MODEL_NAMES = {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
    _QWEN_MODELS = [
        model
        for model in QwenVLImageProcessor.models
        if model.__name__ in _SUPPORTED_QWEN_MODEL_NAMES
    ]

    class AlignedQwenVLProcessor(QwenVLImageProcessor):
        """Process only marked image spans so literal media tokens stay plain text.

        Registered automatically via ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` when
        ``rollout.sglang.align_multimodal_inputs`` is enabled. Does not export or
        write pixel tensors back to rollout workers.
        """

        models = _QWEN_MODELS
        gpu_image_decode = False

        async def process_mm_data_async(
            self, image_data, input_text, request_obj, **kwargs
        ):
            params = getattr(request_obj, "sampling_params", None) or {}
            spans = (params.get("custom_params") or {}).get(IMAGE_SPANS_PARAM)
            if spans is None:
                return await super().process_mm_data_async(
                    image_data, input_text, request_obj, **kwargs
                )
            if (
                not isinstance(input_text, list)
                or request_obj.video_data
                or request_obj.audio_data
                or any(isinstance(image, dict) for image in image_data or [])
            ):
                raise ValueError(
                    "Span-aligned requests require a token-ID prompt and raw images."
                )
            validated = validate_image_spans(
                input_text,
                spans,
                vision_start_id=self.vision_start_token_id,
                image_pad_id=self.IM_TOKEN_ID,
                vision_end_id=self.vision_end_token_id,
                image_count=len(image_data or []),
            )
            markers = [
                self.vision_start_token_id,
                self.IM_TOKEN_ID,
                self.vision_end_token_id,
            ]
            native = await super().process_mm_data_async(
                image_data,
                self._tokenizer.decode(markers * len(validated)),
                request_obj,
                **kwargs,
            )
            return self._place_images(input_text, validated, native)

        def _place_images(self, ids, spans, native):
            from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

            if len(native.mm_items) != len(spans):
                raise ValueError(
                    f"Processed image count mismatch: prompt has {len(spans)} spans, "
                    f"SGLang produced {len(native.mm_items)} mm_items."
                )
            items = []
            rope_ids = []
            cursor = 0
            for (start, end), item in zip(spans, native.mm_items, strict=True):
                if not item.offsets or len(item.offsets) != 1:
                    raise ValueError("Aligned multimodal items must carry one offset.")
                left, right = item.offsets[0]
                width = right - left + 1
                if width != end - start:
                    raise ValueError(
                        f"SGLang image patch width ({width}) differs from prompt span "
                        f"[{start}, {end}) width ({end - start}); ensure RLinf and SGLang "
                        "share the same image size policy."
                    )
                item_updates = {"offsets": [(start, end - 1)]}
                specific = getattr(item, "model_specific_data", None)
                if isinstance(specific, dict):
                    item_updates["model_specific_data"] = dict(specific)
                items.append(_replace_fields(item, **item_updates))
                rope_ids.extend([0] * (start - cursor - 1))
                rope_ids.extend(
                    [self.vision_start_token_id] + [self.IM_TOKEN_ID] * width
                )
                cursor = end
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
            updates = {
                "input_ids": list(ids),
                "mm_items": items,
                "mrope_positions": positions.squeeze(1),
                "mrope_position_delta": delta,
            }
            if hasattr(native, "padded_input_ids") and hasattr(
                type(native), "build_padded_input_ids"
            ):
                updates["padded_input_ids"] = type(native).build_padded_input_ids(
                    list(ids), items
                )
            return _replace_fields(native, **updates)
