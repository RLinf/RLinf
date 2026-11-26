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

"""Capture explicit-span Qwen3.5 image inputs inside SGLang preprocessing."""

import asyncio
from dataclasses import replace

import torch

from .replay import CAPTURE_KEY, capture_file, save_capture

try:
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    )
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
except ImportError:
    _HAS_QWEN35_PROCESSOR = False
else:
    _HAS_QWEN35_PROCESSOR = True

    class TokenizedQwenVLProcessor(QwenVLImageProcessor):
        """Process only marked image spans for capture requests; leave others native.

        Registered automatically via ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` when
        ``rollout.sglang.capture_multi_modal_inputs`` is enabled.
        """

        models = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]

        async def process_mm_data_async(
            self, image_data, input_text, request_obj, **kwargs
        ):
            params = request_obj.sampling_params or {}
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
                raise ValueError(
                    "Image capture requires a token-ID prompt and raw images."
                )
            path = capture_file(capture["ticket"], request_obj.rid)
            spans = capture["image_spans"]
            if (
                not spans
                or len(spans) != len(image_data or [])
                or not path.parent.is_dir()
            ):
                raise ValueError(
                    "Image capture requires a local request and one span per image."
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
            output = self._place_images(
                input_text, spans, capture["prefix_length"], native
            )
            await asyncio.to_thread(save_capture, path, output)
            return output

        def _place_images(self, ids, spans, prefix, native):
            from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

            if not 0 <= prefix <= len(ids) or len(native.mm_items) != len(spans):
                raise ValueError(
                    f"Invalid image history or native image count: prefix={prefix}, "
                    f"prompt_len={len(ids)}, spans={len(spans)}, mm_items={len(native.mm_items)}."
                )
            expanded, rope_ids, items = [], [], []
            cursor = 0
            for (start, end), item in zip(spans, native.mm_items, strict=True):
                if not (cursor <= start - 1 < start < end < len(ids)) or (
                    ids[start - 1] != self.vision_start_token_id
                    or ids[end] != self.vision_end_token_id
                    or ids[start:end] != [self.IM_TOKEN_ID] * (end - start)
                ):
                    raise ValueError(
                        f"Image span [{start}, {end}) does not match the structured prompt."
                    )
                if len(item.offsets) != 1 or not isinstance(item.feature, torch.Tensor):
                    raise ValueError("Capture requires native per-image pixel tensors.")
                left, right = item.offsets[0]
                width = right - left + 1
                if (start < prefix and (end > prefix or width != end - start)) or (
                    start >= prefix and end - start not in (1, width)
                ):
                    raise ValueError(
                        f"Native image width {width} differs from recorded span [{start}, {end}) "
                        f"with prefix {prefix}."
                    )
                expanded.extend(ids[cursor:start])
                left = len(expanded)
                expanded.extend([self.IM_TOKEN_ID] * width)
                items.append(replace(item, offsets=[(left, len(expanded) - 1)]))
                rope_ids.extend([0] * (start - cursor - 1))
                rope_ids.extend(
                    [self.vision_start_token_id] + [self.IM_TOKEN_ID] * width
                )
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
