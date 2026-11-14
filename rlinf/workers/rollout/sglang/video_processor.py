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

"""Preserve reward-video processor outputs in managed SGLang servers."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import torch

from rlinf.models.embodiment.reward.vlm_reward_utils.video_transport import (
    VIDEO_INPUT_KEY,
    decode_video_inputs,
)


class RewardVideoProcessor:
    """Handle preprocessed reward videos and delegate ordinary requests."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    async def process_mm_data_async(
        self, image_data: Any, input_text: Any, request_obj: Any, *args, **kwargs
    ) -> dict[str, Any]:
        if (
            not image_data
            or not isinstance(image_data[0], dict)
            or VIDEO_INPUT_KEY not in image_data[0]
        ):
            return await self.processor.process_mm_data_async(
                image_data, input_text, request_obj, *args, **kwargs
            )
        if len(image_data) != 1:
            raise ValueError(
                "A reward-video request must contain exactly one encoded sample"
            )
        processor = self.processor
        if processor.hf_config.model_type not in ("qwen3_vl", "qwen3_vl_moe"):
            raise ValueError("Preprocessed reward videos require Qwen3-VL")
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        data = decode_video_inputs(image_data[0])
        ids = data.pop("input_ids").reshape(-1)
        if not torch.equal(
            ids, torch.as_tensor(request_obj.input_ids, dtype=torch.int64)
        ):
            raise ValueError("Request token IDs differ from the encoded video input")
        data.pop("attention_mask", None)
        items = processor.collect_mm_items_from_processor_output(data)
        for item in items:
            token = processor.mm_tokens.get_token_id_by_modality(item.modality)
            item.offsets = processor.get_mm_items_offset(ids, token)
        positions, delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=processor.hf_config.vision_config.spatial_merge_size,
            image_token_id=processor.mm_tokens.image_token_id,
            video_token_id=processor.mm_tokens.video_token_id,
            vision_start_token_id=processor.vision_start_token_id,
            model_type=processor.model_type,
            input_ids=ids.unsqueeze(0),
            image_grid_thw=None,
            video_grid_thw=data["video_grid_thw"],
            tokens_per_second=getattr(
                processor.hf_config.vision_config, "tokens_per_second", None
            ),
            second_per_grid_ts=data.get(
                "second_per_grid_ts", data.get("video_second_per_grid")
            ),
        )
        return {
            "input_ids": ids.tolist(),
            "mm_items": items,
            "im_start_id": processor.IM_START_TOKEN_ID,
            "im_end_id": processor.IM_END_TOKEN_ID,
            "im_token_id": processor.mm_tokens.image_token_id,
            "video_token_id": processor.mm_tokens.video_token_id,
            "mrope_positions": positions.squeeze(1),
            "mrope_position_delta": delta,
        }


def install_reward_video_support(http_server: Any) -> None:
    """Extend the FastAPI lifespan before launch; SGLang owns processor cleanup."""
    if getattr(http_server.app.state, "reward_video_support", False):
        return
    http_server.app.state.reward_video_support = True
    original_lifespan = http_server.app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with original_lifespan(app):
            manager = http_server._global_state.tokenizer_manager
            processor = getattr(manager, "mm_processor", None)
            if processor is not None and not isinstance(
                processor, RewardVideoProcessor
            ):
                manager.mm_processor = RewardVideoProcessor(processor)
            try:
                yield
            finally:
                if processor is not None:
                    manager.mm_processor = processor

    http_server.app.router.lifespan_context = lifespan
