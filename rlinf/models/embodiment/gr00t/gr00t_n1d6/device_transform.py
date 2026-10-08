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

"""Device-resident Eagle preprocessing for GR00T N1.6 rollouts.

The upstream processor converts every frame to PIL and renders the chat
template per sample; its collator then re-extracts the frames through
``process_vision_info``. On a rollout host that is the dominant per-call cost.
These subclasses cache the template per (instruction, frame count) and carry
the frames as device tensors straight into the batched HF processor call.
"""

from __future__ import annotations

import numpy as np
import torch
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
    Gr00tN1d6DataCollator,
    Gr00tN1d6Processor,
)


class BatchedEagleCollator(Gr00tN1d6DataCollator):
    """Collator that accepts frames already extracted as tensors."""

    def __call__(self, features):
        for feature in features:
            content = feature.get("vlm_content")
            if content is not None and "conversation" not in content:
                # Frames came through the device path; hand them to the HF
                # processor as-is instead of re-walking a conversation.
                batch = {}
                keys = list(set().union(*(f.keys() for f in features)))
                for key in keys:
                    values = [f[key] for f in features if key in f]
                    if key == "vlm_content":
                        text_list = [v["text"] for v in values]
                        images = [img for v in values for img in v["images"]]
                        vlm_inputs = self.processor(
                            text=text_list,
                            images=images,
                            return_tensors="pt",
                            padding=True,
                        )
                        batch.update(vlm_inputs)
                    else:
                        batch[key] = torch.from_numpy(np.stack(values))
                from transformers.feature_extraction_utils import BatchFeature

                return BatchFeature(data={"inputs": batch})
        return super().__call__(features)


class BatchedEagleProcessor(Gr00tN1d6Processor):
    """Processor whose VLM leg runs from cached templates and device tensors."""

    data_collator_class = BatchedEagleCollator

    _device: torch.device | None = None
    _template_cache: dict[tuple[str, int], str] = {}

    def set_device(self, device) -> None:
        """Send frame preprocessing to `device`; None keeps the host path."""
        self._device = torch.device(device) if device is not None else None

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> dict:
        if self._device is None:
            return super()._apply_vlm_processing(images, language)

        num_images = len(images)
        key = (language, num_images)
        text = self._template_cache.get(key)
        if text is None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": language},
                        *[{"type": "image"}] * num_images,
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            if len(self._template_cache) >= 64:
                self._template_cache.clear()
            self._template_cache[key] = text

        frames = torch.from_numpy(np.ascontiguousarray(images)).to(
            self._device, non_blocking=True
        )
        return {"vlm_content": {"text": text, "images": list(frames.unbind(0))}}
