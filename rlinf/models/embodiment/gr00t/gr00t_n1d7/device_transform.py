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

"""Device-resident Qwen3-VL preprocessing for GR00T N1.7 rollouts.

Mirrors the N1.6 fast path: the chat template is cached per (instruction,
frame count) and the frames reach the batched HF processor as device tensors,
skipping the per-sample PIL conversions. The stock collator already feeds the
image list straight to the processor, so only the per-sample leg changes.
"""

from __future__ import annotations

import numpy as np
import torch
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


class BatchedQwenVLProcessor(Gr00tN1d7Processor):
    """Processor whose VLM leg runs from cached templates and device tensors."""

    _device: torch.device | None = None
    _template_cache: dict[tuple[str, int], str] = {}

    def set_device(self, device) -> None:
        """Send frame preprocessing to `device`; None keeps the host path."""
        self._device = torch.device(device) if device is not None else None

    def _apply_vlm_processing(self, images: np.ndarray, language: str):
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
                        *[{"type": "image"}] * num_images,
                        {"type": "text", "text": language},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            if len(self._template_cache) >= 64:
                self._template_cache.clear()
            self._template_cache[key] = text

        frames = torch.from_numpy(np.ascontiguousarray(images))
        if self._device.type != "npu":
            # CANN's copy kernel rejects the >8-dim intermediates the fast
            # image processor builds on device; NPU keeps the pixels on host
            # and still gains the cached template and batched call.
            frames = frames.to(self._device, non_blocking=True)
        return {"vlm_content": {"text": text, "images": list(frames.unbind(0))}}
