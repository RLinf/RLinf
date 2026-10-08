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

"""Batched, device-resident Eagle preprocessing for GR00T N1.5 rollouts.

Upstream ``GR00TTransform.apply_batch`` splits the batch, converts every frame
to PIL, renders the chat template and runs vision info per sample, then hands
the collected pieces to the Eagle processor. On a rollout host that is hundreds
of milliseconds of Python per call. The subclass keeps the upstream state and
action legs but prepares the Eagle inputs once per batch: the chat template is
cached per (instruction, frame count) and the frames go to the Eagle fast image
processor as device tensors, so the pixel math runs where the model lives.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from gr00t.model.transforms import GR00TTransform


class BatchedEagleTransform(GR00TTransform):
    """GR00TTransform whose Eagle leg runs batched on the model's device.

    With a device set it also absorbs the eval-mode video legs (center crop at
    ``crop_scale``, bilinear-antialias resize to ``video_size``), so raw frames
    reach the device as uint8 and every pixel operation runs there.
    """

    _device: torch.device | None = None
    _template_cache: dict[tuple[str, int], str] = {}
    crop_scale: float = 0.95
    video_size: int = 224

    def set_device(self, device) -> None:
        """Send Eagle pixel preprocessing to `device`; None keeps the host path."""
        self._device = torch.device(device) if device is not None else None

    def _crop_resize(self, frames: torch.Tensor) -> torch.Tensor:
        """Eval-mode video legs on device: center crop then bilinear resize."""
        import torchvision.transforms.functional as TVF

        b, n, c, h, w = frames.shape
        flat = frames.reshape(b * n, c, h, w).float().div_(255.0)
        flat = TVF.center_crop(
            flat, [int(h * self.crop_scale), int(w * self.crop_scale)]
        )
        flat = TVF.resize(
            flat,
            [self.video_size, self.video_size],
            interpolation=TVF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        flat = flat.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
        return flat.reshape(b, n, c, self.video_size, self.video_size)

    def _templated_text(self, language: str, num_images: int) -> str:
        key = (language, num_images)
        text = self._template_cache.get(key)
        if text is None:
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}] * num_images
                    + [{"type": "text", "text": language}],
                }
            ]
            text = self.eagle_processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            if len(self._template_cache) >= 64:
                self._template_cache.clear()
            self._template_cache[key] = text
        return text

    _announced = False

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        import os

        marker = os.environ.get("RLINF_EAGLE_PROBE")
        if marker:
            mode = "host" if self._device is None else f"device:{self._device}"
            with open(marker, "a") as f:
                f.write(f"{os.getpid()} {mode}\n")
        if self._device is None:
            return super().apply_batch(data, batch_size)
        if not BatchedEagleTransform._announced:
            BatchedEagleTransform._announced = True
            logging.getLogger(__name__).info(
                "Eagle device preprocessing active on %s", self._device
            )
            import os

            marker = os.environ.get("RLINF_EAGLE_PROBE")
            if marker:
                with open(marker, "a") as f:
                    f.write(f"{os.getpid()} {self._device}\n")

        batch: dict[str, Any] = {}

        # Eagle leg, batched: [B, T, V, H, W, C] -> per-sample [V*T, C, H, W].
        video = data["video"]
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        frames = np.transpose(video.astype(np.uint8, copy=False), (0, 2, 1, 5, 3, 4))
        frames = frames.reshape(
            batch_size, -1, frames.shape[3], frames.shape[4], frames.shape[5]
        )
        frames_dev = torch.from_numpy(np.ascontiguousarray(frames)).to(
            self._device, non_blocking=True
        )
        if frames_dev.shape[-2] != self.video_size:
            frames_dev = self._crop_resize(frames_dev)
        num_images = frames_dev.shape[1]

        text_list = []
        images = []
        for i in range(batch_size):
            language = self._prepare_language(
                {self._language_key: data[self._language_key][i]}
                if self._language_key is not None
                else {}
            )
            text_list.append(self._templated_text(language, num_images))
            images.extend(frames_dev[i].unbind(0))

        eagle_inputs = self.eagle_processor(
            text=text_list, images=images, return_tensors="pt", padding=True
        )
        for k, v in eagle_inputs.items():
            batch["eagle_" + k] = v

        # State leg, matching upstream apply_single for the eval path.
        states, state_masks = [], []
        for i in range(batch_size):
            sample = {
                k: v[i]
                for k, v in data.items()
                if isinstance(v, (np.ndarray, torch.Tensor)) and k != "video"
            }
            state, state_mask, _ = self._prepare_state(sample)
            states.append(state)
            state_masks.append(state_mask)
        batch["state"] = torch.from_numpy(np.stack(states))
        batch["state_mask"] = torch.from_numpy(np.stack(state_masks))
        batch["embodiment_id"] = torch.full(
            (batch_size,), self.get_embodiment_tag(), dtype=torch.long
        )
        return batch
