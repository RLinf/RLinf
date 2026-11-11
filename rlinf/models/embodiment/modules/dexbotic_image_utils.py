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

"""Shared on-device SigLIP image preprocessing for Dexbotic policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def is_uint8_image_batch(value: Any) -> bool:
    """Return whether ``value`` is a ``uint8`` tensor or NumPy array."""
    return (torch.is_tensor(value) and value.dtype == torch.uint8) or (
        isinstance(value, np.ndarray) and value.dtype == np.uint8
    )


def preprocess_siglip_view_batch(
    raw: Any,
    image_processor: Any,
    device: torch.device | str,
    *,
    image_aspect_ratio: str = "pad",
    pad_with_image_mean: bool = False,
) -> torch.Tensor:
    """Run SigLIP square-pad, bicubic resize, rescale, and normalize on ``device``."""
    import torchvision.transforms.functional as TVF

    t = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
    t = t.to(device, non_blocking=True)
    if t.dtype != torch.uint8:
        t = (
            (t * 255).clamp(0, 255).to(torch.uint8)
            if t.max() <= 1
            else t.to(torch.uint8)
        )
    t = t.permute(0, 3, 1, 2)
    h, w = t.shape[-2:]
    if h != w and image_aspect_ratio == "pad":
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        if pad_with_image_mean:
            background = t.new_tensor(
                [int(mean * 255) for mean in image_processor.image_mean]
            )
            padded = (
                background.view(1, -1, 1, 1).expand(t.shape[0], -1, side, side).clone()
            )
        else:
            padded = torch.zeros(
                t.shape[0], t.shape[1], side, side, dtype=t.dtype, device=t.device
            )
        padded[:, :, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w] = t
        t = padded
    size = (image_processor.size["height"], image_processor.size["width"])
    t = TVF.resize(
        t, list(size), interpolation=TVF.InterpolationMode.BICUBIC, antialias=True
    )
    t = t.float() * image_processor.rescale_factor
    return TVF.normalize(t, image_processor.image_mean, image_processor.image_std)


def maybe_preprocess_dexbotic_images_on_device(
    raw_main_images: Any,
    raw_wrist_images: Any,
    *,
    image_processor: Any,
    device: torch.device | str | None,
    dtype: torch.dtype,
    image_aspect_ratio: str = "pad",
    pad_with_image_mean: bool = False,
    required_num_images: int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Preprocess ``uint8`` main/wrist image batches on ``device``, or return ``None``."""
    if (
        device is None
        or not is_uint8_image_batch(raw_main_images)
        or (raw_wrist_images is not None and not is_uint8_image_batch(raw_wrist_images))
    ):
        return None
    views = [
        preprocess_siglip_view_batch(
            raw_main_images,
            image_processor,
            device,
            image_aspect_ratio=image_aspect_ratio,
            pad_with_image_mean=pad_with_image_mean,
        )
    ]
    if raw_wrist_images is not None:
        views.append(
            preprocess_siglip_view_batch(
                raw_wrist_images,
                image_processor,
                device,
                image_aspect_ratio=image_aspect_ratio,
                pad_with_image_mean=pad_with_image_mean,
            )
        )
    images = torch.stack(views, dim=1).to(dtype)
    batch_size, num_views = images.shape[0], images.shape[1]
    if num_views < required_num_images:
        padding = torch.zeros(
            batch_size,
            required_num_images - num_views,
            *images.shape[2:],
            dtype=images.dtype,
            device=device,
        )
        images = torch.cat([images, padding], dim=1)
    image_masks = torch.zeros(
        batch_size, required_num_images, dtype=torch.bool, device=device
    )
    image_masks[:, :num_views] = True
    return images, image_masks
