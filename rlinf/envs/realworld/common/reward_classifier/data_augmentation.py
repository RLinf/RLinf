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

"""Lightweight image augmentations for reward-classifier training.

All transforms operate on **PyTorch tensors** in ``(B, C, H, W)`` layout
and ``float32`` range ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_crop(
    images: torch.Tensor,
    padding: int = 4,
) -> torch.Tensor:
    """Pad edges then randomly crop back to the original size.

    Args:
        images: ``(B, C, H, W)`` float tensor.
        padding: Pixels to pad on each side (edge-replicate).

    Returns:
        Randomly cropped tensor with the same shape as *images*.
    """
    B, C, H, W = images.shape
    padded = F.pad(images, [padding] * 4, mode="replicate")
    crop_h = torch.randint(0, 2 * padding + 1, (B,))
    crop_w = torch.randint(0, 2 * padding + 1, (B,))
    cropped = torch.stack([
        padded[i, :, crop_h[i]:crop_h[i] + H, crop_w[i]:crop_w[i] + W]
        for i in range(B)
    ])
    return cropped


def random_color_jitter(
    images: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> torch.Tensor:
    """Apply random brightness, contrast, and saturation jitter.

    Args:
        images: ``(B, C, H, W)`` float tensor in ``[0, 1]``.
        brightness: Maximum brightness shift.
        contrast: Maximum contrast scaling deviation from 1.
        saturation: Maximum saturation scaling deviation from 1.

    Returns:
        Augmented tensor clamped to ``[0, 1]``.
    """
    B = images.shape[0]
    device = images.device

    # Brightness
    b_delta = torch.empty(B, 1, 1, 1, device=device).uniform_(-brightness, brightness)
    images = images + b_delta

    # Contrast
    c_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(1 - contrast, 1 + contrast)
    mean = images.mean(dim=(-2, -1), keepdim=True)
    images = c_factor * (images - mean) + mean

    # Saturation (simple: blend with grayscale)
    s_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(1 - saturation, 1 + saturation)
    gray = images[:, :1, :, :] * 0.2989 + images[:, 1:2, :, :] * 0.5870 + images[:, 2:3, :, :] * 0.1140
    gray = gray.expand_as(images)
    images = s_factor * images + (1 - s_factor) * gray

    return images.clamp(0.0, 1.0)


def augment_batch(
    images: torch.Tensor,
    crop_padding: int = 4,
    color_jitter: bool = True,
) -> torch.Tensor:
    """Convenience function: crop + optional colour jitter.

    Args:
        images: ``(B, C, H, W)`` float ``[0, 1]`` tensor.
        crop_padding: Padding for random crop.
        color_jitter: Whether to apply colour jitter.

    Returns:
        Augmented tensor of the same shape.
    """
    images = random_crop(images, padding=crop_padding)
    if color_jitter:
        images = random_color_jitter(images)
    return images
