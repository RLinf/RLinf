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

"""Camera transforms shared by Cosmos3 training and rollout."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

CONCAT_VIEW_DESCRIPTION = (
    "The left half shows the third-person view; the right half "
    "shows the wrist-mounted camera."
)


def as_uint8_bhwc(images: Any, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert one batched RGB camera stream to uint8 BHWC."""
    if isinstance(images, (list, tuple)):
        images = torch.stack([torch.as_tensor(image) for image in images])
    images = torch.as_tensor(images).detach().to(device)
    if images.ndim == 5 and images.shape[1] == 1:
        images = images[:, 0]
    if images.shape[-1] != 3 and images.shape[1] == 3:
        images = images.permute(0, 2, 3, 1)
    if images.is_floating_point() and images.amax() <= 1:
        images = images * 255
    return images.to(torch.uint8).contiguous()


def resize_camera_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize a BHWC camera batch and return BCHW frames."""
    frames = frames.permute(0, 3, 1, 2)
    if frames.shape[-2:] == (image_size, image_size):
        return frames
    return F.interpolate(
        frames.float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).to(torch.uint8)
