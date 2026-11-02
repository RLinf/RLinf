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

"""Frame resizing shared by real-world environments."""

import cv2
import numpy as np


def crop_and_resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Centre-crop a frame to the target aspect ratio, then resize it.

    Camera drivers read at their native resolution, which rarely matches the
    observation space. Cropping rather than stretching keeps objects the shape
    the policy was trained on.

    Args:
        frame: BGR frame of any resolution.
        size: Target ``(width, height)`` in pixels.

    Returns:
        The frame at ``size``, as a new array.
    """
    height, width = frame.shape[:2]
    target_width, target_height = size
    scale = min(width / target_width, height / target_height)
    crop_width = int(target_width * scale)
    crop_height = int(target_height * scale)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return cv2.resize(
        frame[top : top + crop_height, left : left + crop_width],
        size,
        interpolation=cv2.INTER_AREA,
    )
