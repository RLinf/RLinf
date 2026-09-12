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

"""Turn a camera's native frame into the square image a policy reads.

Every camera driver delivers BGR at its own resolution. A policy sees RGB at
the size its observation space declares, cropped to the view the camera was
mounted for. Depth goes through the same crop so each pixel still lines up
with the colour one beside it.
"""

from typing import Any, Optional

import cv2
import numpy as np

#: A fractional crop, ``(top, left, bottom, right)``, each in ``0..1``.
CropRegion = tuple[float, float, float, float]


def crop_region(value: Any, *, camera: str, serial: str) -> CropRegion:
    """Check a crop region from a config and return it as floats.

    Args:
        value: ``[top, left, bottom, right]``, each a fraction in ``0..1``.
        camera: The camera's name, for the error message.
        serial: The camera's serial, for the error message.

    Raises:
        ValueError: If the region is not four fractions enclosing an area.
    """
    where = f"crop_region for camera '{camera}' ({serial})"
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid {where}: expected [top, left, bottom, right].")
    try:
        top, left, bottom, right = (float(fraction) for fraction in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {where}: expected numeric values, got {value!r}."
        ) from exc
    if not all(0.0 <= fraction <= 1.0 for fraction in (top, left, bottom, right)):
        raise ValueError(
            f"Invalid {where}: values must be within [0, 1], got {value!r}."
        )
    if bottom <= top or right <= left:
        raise ValueError(
            f"Invalid {where}: expected bottom > top and right > left, got {value!r}."
        )
    return (top, left, bottom, right)


def crop(frame: np.ndarray, region: Optional[CropRegion] = None) -> np.ndarray:
    """Cut ``region`` out of ``frame``, or its centred square when ``None``."""
    height, width = frame.shape[:2]
    if region is not None:
        top, left, bottom, right = region
        return frame[
            int(height * top) : int(height * bottom),
            int(width * left) : int(width * right),
        ]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def policy_frame(
    frame: np.ndarray,
    size: tuple[int, int],
    region: Optional[CropRegion] = None,
    *,
    to_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop and resize a BGR frame, and return it for the policy and the viewer.

    Args:
        frame: The camera's BGR frame, ``(height, width, 3)``.
        size: The ``(height, width)`` the observation space declares.
        region: Where the camera looks at the task; ``None`` for the centre.
        to_rgb: Flip the channels, for a policy trained on RGB. A policy
            trained on the driver's own BGR asks for ``False``.

    Returns:
        The resized frame in the policy's channel order, and the crop in BGR
        as the camera saw it, for display.
    """
    cropped = crop(frame, region)
    resized = cv2.resize(cropped, (size[1], size[0]))
    if to_rgb:
        resized = resized[..., ::-1]
    return np.ascontiguousarray(resized), cropped


def policy_depth(
    depth: np.ndarray, size: tuple[int, int], region: Optional[CropRegion] = None
) -> np.ndarray:
    """Crop and resize a depth map to match :func:`policy_frame`.

    Averaging a depth map invents distances between an object and whatever is
    behind it, so this resamples by nearest neighbour.
    """
    cropped = crop(np.asarray(depth, dtype=np.float32), region)
    return cv2.resize(cropped, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
