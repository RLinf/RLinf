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

"""Shared helpers for processing RGB-D camera frames in realworld envs.

These operate on the camera abstractions (``CameraInfo``/``BaseCamera``) plus
plain NumPy arrays, so they are reusable by any realworld environment (Franka,
DoSW1, GIM Arm, …) rather than being specific to one robot.
"""

from typing import Optional

import cv2
import numpy as np

from .base_camera import BaseCamera, CameraInfo, supports_depth


def derive_enable_depth(camera_type: str, enable_camera_depth: bool) -> bool:
    """Whether a camera of *camera_type* should capture depth frames."""
    return bool(enable_camera_depth) and supports_depth(camera_type)


def validate_depth_cameras(
    camera_infos: list[CameraInfo], enable_camera_depth: bool
) -> None:
    """Raise ``ValueError`` if depth is requested but no camera supports it."""
    if not enable_camera_depth:
        return
    if any(info.enable_depth for info in camera_infos):
        return
    camera_types = sorted({info.camera_type for info in camera_infos})
    raise ValueError(
        "enable_camera_depth=True, but none of the configured cameras "
        f"support depth (camera_types={camera_types!r})."
    )


def split_rgb_depth(
    camera: BaseCamera,
    frame: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Split a backend frame into BGR ``uint8`` and metric depth, if present.

    Backends return either ``(H, W, 3)`` BGR or ``(H, W, 4)`` BGR+depth with
    the depth channel packed last. Depth is converted to meters using the
    camera's ``depth_scale``.
    """
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(
            f"Camera {camera._camera_info.name} returned invalid frame "
            f"shape {frame.shape}; expected HxWxC with C>=3."
        )
    color = np.asarray(frame[..., :3])
    if color.dtype != np.uint8:
        color = np.clip(color, 0, 255).astype(np.uint8)
    depth = None
    if frame.shape[-1] >= 4:
        depth = np.asarray(frame[..., 3], dtype=np.float32) * float(camera.depth_scale)
    return color, depth


def crop_bounds(
    *,
    width: int,
    height: int,
    crop_region: Optional[tuple[float, float, float, float]] = None,
    default_square_crop: bool = True,
) -> tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` pixel crop bounds.

    When *crop_region* is ``None`` and *default_square_crop* is true, a
    center-square crop is used; otherwise the full resolution is preserved.
    """
    if crop_region is not None:
        top, left, bottom, right = crop_region
        return (
            int(width * left),
            int(height * top),
            int(width * right),
            int(height * bottom),
        )
    if not default_square_crop:
        return 0, 0, int(width), int(height)
    crop_size = min(height, width)
    x1 = (width - crop_size) // 2
    y1 = (height - crop_size) // 2
    return x1, y1, x1 + crop_size, y1 + crop_size


def crop_frame(
    frame: np.ndarray,
    reshape_size: tuple[int, int],
    crop_region: Optional[tuple[float, float, float, float]] = None,
    resize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop *frame* and (optionally) resize, returning ``(cropped, resized)``."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_bounds(
        width=w,
        height=h,
        crop_region=crop_region,
        default_square_crop=resize,
    )
    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, reshape_size) if resize else cropped
    return cropped, resized


def crop_depth_frame(
    depth: np.ndarray,
    reshape_size: tuple[int, int],
    crop_region: Optional[tuple[float, float, float, float]] = None,
    resize: bool = True,
) -> np.ndarray:
    """Crop and (optionally) resize a depth map using nearest-neighbor."""
    h, w = depth.shape
    x1, y1, x2, y2 = crop_bounds(
        width=w,
        height=h,
        crop_region=crop_region,
        default_square_crop=resize,
    )
    cropped = depth[y1:y2, x1:x2]
    if resize:
        return cv2.resize(cropped, reshape_size, interpolation=cv2.INTER_NEAREST)
    return cropped
