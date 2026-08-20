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

from typing import Any, Optional

from .base import BaseCamera, CameraInfo
from .realsense import RealSenseCamera

__all__ = [
    "BaseCamera",
    "CameraInfo",
    "RealSenseCamera",
    "camera_cls",
    "declare_cameras",
    "create_camera",
]


def camera_cls(camera_type: str) -> type[BaseCamera]:
    """Return the camera class for a backend, imported lazily.

    Supported ``camera_type`` values:

    * ``"realsense"`` / ``"rs"`` -- Intel RealSense (requires ``pyrealsense2``)
    * ``"zed"`` -- Stereolabs ZED (requires the ZED SDK / ``pyzed``)
    * ``"lumos"`` -- LUMOS V4L2 USB camera (requires ``opencv-python``)
    """
    kind = camera_type.lower()
    if kind == "zed":
        from .zed import ZEDCamera

        return ZEDCamera
    if kind in ("realsense", "rs"):
        return RealSenseCamera
    if kind == "lumos":
        from .lumos import LumosCamera

        return LumosCamera
    raise ValueError(
        f"Unsupported camera_type={camera_type!r}. "
        "Supported types: 'realsense', 'zed', 'lumos'."
    )


def create_camera(camera_info: CameraInfo) -> BaseCamera:
    """Build a camera of the backend named by *camera_info*, in this process.

    Prefer :class:`CameraConfig` inside a robot, which declares the camera and
    lets :meth:`Robot.connect` place it on the node it is plugged into.
    """
    return camera_cls(camera_info.camera_type)(camera_info)


def declare_cameras(
    cameras: "Optional[dict[str, Any]]" = None,
    *,
    node_rank: Optional[int] = None,
) -> dict[str, Any]:
    """Declare a camera per descriptor, ready to compose into a robot.

    Each becomes a part placed on ``node_rank`` -- the machine it is plugged
    into -- and opened when the robot connects. Pass ``{name: CameraInfo}``.
    """
    return {
        name: camera_cls(info.camera_type).at(info, node_rank=node_rank)
        for name, info in (cameras or {}).items()
    }
