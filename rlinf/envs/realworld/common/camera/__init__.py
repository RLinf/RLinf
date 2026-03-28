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

from .base_camera import BaseCamera, CameraInfo
from .realsense_camera import RealSenseCamera

__all__ = [
    "BaseCamera",
    "CameraInfo",
    "RealSenseCamera",
    "create_camera",
]


def create_camera(camera_info: CameraInfo) -> BaseCamera:
    """Factory that instantiates the right camera backend from *camera_info*.

    Supported ``camera_info.camera_type`` values:

    * ``"realsense"`` / ``"rs"`` — Intel RealSense (requires ``pyrealsense2``)
    * ``"zed"`` — Stereolabs ZED (requires the ZED SDK / ``pyzed``)
    """
    camera_type = camera_info.camera_type.lower()
    if camera_type == "zed":
        from .zed_camera import ZEDCamera

        return ZEDCamera(camera_info)
    if camera_type in ("realsense", "rs"):
        return RealSenseCamera(camera_info)
    raise ValueError(
        f"Unsupported camera_type={camera_type!r}. Supported types: 'realsense', 'zed'."
    )
