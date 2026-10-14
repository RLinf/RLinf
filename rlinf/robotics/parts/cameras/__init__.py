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

"""Camera interfaces and registered hardware backends.

Importing this package registers the built-in drivers. Vendor SDKs are loaded
only when a driver opens or discovers hardware.
"""

from .base import BaseCamera, Camera, CameraInfo

# Import built-in drivers to populate the camera registry.
from .lumos import LumosCamera
from .realsense import RealSenseCamera
from .zed import ZEDCamera

__all__ = [
    "BaseCamera",
    "Camera",
    "CameraInfo",
    "LumosCamera",
    "RealSenseCamera",
    "ZEDCamera",
]
