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

"""Front camera for the Unitree Go2 SDK2 video service."""

from contextlib import ExitStack
from functools import cache
from typing import Any

import numpy as np

from rlinf.robotics.go2_dds import go2_dds_session

from .base import BaseCamera, Camera, CameraInfo

GO2_FRONT_CAMERA_RESOLUTION = (640, 360)


@cache
def _video_client() -> Any:
    # SDK2 keeps a response reader alive and has no public close method.
    from unitree_sdk2py.go2.video.video_client import VideoClient

    client = VideoClient()
    client.SetTimeout(0.5)
    client.Init()
    return client


@Camera.register("unitree_go2")
class Go2FrontCamera(BaseCamera):
    """Read the built-in color camera on an explicitly selected DDS interface."""

    SDK = "unitree_sdk2py"

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("The Go2 front camera provides only color frames.")
        self._dds = ExitStack()

    def _open(self) -> Any:
        with ExitStack() as resources:
            resources.enter_context(
                go2_dds_session(self.camera_info.serial_number, channel="video")
            )
            client = _video_client()
            self._dds = resources.pop_all()
        return client

    def _read_frame(self) -> tuple[bool, np.ndarray]:
        import cv2

        code, data = self._device.GetImageSample()
        if code != 0:
            raise RuntimeError(f"Go2 front camera SDK returned {code}.")
        if not data:
            raise ValueError("Go2 front camera returned an empty image.")
        frame = cv2.imdecode(
            np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise ValueError("Go2 front camera returned an invalid JPEG.")
        return True, cv2.resize(
            frame, self.camera_info.resolution, interpolation=cv2.INTER_AREA
        )

    def _release(self, device: Any) -> None:
        self._dds.close()
