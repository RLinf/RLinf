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

"""Generic USB Video Class camera capture through OpenCV."""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

from .base import BaseCamera, Camera, CameraInfo

_logger = get_logger()


@Camera.register("uvc")
class UVCCamera(BaseCamera):
    """Capture BGR frames from a V4L2/UVC device."""

    SDK = "cv2"

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("UVCCamera does not support depth capture.")
        self._cv2: Any = None

    def _resolve_device(self) -> str | int:
        serial = self.camera_info.serial_number
        if isinstance(serial, int):
            return serial
        if serial.startswith("video"):
            return f"/dev/{serial}"
        if os.path.exists(serial):
            return serial
        try:
            return int(serial)
        except ValueError as exc:
            raise ValueError(f"Could not resolve UVC device {serial!r}.") from exc

    def _open(self) -> Any:
        import cv2

        self._cv2 = cv2
        info = self.camera_info
        capture = cv2.VideoCapture(self._resolve_device(), cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open UVC camera {info.serial_number!r}.")
        preferred_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        capture.set(cv2.CAP_PROP_FOURCC, preferred_fourcc)
        width, height = info.resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, info.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        if actual_fourcc != preferred_fourcc:
            _logger.warning(
                "UVC camera %r did not accept MJPG; using FOURCC %#010x.",
                info.serial_number,
                actual_fourcc,
            )
        return capture

    def _read_frame(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._device.read()
        if not ok or frame is None:
            return False, None
        return True, np.asarray(frame, dtype=np.uint8)

    def _release(self, device: Any) -> None:
        if device is not None:
            device.release()

    @classmethod
    def discover(cls) -> set[str]:
        """Return V4L2 device names visible on this node."""
        return set(glob.glob("/dev/video*"))
