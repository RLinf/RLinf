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

"""LUMOS camera capture via OpenCV's V4L2 backend.

LUMOS cameras expose a raw I420 (YU12) stream over V4L2; this class disables
OpenCV's built-in RGB conversion, reshapes the packed YUV buffer, and does
the I420→BGR conversion manually so the output matches the RealSense / ZED
backends (BGR ``uint8``).

Depth is not available from this V4L2 interface.
"""

import glob
import os
from typing import Optional, Union

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_logger = get_logger()


class LumosCamera(BaseCamera):
    """Camera capture for LUMOS USB cameras (V4L2, I420 stream).

    ``camera_info.serial_number`` may be:

    * a ``/dev/v4l/by-id/`` filename (preferred — stable across reboots)
    * a ``"videoN"`` shorthand resolved to ``/dev/videoN``
    * a numeric string or int interpreted as a V4L2 device index
    """

    _NATIVE_W = 1280
    _NATIVE_H = 1280

    def __init__(self, camera_info: CameraInfo):
        import cv2

        super().__init__(camera_info)
        self._cv2 = cv2

        if camera_info.enable_depth:
            raise ValueError("LumosCamera does not support depth capture via V4L2.")

        self._out_w, self._out_h = camera_info.resolution
        # XVisio vSLAM only streams YU12 at 1280x1280; off-spec hangs at select(). Resize in software.
        self._native_w, self._native_h = self._NATIVE_W, self._NATIVE_H
        dev_path: Union[str, int] = self._resolve_device_path(camera_info.serial_number)

        self._cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open LUMOS camera (serial={camera_info.serial_number}, "
                f"dev_path={dev_path})."
            )

        expected_fourcc = cv2.VideoWriter_fourcc(*"YU12")
        self._cap.set(cv2.CAP_PROP_FOURCC, expected_fourcc)
        # Keep OpenCV from silently reinterpreting the I420 buffer.
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._native_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._native_h)
        self._cap.set(cv2.CAP_PROP_FPS, camera_info.fps)

        actual_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if actual_fourcc != expected_fourcc:
            raise RuntimeError(
                f"LUMOS camera (serial={camera_info.serial_number}, dev_path={dev_path}) "
                f"does not support YU12. Actual FOURCC={actual_fourcc:#010x}."
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self._native_w, self._native_h):
            raise RuntimeError(
                f"LUMOS camera (serial={camera_info.serial_number}, dev_path={dev_path}) "
                f"returned resolution {actual_w}x{actual_h}; expected "
                f"{self._native_w}x{self._native_h}."
            )

        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            _logger.warning(
                "Failed to set LUMOS buffer size (serial=%s): %s",
                self.camera_info.serial_number,
                exc,
            )

    @staticmethod
    def _resolve_device_path(serial_number: Union[str, int]) -> Union[str, int]:
        if isinstance(serial_number, int):
            return serial_number
        if serial_number.startswith("video"):
            return f"/dev/{serial_number}"
        by_id = f"/dev/v4l/by-id/{serial_number}"
        if os.path.exists(by_id):
            return by_id
        try:
            return int(serial_number)
        except ValueError as exc:
            raise ValueError(
                f"Could not resolve LUMOS serial_number={serial_number!r} to a V4L2 device."
            ) from exc

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, None
        try:
            yuv = np.ascontiguousarray(raw).reshape(
                self._native_h * 3 // 2, self._native_w
            )
        except ValueError as exc:
            _logger.warning(
                "Dropping malformed LUMOS frame (serial=%s): %s",
                self.camera_info.serial_number,
                exc,
            )
            return False, None
        bgr = self._cv2.cvtColor(yuv, self._cv2.COLOR_YUV2BGR_I420)
        if (self._native_w, self._native_h) != (self._out_w, self._out_h):
            bgr = self._cv2.resize(
                bgr, (self._out_w, self._out_h), interpolation=self._cv2.INTER_AREA
            )
        return True, bgr

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        """Return stable by-id identifiers for connected V4L2 cameras.

        Falls back to ``videoN`` names when ``/dev/v4l/by-id/`` is unavailable.
        """
        devices = glob.glob("/dev/v4l/by-id/*")
        if devices:
            return [os.path.basename(d) for d in devices]
        return [os.path.basename(v) for v in glob.glob("/dev/video*")]
