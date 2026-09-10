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
OpenCV's built-in RGB conversion, reshapes the packed YUV buffer, and converts
I420 to BGR so its ``uint8`` output matches the other camera backends.

Depth is not available from this V4L2 interface.
"""

import glob
import os
from typing import Any, Optional, Union

import numpy as np

from rlinf.utils.logging import get_logger

from .base import BaseCamera, Camera, CameraInfo

_logger = get_logger()


@Camera.register("lumos")
class LumosCamera(BaseCamera):
    """Camera capture for LUMOS USB cameras (V4L2, I420 stream).

    ``camera_info.serial_number`` may be:

    * a ``/dev/v4l/by-id/`` filename, which is stable across reboots
    * a ``"videoN"`` shorthand resolved to ``/dev/videoN``
    * a numeric string or int interpreted as a V4L2 device index
    """

    SDK = "cv2"

    _NATIVE_W = 1280
    _NATIVE_H = 1280

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)

        if camera_info.enable_depth:
            raise ValueError("LumosCamera does not support depth capture via V4L2.")

        self._out_w, self._out_h = camera_info.resolution
        # XVisio vSLAM requires 1280x1280 YU12 capture; resize in software.
        self._native_w, self._native_h = self._NATIVE_W, self._NATIVE_H
        self._cv2: Any = None

    def _open(self) -> Any:
        """Open the V4L2 device and configure its native stream format."""
        import cv2

        self._cv2 = cv2
        info = self._camera_info
        dev_path: Union[str, int] = self._resolve_device_path(info.serial_number)

        capture = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(
                f"Failed to open LUMOS camera (serial={info.serial_number}, "
                f"dev_path={dev_path})."
            )

        try:
            self._configure(capture, dev_path)
        except Exception:
            # Release the device if configuration fails after opening it.
            capture.release()
            raise
        return capture

    def _configure(self, capture: Any, dev_path: Union[str, int]) -> None:
        """Put the device in YU12 at its native size, or say why it will not go."""
        cv2 = self._cv2
        info = self._camera_info

        expected_fourcc = cv2.VideoWriter_fourcc(*"YU12")
        capture.set(cv2.CAP_PROP_FOURCC, expected_fourcc)
        # Keep OpenCV from silently reinterpreting the I420 buffer.
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._native_w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._native_h)
        capture.set(cv2.CAP_PROP_FPS, info.fps)

        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        if actual_fourcc != expected_fourcc:
            raise RuntimeError(
                f"LUMOS camera (serial={info.serial_number}, dev_path={dev_path}) "
                f"does not support YU12. Actual FOURCC={actual_fourcc:#010x}."
            )

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self._native_w, self._native_h):
            raise RuntimeError(
                f"LUMOS camera (serial={info.serial_number}, dev_path={dev_path}) "
                f"returned resolution {actual_w}x{actual_h}; expected "
                f"{self._native_w}x{self._native_h}."
            )

        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            _logger.warning(
                "Failed to set LUMOS buffer size (serial=%s): %s",
                info.serial_number,
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
        ok, raw = self._device.read()
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

    def _release(self, device: Any) -> None:
        """Release the supplied video capture."""
        if device is not None:
            device.release()

    @classmethod
    def discover(cls) -> set[str]:
        """Stable ``by-id`` identifiers for the V4L2 cameras attached here.

        Falls back to ``videoN`` names when ``/dev/v4l/by-id/`` is unavailable.
        """
        devices = glob.glob("/dev/v4l/by-id/*")
        if not devices:
            devices = glob.glob("/dev/video*")
        return {os.path.basename(device) for device in devices}
