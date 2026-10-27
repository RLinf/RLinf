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

"""Generic colour webcam capture via OpenCV.

A plain UVC webcam (or the built-in camera on macOS) already returns BGR
frames, so this driver is the minimal :class:`BaseCamera`: open, read, release.
It is the simpler counterpart to :class:`~rlinf.robotics.parts.cameras.LumosCamera`,
which has to unwrap LUMOS's packed I420 stream; a generic webcam needs none of
that. Depth is not available from a single colour webcam.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Optional, Union

import numpy as np

from .base import BaseCamera, Camera, CameraInfo


@Camera.register("webcam")
class WebcamCamera(BaseCamera):
    """Camera capture for a generic colour webcam.

    ``camera_info.serial_number`` may be:

    * a numeric string or int interpreted as the device index
    * a ``"videoN"`` shorthand resolved to ``/dev/videoN`` (Linux)
    * a ``/dev/v4l/by-id/`` basename, as reported by :meth:`discover`
    * a full device path such as ``/dev/video0`` or ``/dev/v4l/by-id/...``
    """

    SDK = "cv2"

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("WebcamCamera does not support depth capture.")
        self._out_w, self._out_h = camera_info.resolution
        self._cv2: Any = None

    def _open(self) -> Any:
        """Open the device and ask for the configured resolution and frame rate."""
        import cv2

        self._cv2 = cv2
        info = self._camera_info
        device = self._resolve_device(info.serial_number)
        capture = cv2.VideoCapture(device)
        if not capture.isOpened():
            raise RuntimeError(
                f"Failed to open webcam (serial={info.serial_number!r}, "
                f"device={device!r})."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._out_w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._out_h)
        capture.set(cv2.CAP_PROP_FPS, info.fps)
        return capture

    @staticmethod
    def _resolve_device(serial_number: Union[str, int]) -> Union[str, int]:
        """Map the descriptor's serial to a ``cv2.VideoCapture`` argument."""
        if isinstance(serial_number, int):
            return serial_number
        if serial_number.isdigit():
            return int(serial_number)
        if serial_number.startswith("video"):
            return f"/dev/{serial_number}"
        by_id = f"/dev/v4l/by-id/{serial_number}"
        if os.path.exists(by_id):
            return by_id
        return serial_number

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame, resizing in software when the device disagrees."""
        ok, frame = self._device.read()
        if not ok or frame is None:
            return False, None
        height, width = frame.shape[:2]
        if (width, height) != (self._out_w, self._out_h):
            frame = self._cv2.resize(
                frame, (self._out_w, self._out_h), interpolation=self._cv2.INTER_AREA
            )
        return True, frame

    def _release(self, device: Any) -> None:
        """Release the supplied video capture."""
        if device is not None:
            device.release()

    @classmethod
    def discover(cls) -> set[str]:
        """Return device identifiers for webcams attached to this node.

        On Linux the camera is a V4L2 ``/dev/videoN`` node, which is not always
        ``/dev/video0`` after a device is re-enumerated, so list the nodes
        directly rather than probing index 0..7. Elsewhere, fall back to
        probing a small range of device indices.

        Every identifier returned here is accepted by :meth:`_resolve_device`,
        so a discovered camera can be passed straight back as
        ``camera_info.serial_number``.
        """
        if sys.platform.startswith("linux"):
            devices = glob.glob("/dev/v4l/by-id/*") or glob.glob("/dev/video*")
            return {os.path.basename(device) for device in devices}
        import cv2

        found: set[str] = set()
        for index in range(8):
            identifier = str(index)
            capture = cv2.VideoCapture(cls._resolve_device(identifier))
            if capture.isOpened():
                found.add(identifier)
                capture.release()
        return found
