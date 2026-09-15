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

"""Generic UVC/V4L2 camera backend for ordinary USB cameras."""

from __future__ import annotations

import glob
import os
from typing import Any, Optional, Union

import numpy as np

from .base import BaseCamera, Camera, CameraInfo


@Camera.register("uvc", "opencv", "v4l2")
class UVCCamera(BaseCamera):
    """Capture color frames from a standard UVC/V4L2 camera.

    The device is opened during :meth:`connect`, following the robotics
    connection lifecycle. ``serial_number`` accepts a V4L2 path, ``videoN``
    shorthand, a stable ``/dev/v4l/by-id`` basename, or a numeric index.
    """

    SDK = "cv2"

    def __init__(self, camera_info: CameraInfo) -> None:
        if camera_info.enable_depth:
            raise ValueError("UVCCamera supports color frames only.")
        super().__init__(camera_info)
        self._cv2: Any = None

    @staticmethod
    def _resolve_device_path(serial_number: Union[str, int]) -> Union[str, int]:
        """Resolve a configured camera identifier to a V4L2 device."""
        if isinstance(serial_number, int):
            return serial_number
        if serial_number.startswith("/dev/"):
            return serial_number
        if serial_number.startswith("video"):
            return f"/dev/{serial_number}"
        by_id = f"/dev/v4l/by-id/{serial_number}"
        if os.path.exists(by_id):
            return by_id
        try:
            return int(serial_number)
        except ValueError as error:
            raise ValueError(
                f"Could not resolve UVC camera identifier {serial_number!r}."
            ) from error

    def _open(self) -> Any:
        """Open and configure the V4L2 capture device."""
        import cv2

        self._cv2 = cv2
        info = self.camera_info
        path = self._resolve_device_path(info.serial_number)
        capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(
                f"Failed to open UVC camera (serial={info.serial_number}, path={path})."
            )
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, info.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, info.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, info.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            capture.release()
            raise
        return capture

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one BGR uint8 color frame."""
        ok, frame = self._device.read()
        if not ok or frame is None:
            return False, None
        if frame.ndim == 2:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return False, None
        return True, np.asarray(frame, dtype=np.uint8)

    def _release(self, device: Any) -> None:
        """Release the V4L2 capture handle."""
        if device is not None:
            device.release()

    @classmethod
    def discover(cls) -> set[str]:
        """Return stable V4L2 identifiers visible on this node."""
        by_id_devices = glob.glob("/dev/v4l/by-id/*")
        if by_id_devices:
            # Configurations commonly use the stable by-id basename, while
            # diagnostics and hand-written configs may use the actual video
            # node. Include all spellings accepted by ``_resolve_device_path``.
            identifiers: set[str] = set()
            for device in by_id_devices:
                identifiers.update((device, os.path.basename(device)))
                try:
                    identifiers.add(os.path.realpath(device))
                except OSError:
                    pass
            return identifiers

        # Keep the full device path: it is accepted by _resolve_device_path
        # and therefore must also pass RobotDiscovery.validate_cameras.
        return set(glob.glob("/dev/video*"))
