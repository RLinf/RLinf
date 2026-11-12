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

"""Colour capture from Linux V4L2 cameras through OpenCV."""

import glob
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseCamera, Camera, CameraInfo


@Camera.register("v4l2")
class V4L2Camera(BaseCamera):
    """Capture BGR frames from a V4L2 video node.

    ``serial_number`` accepts a full ``/dev/videoN`` path or a stable
    ``/dev/v4l/by-path/...`` symlink. A by-path link is recommended when
    identical cameras report the same USB serial number.
    """

    SDK = "cv2"

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("V4L2Camera does not support depth capture.")

    def _open(self) -> Any:
        import cv2

        reference = str(self._camera_info.serial_number)
        capture = cv2.VideoCapture(reference, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open V4L2 camera {reference!r}.")
        width, height = self._camera_info.resolution
        # Compressed transport lets several cameras share a USB 2.0 hub.
        # Unsupported cameras retain their current pixel format.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, self._camera_info.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        import cv2

        success, frame = self._device.read()
        if not success or frame is None:
            return False, None
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError(
                f"V4L2 camera {self.name!r} returned an unsupported frame "
                f"with shape {frame.shape} and dtype {frame.dtype}."
            )
        width, height = self._camera_info.resolution
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return True, np.ascontiguousarray(frame)

    def _release(self, device: Any) -> None:
        device.release()

    @classmethod
    def discover(cls) -> set[str]:
        """Return one stable V4L2 capture path per USB camera."""
        by_path = {
            path
            for path in glob.glob("/dev/v4l/by-path/*-usb-*-video-index0")
            if Path(path).exists()
        }
        if by_path:
            return by_path
        return {
            path
            for path in glob.glob("/dev/video[0-9]*")
            if cls._is_primary_video_node(path)
        }

    @staticmethod
    def _is_primary_video_node(path: str) -> bool:
        index = Path("/sys/class/video4linux") / Path(path).name / "index"
        try:
            return index.read_text().strip() == "0"
        except OSError:
            return False
