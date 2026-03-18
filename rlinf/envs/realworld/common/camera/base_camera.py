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

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False


class BaseCamera(ABC):
    """Abstract base class for threaded camera capture.

    Subclasses must implement ``_read_frame`` (hardware-specific frame
    acquisition) and ``_close_device`` (hardware-specific cleanup).
    The threading, queue management, and public API (``open``, ``close``,
    ``get_frame``) are handled here.
    """

    def __init__(self, camera_info: CameraInfo):
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = False

    @property
    def name(self) -> str:
        return self._camera_info.name

    def open(self):
        """Start the background frame-capturing thread."""
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self):
        """Stop the capture thread and release hardware resources."""
        self._frame_capturing_start = False
        if self._frame_capturing_thread.is_alive():
            self._frame_capturing_thread.join()
        self._close_device()

    def get_frame(self, timeout: int = 5) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        Args:
            timeout: Maximum seconds to wait for a frame.
        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Call open() first."
        )
        return self._frame_queue.get(timeout=timeout)

    # ── internal ──────────────────────────────────────────────────────

    def _capture_frames(self):
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            has_frame, frame = self._read_frame()
            if not has_frame:
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        ...

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware-specific resources (pipeline, SDK handle, …)."""
        ...
