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

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.base import Features, Observation, RobotPart
from rlinf.utils.logging import get_logger

_logger = get_logger()


class Camera(RobotPart):
    """Observable camera category used by registered camera drivers."""

    @classmethod
    def of(cls, camera_info: "CameraInfo", **placement: Any) -> "Camera":
        """Declare a camera from its descriptor and placement settings."""
        return cls.backend(camera_info.camera_type)(camera_info, **placement)

    @classmethod
    def declare(
        cls,
        cameras: "Optional[Mapping[str, CameraInfo]]" = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, "Camera"]:
        """Declare named cameras for composition into a robot.

        All returned cameras use ``node_rank`` and remain unconnected.
        """
        return {
            name: cls.of(info, node_rank=node_rank)
            for name, info in (cameras or {}).items()
        }

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Whether this camera is delivering frames.

        Separate from :pyattr:`is_connected`: a camera can be open while its
        stream has stalled, which is what an env wants to know before it
        starts an episode.

        Args:
            timeout: Seconds to wait for a frame before reporting not ready.
        """
        return True


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False
    crop_region: Optional[tuple[float, float, float, float]] = None


class BaseCamera(Camera, ABC):
    """Base class for cameras captured on a background thread."""

    def __init__(self, camera_info: CameraInfo) -> None:
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread: Optional[threading.Thread] = None
        self._frame_capturing_start = False

    @property
    def name(self) -> str:
        return self._camera_info.name

    @property
    def camera_info(self) -> CameraInfo:
        """Return the immutable camera connection descriptor."""
        return self._camera_info

    @property
    def is_connected(self) -> bool:
        """Whether the camera is opened and its capture thread is running."""
        return self._device is not None

    @property
    def observation_features(self) -> Features:
        """Describe the raw BGR frame returned by this camera."""
        width, height = self._camera_info.resolution
        channels = 4 if self._camera_info.enable_depth else 3
        return {
            "frame": {
                "shape": (height, width, channels),
                "dtype": "uint16" if self._camera_info.enable_depth else "uint8",
            }
        }

    def _opened(self) -> None:
        """Create a fresh frame queue and start the capture thread.

        Recreating both resources supports disconnect and reconnect without
        retaining stale frames.
        """
        if self._frame_capturing_start:
            return
        self._frame_queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def _closing(self) -> None:
        """Stop capturing, while the camera is still open to be read."""
        self._frame_capturing_start = False
        thread = self._frame_capturing_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._frame_capturing_thread = None

    def reopen(self) -> None:
        """Reconnect the camera on the node that owns it."""
        self.disconnect()
        self.connect()

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Whether a frame can be read within *timeout*."""
        if not self.is_connected:
            return False
        try:
            self.get_frame(timeout=timeout)
        except Exception:
            return False
        return True

    def get_observation(self) -> Observation:
        """Return the latest raw frame under the canonical camera key."""
        return {"frame": self.get_frame()}

    def get_frame(
        self, timeout: float = 5, attempts: int = 1, wait: float = 0.0
    ) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        A camera that stops producing frames is usually recoverable, so a
        failed read reopens the device before the next attempt and before the
        error reaches the caller. Reading is the only place that learns a
        camera has stalled, which is why recovering it lives here rather than
        in every loop that reads one.

        Args:
            timeout: Maximum seconds to wait for a frame, per attempt.
            attempts: How many times to ask before giving up.
            wait: Seconds to settle after reopening, before the next attempt.

        Raises:
            queue.Empty: If no attempt produced a frame.
        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Call connect() first."
        )
        for attempt in range(1, max(attempts, 1) + 1):
            try:
                return self._frame_queue.get(timeout=timeout)
            except queue.Empty:
                get_logger().warning(
                    "Camera %s produced no frame within %.2fs "
                    "(attempt %d of %d); reopening.",
                    self.name,
                    timeout,
                    attempt,
                    max(attempts, 1),
                )
                self.reopen()
                if wait:
                    time.sleep(wait)
        raise queue.Empty(
            f"Camera {self.name} produced no frame after {max(attempts, 1)} "
            "attempts, each followed by a reopen."
        )

    # Internal capture loop

    def _capture_frames(self) -> None:
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            try:
                has_frame, frame = self._read_frame()
            except Exception as e:
                _logger.error(
                    "[%s] _read_frame raised %s: %s; stopping capture thread.",
                    self._camera_info.name,
                    type(e).__name__,
                    e,
                )
                break
            if not has_frame:
                _logger.error(
                    "[%s] _read_frame returned (False, None); stopping capture thread.",
                    self._camera_info.name,
                )
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    @abstractmethod
    def _open(self) -> Any:
        """Open the camera and return its device handle."""

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release the camera handle returned by :meth:`_open`."""
        raise NotImplementedError
