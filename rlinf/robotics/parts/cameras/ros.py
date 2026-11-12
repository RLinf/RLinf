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

"""Read camera frames from an existing ROS 1 image publisher."""

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlinf.robotics.parts.base import Features, Observation
from rlinf.utils.logging import get_logger

from .base import Camera, CameraInfo

_NODE_LOCK = threading.Lock()


@dataclass
class ROSCameraInfo(CameraInfo):
    """Describe a ROS image stream; ``serial_number`` holds its topic name."""

    camera_type: str = "ros"
    message_type: str = "CompressedImage"
    frame_timeout_s: float = 1.0
    stale_timeout_s: float = 2.0


@Camera.register("ros")
class ROSCamera(Camera):
    """Subscribe to BGR frames without starting or stopping the ROS master."""

    SDK = "rospy"

    def __init__(self, camera_info: ROSCameraInfo) -> None:
        if not isinstance(camera_info, ROSCameraInfo):
            raise TypeError("ROSCamera requires a ROSCameraInfo descriptor.")
        if not camera_info.serial_number:
            raise ValueError("ROSCamera requires a nonempty topic in serial_number.")
        if camera_info.message_type not in {"Image", "CompressedImage"}:
            raise ValueError(
                "Expected ROS Image or CompressedImage, "
                f"got {camera_info.message_type!r}."
            )
        if camera_info.enable_depth or camera_info.crop_region is not None:
            raise ValueError(
                "ROSCamera supports RGB streams without depth or cropping."
            )
        if len(camera_info.resolution) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in camera_info.resolution
        ):
            raise ValueError(
                "ROSCamera resolution must contain positive width and height."
            )
        for name in ("frame_timeout_s", "stale_timeout_s"):
            value = getattr(camera_info, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"ROSCamera {name} must be finite and positive.")
        self._camera_info = camera_info
        self._condition = threading.Condition()
        self._generation: object | None = None
        self._latest: tuple[float, np.ndarray] | None = None
        self._error: Exception | None = None
        self._cv2: Any = None

    @property
    def camera_info(self) -> ROSCameraInfo:
        """Return the stream descriptor used to declare this camera."""
        return self._camera_info

    @property
    def observation_features(self) -> Features:
        """Describe the BGR uint8 frame returned by this camera."""
        width, height = self._camera_info.resolution
        return {"frame": {"shape": (height, width, 3), "dtype": "uint8"}}

    def _open(self) -> Any:
        import cv2
        import rospy
        from sensor_msgs import msg

        self._cv2 = cv2
        with _NODE_LOCK:
            if not rospy.core.is_initialized():
                rospy.init_node("rlinf_camera", anonymous=True, disable_signals=True)
        generation = object()
        with self._condition:
            self._generation = generation
            self._latest = None
            self._error = None
        try:
            return rospy.Subscriber(
                self._camera_info.serial_number,
                getattr(msg, self._camera_info.message_type),
                lambda message: self._receive(message, generation),
                queue_size=1,
                buff_size=16 * 1024 * 1024,
            )
        except BaseException:
            with self._condition:
                self._generation = None
                self._latest = None
                self._condition.notify_all()
            raise

    def _release(self, device: Any) -> None:
        with self._condition:
            self._generation = None
            self._latest = None
            self._error = None
            self._condition.notify_all()
        device.unregister()

    def _decode(self, message: Any) -> np.ndarray:
        if self._camera_info.message_type == "CompressedImage":
            frame = self._cv2.imdecode(
                np.frombuffer(message.data, dtype=np.uint8), self._cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError("Expected a decodable compressed RGB image.")
        else:
            encoding = message.encoding.lower()
            if encoding not in {"bgr8", "rgb8", "mono8"}:
                raise ValueError(f"Expected bgr8, rgb8 or mono8, got {encoding!r}.")
            height, width, step = (
                int(message.height),
                int(message.width),
                int(message.step),
            )
            channels = 1 if encoding == "mono8" else 3
            data = np.frombuffer(message.data, dtype=np.uint8)
            if (
                min(height, width) <= 0
                or step < width * channels
                or data.size != height * step
            ):
                raise ValueError(
                    "ROS image dimensions, row stride and payload disagree."
                )
            frame = data.reshape(height, step)[:, : width * channels]
            frame = np.ascontiguousarray(frame.reshape(height, width, channels))
            if encoding == "rgb8":
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            elif encoding == "mono8":
                frame = self._cv2.cvtColor(frame[..., 0], self._cv2.COLOR_GRAY2BGR)
        width, height = self._camera_info.resolution
        if frame.shape[:2] != (height, width):
            frame = self._cv2.resize(frame, (width, height))
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _receive(self, message: Any, generation: object) -> None:
        received_at = time.monotonic()
        with self._condition:
            if generation is not self._generation:
                return
        try:
            frame = self._decode(message)
        except Exception as exc:
            with self._condition:
                if generation is not self._generation:
                    return
                self._latest = None
                self._error = exc
                self._condition.notify_all()
            get_logger().warning(
                "Invalid frame on ROS topic %s: %s",
                self._camera_info.serial_number,
                exc,
            )
            return
        with self._condition:
            if generation is self._generation:
                self._latest = received_at, frame
                self._error = None
                self._condition.notify_all()

    def get_observation(self, timeout: float | None = None) -> Observation:
        """Wait for a fresh frame, failing on missing or malformed input.

        Args:
            timeout: Maximum wait in seconds; defaults to the stream descriptor.
        """
        timeout = self._camera_info.frame_timeout_s if timeout is None else timeout
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError("ROSCamera timeout must be finite and nonnegative.")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if not self.is_connected or self._generation is None:
                    raise RuntimeError(
                        "ROSCamera is disconnected; call connect() first."
                    )
                if self._error is not None:
                    raise RuntimeError(
                        f"Invalid image on ROS topic {self._camera_info.serial_number!r}."
                    ) from self._error
                now = time.monotonic()
                if self._latest is not None:
                    received_at, frame = self._latest
                    if now - received_at <= self._camera_info.stale_timeout_s:
                        return {"frame": frame.copy()}
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(
                        f"No fresh image on ROS topic {self._camera_info.serial_number!r} "
                        f"within {timeout:.3f}s."
                    )
                self._condition.wait(remaining)

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Return whether a fresh image is available within the timeout."""
        try:
            self.get_observation(timeout=timeout)
        except (RuntimeError, TimeoutError):
            return False
        return True
