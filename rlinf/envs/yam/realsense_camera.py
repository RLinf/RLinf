# Copyright 2026 Shirui Chen
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

"""RealSense camera driver using pyrealsense2.

Drop-in replacement for OpencvCamera that avoids V4L2 device scanning.
Uses serial numbers for stable device identification across USB re-plugs.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from yam_realtime.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


@dataclass
class RealsenseCamera(CameraDriver):
    """RealSense camera driver.

    Usage in YAML config::

        camera:
          _target_: rlinf.envs.yam.realsense_camera.RealsenseCamera
          serial_number: "128422272697"
          resolution: [640, 480]
          fps: 30
    """

    serial_number: str = ""
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"RealsenseCamera(serial={self.serial_number!r}, "
            f"name={self.name!r}, resolution={self.resolution}, fps={self.fps})"
        )

    def __post_init__(self):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()

        if self.serial_number:
            config.enable_device(self.serial_number)

        w, h = self.resolution
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, self.fps)

        try:
            profile = self._pipeline.start(config)
            device = profile.get_device()
            actual_serial = device.get_info(rs.camera_info.serial_number)
            logger.info(
                f"RealSense opened: serial={actual_serial}, {w}x{h}@{self.fps}fps"
            )
        except Exception as e:
            sn = self.serial_number or "auto"
            logger.error(f"Failed to open RealSense (serial={sn}): {e}")
            raise

    def read(self) -> CameraData:
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        capture_time_ms = time.time() * 1000
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received from RealSense")
        frame = np.ascontiguousarray(np.asarray(color_frame.get_data()))
        return CameraData(images={"rgb": frame}, timestamp=capture_time_ms)

    def get_camera_info(self) -> dict:
        import pyrealsense2 as rs

        profile = self._pipeline.get_active_profile()
        device = profile.get_device()
        return {
            "camera_type": "realsense",
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
        }

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def read_calibration_data_intrinsics(self) -> dict[str, Any]:
        import pyrealsense2 as rs

        profile = self._pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ]
        )
        D = np.array(intrinsics.coeffs)
        return {"rgb": {"K": K, "D": D}}
