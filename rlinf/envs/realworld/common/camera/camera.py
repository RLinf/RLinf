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
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraInfo:
    name: str
    serial_number: str
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False


class Camera:
    """Camera class for capturing images from Intel RealSense cameras and storing them in a queue. This is adapted from SERL's RSCapture class.
    For RealSense usage, see https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/quick_start_live.ipynb.
    """

    def __init__(
        self,
        camera_info: CameraInfo,
    ):
        import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API

        self._camera_info = camera_info
        self._device_info = {}
        for device in rs.context().devices:
            self._device_info[device.get_info(rs.camera_info.serial_number)] = device
        assert camera_info.serial_number in self._device_info.keys(), (
            f"{self._device_info.keys()=}"
        )

        self._serial_number = camera_info.serial_number
        self._device = self._device_info[self._serial_number]
        self._enable_depth = camera_info.enable_depth

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self._serial_number)
        self._config.enable_stream(
            rs.stream.color,
            camera_info.resolution[0],
            camera_info.resolution[1],
            rs.format.bgr8,
            camera_info.fps,
        )
        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth,
                camera_info.resolution[0],
                camera_info.resolution[1],
                rs.format.z16,
                camera_info.fps,
            )
        self.profile = self._pipeline.start(self._config)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self._align = rs.align(rs.stream.color)

        # Create a queue to store captured frames
        self._frame_queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = False

    @property
    def name(self):
        return self._camera_info.name

    def open(self):
        """Start the frame capturing thread."""
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self):
        """Stop the frame capturing thread and close the camera."""
        self._frame_capturing_start = False
        self._frame_capturing_thread.join()
        self._pipeline.stop()
        self._config.disable_all_streams()

    def get_frame(self, timeout: int = 5):
        """Get the latest frame from the frame queue. The frame is in RGB format.

        Args:
            timeout (int): The maximum time to wait for a frame (in seconds).

        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Cannot get frame."
        )
        return self._frame_queue.get(timeout=timeout)

    def _capture_frames(self):
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            has_frame, frame = self._read_frame()
            if not has_frame:
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()  # discard previous frame
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    def _read_frame(self):
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self._enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            frame = np.asarray(color_frame.get_data())
            if self._enable_depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((frame, depth), axis=-1)
            else:
                return True, frame
        else:
            return False, None
