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

import os
import subprocess
from typing import Optional

import numpy as np

from rlinf.scheduler import EmbodiedRuntimeCLI

from .base_camera import BaseCamera, CameraInfo


class EmbodiedRuntimeCamera(BaseCamera):
    """Camera capture through the embodied-runtime ``camctr`` CLI."""

    def __init__(self, camera_info: CameraInfo):
        if camera_info.enable_depth:
            raise ValueError("EmbodiedRuntimeCamera does not support depth capture.")
        super().__init__(camera_info)
        self._cli = EmbodiedRuntimeCLI("camctr")
        self._camera_id = self._cli.resolve_camera_id(camera_info.serial_number)
        self._opened = False
        self._watch_process: Optional[subprocess.Popen] = None
        self._buffer = bytearray()
        self._jpeg_scan = 0

    def open(self):
        """Open the managed camera and start its JPEG stream."""
        if self._opened:
            return
        width, height = self._camera_info.resolution
        info = self._cli.run_json("info", self._camera_id).get("camera", {})
        if info.get("state") not in (1, "CAMERA_STATE_CLOSED", "closed"):
            self._cli.run("close", self._camera_id)
        response = self._cli.run_json(
            "open",
            self._camera_id,
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(self._camera_info.fps),
            "--encoding",
            "jpeg",
        )
        if response.get("encoding") != "jpeg":
            self._cli.run("close", self._camera_id)
            raise RuntimeError(
                f"embodied-runtime camera {self._camera_id!r} returned "
                f"incompatible encoding {response.get('encoding')!r}."
            )
        self._opened = True
        self._watch_process = subprocess.Popen(
            [self._cli.executable, "watch", self._camera_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        super().open()

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read and decode the next JPEG frame from ``camctr watch``."""
        import cv2

        if self._watch_process is None or self._watch_process.stdout is None:
            return False, None
        while True:
            end = self._find_jpeg_end()
            if end is not None:
                data = np.frombuffer(self._buffer[:end], dtype=np.uint8)
                del self._buffer[:end]
                self._jpeg_scan = 0
                break
            chunk = os.read(self._watch_process.stdout.fileno(), 65536)
            if not chunk:
                return False, None
            self._buffer.extend(chunk)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return frame is not None, frame

    def _find_jpeg_end(self) -> Optional[int]:
        data = self._buffer
        if self._jpeg_scan == 0:
            jpeg_start = data.find(b"\xff\xd8")
            if jpeg_start < 0:
                if len(data) > 1:
                    del data[:-1]
                return None
            if jpeg_start:
                del data[:jpeg_start]
            self._jpeg_scan = 2

        index = self._jpeg_scan
        while index + 1 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker == 0xD9:
                self._jpeg_scan = index + 2
                return index + 2
            if marker == 0xFF:
                index += 1
                continue
            if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            if marker == 0xDA:
                if index + 4 > len(data):
                    break
                length = int.from_bytes(data[index + 2 : index + 4], "big")
                if length < 2:
                    raise RuntimeError("Invalid JPEG marker length from camctr stream.")
                index += 2 + length
                while index + 1 < len(data):
                    if data[index] != 0xFF:
                        index += 1
                        continue
                    next_byte = data[index + 1]
                    if next_byte == 0xD9:
                        self._jpeg_scan = index + 2
                        return index + 2
                    if next_byte == 0xFF:
                        index += 1
                        continue
                    if next_byte == 0x00 or 0xD0 <= next_byte <= 0xD7:
                        index += 2
                        continue
                    break
                continue
            if marker == 0xD8:
                index += 2
                continue
            if index + 4 > len(data):
                break
            length = int.from_bytes(data[index + 2 : index + 4], "big")
            if length < 2:
                raise RuntimeError("Invalid JPEG marker length from camctr stream.")
            index += 2 + length
        self._jpeg_scan = max(index - 1, 2)
        return None

    def _close_device(self) -> None:
        """Stop the watch client and close the managed camera."""
        if self._watch_process is not None:
            try:
                self._watch_process.terminate()
                try:
                    self._watch_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._watch_process.kill()
                    self._watch_process.wait(timeout=2)
            except OSError:
                pass
            self._watch_process = None
        if self._opened:
            try:
                self._cli.run("close", self._camera_id)
            finally:
                self._opened = False

    @staticmethod
    def get_device_serial_numbers() -> set[str]:
        """Return camera identifiers managed by embodied-runtime."""
        if not EmbodiedRuntimeCLI.is_enabled("camctr"):
            return set()
        return {
            camera.get("serialNumber") or camera["cameraId"]
            for camera in EmbodiedRuntimeCLI("camctr").list_cameras()
        }
