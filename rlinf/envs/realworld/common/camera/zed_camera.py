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

"""ZED camera capture using the Stereolabs ZED SDK (pyzed).

Requires the ZED SDK to be installed system-wide (not pip-installable);
the ``pyzed`` Python bindings are bundled with the SDK.
"""

from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_logger = get_logger()

# Lazily populated after the first ``pyzed`` import.
_ZED_RESOLUTION_MAP: Optional[dict] = None


def _build_resolution_map():
    global _ZED_RESOLUTION_MAP
    if _ZED_RESOLUTION_MAP is not None:
        return
    import pyzed.sl as sl

    _ZED_RESOLUTION_MAP = {
        (2208, 1242): sl.RESOLUTION.HD2K,
        (1920, 1080): sl.RESOLUTION.HD1080,
        (1280, 720): sl.RESOLUTION.HD720,
        (672, 376): sl.RESOLUTION.VGA,
    }


def _find_closest_resolution(target: tuple[int, int]):
    """Map the requested ``(width, height)`` to the nearest ZED preset."""
    _build_resolution_map()
    best = None
    best_dist = float("inf")
    for (w, h), res_enum in _ZED_RESOLUTION_MAP.items():
        dist = abs(w - target[0]) + abs(h - target[1])
        if dist < best_dist:
            best_dist = dist
            best = res_enum
    return best


class ZEDCamera(BaseCamera):
    """Camera capture for Stereolabs ZED cameras.

    The interface is identical to :class:`RealSenseCamera`: ``open``,
    ``close``, ``get_frame`` return BGR ``uint8`` numpy arrays.

    ZED cameras output BGRA by default; this class strips the alpha channel
    to produce BGR, consistent with the RealSense pipeline.
    """

    def __init__(self, camera_info: CameraInfo):
        import pyzed.sl as sl

        super().__init__(camera_info)
        self._sl = sl

        self._camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(camera_info.serial_number))
        init_params.camera_resolution = _find_closest_resolution(camera_info.resolution)
        init_params.camera_fps = camera_info.fps

        if camera_info.enable_depth:
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        status = self._camera.open(init_params)
        if status == sl.ERROR_CODE.SUCCESS:
            pass
        elif "CALIBRATION" in str(status):
            _logger.warning(
                "ZED camera (serial=%s) opened with warning: %s. "
                "Run ZED Calibration tool to resolve.",
                camera_info.serial_number,
                status,
            )
        else:
            raise RuntimeError(
                f"Failed to open ZED camera "
                f"(serial={camera_info.serial_number}): {status}"
            )

        self._image = sl.Mat()
        self._depth = sl.Mat() if camera_info.enable_depth else None
        self._runtime_params = sl.RuntimeParameters()

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._camera.grab(self._runtime_params) != self._sl.ERROR_CODE.SUCCESS:
            return False, None

        self._camera.retrieve_image(self._image, self._sl.VIEW.LEFT)
        # BGRA → BGR (strip alpha to match RealSense output)
        frame = self._image.get_data()[:, :, :3].copy()

        if self._depth is not None:
            self._camera.retrieve_measure(self._depth, self._sl.MEASURE.DEPTH)
            depth_data = self._depth.get_data()
            depth = np.expand_dims(depth_data, axis=2)
            return True, np.concatenate((frame, depth), axis=-1)

        return True, frame

    def _close_device(self) -> None:
        self._camera.close()

    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        """Return serial numbers of all connected ZED cameras."""
        try:
            import pyzed.sl as sl
        except ImportError:
            return []
        devices = sl.Camera.get_device_list()
        return [str(dev.serial_number) for dev in devices]
