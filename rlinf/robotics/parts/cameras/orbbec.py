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

"""USB camera capture using Orbbec SDK v2 Python bindings."""

import time
from typing import Any, Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base import BaseCamera, Camera, CameraInfo

_logger = get_logger()


@Camera.register("orbbec")
class OrbbecCamera(BaseCamera):
    """Capture BGR images and optional aligned depth from one USB camera.

    The serial number is required. Native stream profiles must match the
    requested frame rate. Color uses the native resolution closest to
    ``camera_info.resolution``; depth uses the device's default depth
    resolution, so the depth mode does not follow the output size. Depth is
    aligned to color, both are resized to ``camera_info.resolution``, and
    :meth:`get_observation` returns depth in metres.

    Color-only capture does not enable the depth sensor. Models that require
    external power for depth must have that supply connected when
    ``enable_depth`` is true; missing depth never falls back to color only.
    """

    SDK = "pyorbbecsdk"

    def __init__(self, camera_info: CameraInfo) -> None:
        super().__init__(camera_info)
        if not camera_info.serial_number:
            raise ValueError("An Orbbec camera serial number is required.")
        if camera_info.fps <= 0 or any(size <= 0 for size in camera_info.resolution):
            raise ValueError("Orbbec camera resolution and fps must be positive.")
        self._context = None
        self._pipeline = None
        self._start_attempted = False
        self._align = None
        self._first_frame: Optional[np.ndarray] = None

    @classmethod
    def require_sdk(cls, where: str = "this machine") -> None:
        """Raise unless the Orbbec SDK, which ``install.sh`` omits, is installed."""
        try:
            super().require_sdk(where)
        except ModuleNotFoundError as missing:
            raise ModuleNotFoundError(
                f"{cls.__name__} needs the {cls.SDK!r} module, which is not "
                f"installed on {where}. Install it with "
                "'pip install --no-deps pyorbbecsdk2==2.1.2'."
            ) from missing

    def _open(self) -> Any:
        import pyorbbecsdk as ob

        context, devices, serials = self._query_usb_devices(ob)
        serial = self._camera_info.serial_number
        if serial not in serials:
            raise ValueError(
                f"Orbbec camera {serial!r} is not connected. "
                f"Available USB serial numbers: {sorted(serials)}."
            )

        self._context = context
        try:
            # Construct only the selected device: opening an SDK device can
            # reset its streams even before a pipeline is started.
            pipeline = ob.Pipeline(devices.get_device_by_serial_number(serial))
            self._pipeline = pipeline
            config = ob.Config()
            color_profile = self._select_profile(
                pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
                (
                    ob.OBFormat.BGR,
                    ob.OBFormat.RGB,
                    ob.OBFormat.MJPG,
                    ob.OBFormat.YUYV,
                    ob.OBFormat.YUY2,
                ),
                "color",
                self._camera_info.resolution,
            )
            config.enable_stream(color_profile)
            selected = [color_profile]
            if self._camera_info.enable_depth:
                depth_profiles = pipeline.get_stream_profile_list(
                    ob.OBSensorType.DEPTH_SENSOR
                )
                # Aligned depth takes the color resolution anyway, so the
                # native size only picks the depth mode; keep the default.
                default = depth_profiles.get_default_video_stream_profile()
                depth_profile = self._select_profile(
                    depth_profiles,
                    (ob.OBFormat.Y16, ob.OBFormat.Z16),
                    "depth",
                    (default.get_width(), default.get_height()),
                )
                config.enable_stream(depth_profile)
                selected.append(depth_profile)
                config.set_frame_aggregate_output_mode(
                    ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
                pipeline.enable_frame_sync()
                self._align = ob.AlignFilter(
                    align_to_stream=ob.OBStreamType.COLOR_STREAM
                )
                self._align.set_match_target_resolution(True)
                distortion = color_profile.get_distortion().model
                supported_distortions = (
                    ob.OBCameraDistortionModel.NONE,
                    ob.OBCameraDistortionModel.BROWN_CONRADY,
                    ob.OBCameraDistortionModel.BROWN_CONRADY_K6,
                    ob.OBCameraDistortionModel.KANNALA_BRANDT4,
                )
                if distortion not in supported_distortions:
                    raise ValueError(
                        "Orbbec depth alignment does not support the color "
                        f"distortion model {distortion}."
                    )
                # The filter keeps the original color image. Its depth must
                # therefore target distorted color pixels, not an ideal lens.
                self._align.set_config_value(
                    "TargetDistortion",
                    float(distortion != ob.OBCameraDistortionModel.NONE),
                )

            _logger.info(
                "Orbbec camera %s streams %s.",
                serial,
                " and ".join(self._describe(profile) for profile in selected),
            )
            width, height = self._camera_info.resolution
            native = (color_profile.get_width(), color_profile.get_height())
            if native[0] * height != native[1] * width:
                _logger.warning(
                    "Orbbec camera %s stretches its %dx%d color stream to the "
                    "configured %dx%d resolution.",
                    serial,
                    *native,
                    width,
                    height,
                )

            self._start_attempted = True
            pipeline.start(config)
            # Validate the requested streams before connect() reports success.
            # Startup may take longer than a normal read; no thread exists yet.
            self._first_frame = self._wait_for_frame(timeout=5.0)
        except BaseException as error:
            try:
                self._release(self._pipeline)
            except Exception as cleanup_error:
                _logger.warning(
                    "Orbbec camera %s cleanup failed after startup: %s",
                    serial,
                    cleanup_error,
                )
            if isinstance(error, Exception) and not isinstance(error, ValueError):
                raise RuntimeError(
                    f"Could not start Orbbec camera {serial!r}: {error}. "
                    f"{self._connection_hint()}"
                ) from error
            raise
        return pipeline

    def _select_profile(
        self,
        profiles: Any,
        formats: tuple[Any, ...],
        stream: str,
        size: tuple[int, int],
    ) -> Any:
        """Choose a supported format at the exact fps, nearest to ``size``."""
        width, height = size
        candidates = []
        available = []
        for index in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(
                index
            ).as_video_stream_profile()
            available.append(self._describe(profile))
            w, h = profile.get_width(), profile.get_height()
            fps, pixel_format = profile.get_fps(), profile.get_format()
            if fps == self._camera_info.fps and pixel_format in formats:
                score = (
                    (w - width) ** 2 + (h - height) ** 2,
                    formats.index(pixel_format),
                    w,
                    h,
                )
                candidates.append((score, profile))
        if not candidates:
            raise ValueError(
                f"Orbbec camera {self._camera_info.serial_number!r} has no supported "
                f"{stream} profile at {self._camera_info.fps} fps. "
                f"Available profiles: {', '.join(available) or 'none'}. "
                f"{self._connection_hint()}"
            )
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _describe(profile: Any) -> str:
        return (
            f"{profile.get_width()}x{profile.get_height()}"
            f"@{profile.get_fps()} {profile.get_format()}"
        )

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            return True, frame
        # Short polls let disconnect() stop this thread within its join
        # timeout. A longer stall is get_frame()'s to detect and reopen.
        while True:
            frame = self._poll_frame(timeout_ms=100)
            if frame is not None:
                return True, frame
            if not self._frame_capturing_start:
                return False, None

    def _wait_for_frame(self, timeout: float) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            frame = self._poll_frame(timeout_ms=min(100, remaining_ms))
            if frame is not None:
                return frame
        raise TimeoutError(
            f"Orbbec camera {self._camera_info.serial_number!r} produced no complete "
            f"{'color/depth' if self._camera_info.enable_depth else 'color'} frame "
            f"within {timeout:g} seconds. {self._connection_hint()}"
        )

    def _poll_frame(self, timeout_ms: int) -> Optional[np.ndarray]:
        """Return one complete frame, or ``None`` if none arrives in time."""
        import cv2

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None or frames.get_color_frame() is None:
            return None
        if self._align is not None:
            if frames.get_depth_frame() is None:
                return None
            aligned = self._align.process(frames)
            if aligned is None:
                return None
            frames = aligned.as_frame_set()
            if frames is None or frames.get_color_frame() is None:
                return None
            if frames.get_depth_frame() is None:
                return None

        color = self._color_to_bgr(frames.get_color_frame())
        depth = None
        if self._align is not None:
            depth_frame = frames.get_depth_frame()
            shape = (depth_frame.get_height(), depth_frame.get_width())
            if shape != color.shape[:2]:
                raise RuntimeError(
                    "Orbbec aligned depth does not match the color resolution."
                )
            scale = depth_frame.get_depth_scale()
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(
                    f"Orbbec camera reported an invalid depth scale: {scale}."
                )
            # Scale belongs to this frame, not to later queued frames.
            depth = (
                np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                .reshape(shape)
                .astype(np.float32)
            )
            depth *= scale * 1e-3

        resolution = self._camera_info.resolution
        if color.shape[:2] != resolution[::-1]:
            color = cv2.resize(color, resolution, interpolation=cv2.INTER_AREA)
            if depth is not None:
                depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            return np.concatenate((color, depth[..., None]), axis=-1)
        # A numpy view must not outlive the SDK frame that owns its buffer.
        return color.copy()

    @staticmethod
    def _color_to_bgr(frame: Any) -> np.ndarray:
        """Decode the native color formats accepted during profile selection."""
        import cv2
        import pyorbbecsdk as ob

        width, height = frame.get_width(), frame.get_height()
        pixel_format = frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if pixel_format == ob.OBFormat.BGR:
            return data.reshape(height, width, 3)
        if pixel_format == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if pixel_format == ob.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None or image.shape != (height, width, 3):
                raise ValueError("Orbbec camera returned an invalid MJPG color frame.")
            return image
        if pixel_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
        raise ValueError(f"Unsupported Orbbec color format: {pixel_format}.")

    def _connection_hint(self) -> str:
        hint = "Check the USB connection and stream profiles."
        if self._camera_info.enable_depth:
            hint += (
                " Depth may require external power (12 V for Femto Bolt) "
                "and a USB 3 connection; use enable_depth=False for color-only capture."
            )
        return hint

    def _release(self, device: Any) -> None:
        try:
            if self._start_attempted:
                device.stop()
        finally:
            self._start_attempted = False
            self._first_frame = None
            self._align = None
            self._pipeline = None
            self._context = None

    @staticmethod
    def _query_usb_devices(ob: Any) -> tuple[Any, Any, set[str]]:
        """Return the SDK context, its USB device list, and their serials."""
        context = ob.Context()
        context.enable_net_device_enumeration(False)
        devices = context.query_devices()
        serials = {
            devices.get_device_serial_number_by_index(index)
            for index in range(devices.get_count())
        }
        return context, devices, serials

    @classmethod
    def discover(cls) -> set[str]:
        """Return USB serial numbers without constructing or streaming devices."""
        try:
            import pyorbbecsdk as ob
        except ImportError:
            return set()
        return cls._query_usb_devices(ob)[2]
