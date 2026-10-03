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

"""Fake RealSense, Orbbec, ZED, and OpenCV camera interfaces.

Synthetic frames and controllable device state support lifecycle and image
conversion checks without attached cameras.
"""

from __future__ import annotations

import importlib.machinery
import time
import types
from typing import Any

import numpy as np

from ._fakes import module

#: Camera serials available to multi-camera tests.
SERIALS = ("MOCK0001", "MOCK0002", "MOCK0003")

#: Metres per raw depth count, as a RealSense device reports it.
DEPTH_SCALE = 0.001

#: Raw counts in the near and far halves of the mock depth map. Two distinct
#: values let a test see whether a resize averaged across the step between
#: them, which is what separates nearest-neighbour from interpolation.
DEPTH_NEAR = 500
DEPTH_FAR = 1500
SERIAL = SERIALS[0]


class _Frames:
    def __init__(self, shape, depth):
        self._shape = shape
        self._depth = depth
        self._count = 0

    def _image(self, channels):
        self._count += 1
        frame = np.zeros((*self._shape, channels), dtype=np.uint8)
        frame[0, 0, 0] = self._count % 256
        return frame

    def get_color_frame(self):
        image = self._image(3)
        return types.SimpleNamespace(
            is_video_frame=lambda: True, get_data=lambda: image
        )

    def get_depth_frame(self):
        image = np.full(self._shape, DEPTH_FAR, dtype=np.uint16)
        image[:, : self._shape[1] // 2] = DEPTH_NEAR
        return types.SimpleNamespace(
            is_depth_frame=lambda: True, get_data=lambda: image
        )


def realsense(width: int = 64, height: int = 48) -> types.ModuleType:
    """Return a ``pyrealsense2`` module that yields synthetic frames."""
    opened: list[str] = []

    class Pipeline:
        def __init__(self):
            self.started = False

        def start(self, config):
            self.started = True
            opened.append(config.serial)
            depth_sensor = types.SimpleNamespace(get_depth_scale=lambda: DEPTH_SCALE)
            return types.SimpleNamespace(
                get_device=lambda: types.SimpleNamespace(
                    first_depth_sensor=lambda: depth_sensor
                )
            )

        def stop(self):
            self.started = False

        def wait_for_frames(self):
            if not self.started:
                raise RuntimeError("pipeline not started")
            return _Frames((height, width), depth=True)

    class Config:
        def __init__(self):
            self.serial = None
            self.streams = []

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            self.streams.append(args)

        def disable_all_streams(self):
            self.streams.clear()

    class Align:
        def __init__(self, _stream):
            pass

        def process(self, frames):
            return frames

    devices = [
        types.SimpleNamespace(get_info=lambda _key, serial=serial: serial)
        for serial in SERIALS
    ]
    fake = module(
        "pyrealsense2",
        pipeline=Pipeline,
        config=Config,
        align=Align,
        context=lambda: types.SimpleNamespace(devices=devices),
        camera_info=types.SimpleNamespace(serial_number="serial_number"),
        stream=types.SimpleNamespace(color="color", depth="depth"),
        format=types.SimpleNamespace(bgr8="bgr8", z16="z16"),
    )
    fake.opened = opened
    return fake


def orbbec(
    width: int = 64,
    height: int = 48,
    *,
    color_format: str = "BGR",
    fps: tuple[int, ...] = (15, 30),
) -> types.ModuleType:
    """Return an Orbbec SDK with serial selection and controllable stream faults.

    The module exposes hardware state for tests: ``opened``, ``pipelines``,
    ``fail_start``, ``missing_depth``, ``no_frames``, ``transient_misses``, and
    ``depth_scale_mm``. Native profiles have two sizes at the advertised FPS.
    """
    fake = module("pyorbbecsdk")
    fake.opened = []
    fake.pipelines = []
    fake.fail_start = False
    fake.fail_profiles = False
    fake.missing_depth = False
    fake.no_frames = False
    fake.transient_misses = 0
    fake.depth_scale_mm = 0.25
    fake.distortion_model = "BROWN_CONRADY_K6"
    fake.aligned = 0
    fake.align_filters = []
    fake.depth_reads = 0
    fake.color_bgr = np.array([32, 96, 192], dtype=np.uint8)

    class Profile:
        def __init__(self, sensor: str, size: tuple[int, int], rate: int):
            self.sensor = sensor
            self.size = size
            self.rate = rate

        def as_video_stream_profile(self):
            return self

        def get_width(self):
            return self.size[0]

        def get_height(self):
            return self.size[1]

        def get_fps(self):
            return self.rate

        def get_format(self):
            return color_format if self.sensor == "color" else "Y16"

        def get_distortion(self):
            return types.SimpleNamespace(model=fake.distortion_model)

    class ProfileList:
        def __init__(self, sensor: str):
            self.profiles = [
                Profile(sensor, size, rate)
                for size in ((width * 2, height * 2), (width, height))
                for rate in fps
            ]

        def get_count(self):
            return len(self.profiles)

        def get_stream_profile_by_index(self, index: int):
            return self.profiles[index]

    class DeviceList:
        def get_count(self):
            return len(SERIALS)

        def get_device_serial_number_by_index(self, index: int):
            return SERIALS[index]

        def get_device_by_serial_number(self, serial_number: str):
            if serial_number not in SERIALS:
                raise RuntimeError(f"Device {serial_number!r} not found")
            fake.opened.append(serial_number)
            return types.SimpleNamespace(serial=serial_number)

        def get_device_by_index(self, index: int):
            raise AssertionError("An Orbbec camera must be opened by serial number")

    class Config:
        def __init__(self):
            self.streams = {}
            self.aggregate_mode = None

        def enable_stream(self, profile: Profile):
            self.streams[profile.sensor] = profile

        def set_frame_aggregate_output_mode(self, mode: str):
            self.aggregate_mode = mode

    class VideoFrame:
        def __init__(self, profile: Profile, data: np.ndarray):
            self.profile = profile
            self.data = data

        def get_width(self):
            return self.profile.get_width()

        def get_height(self):
            return self.profile.get_height()

        def get_format(self):
            return self.profile.get_format()

        def get_data(self):
            return self.data

    class DepthFrame(VideoFrame):
        def __init__(self, profile: Profile, data: np.ndarray):
            super().__init__(profile, data)
            self.scale = fake.depth_scale_mm

        def get_depth_scale(self):
            return self.scale

    class Frames:
        def __init__(self, config: Config):
            color = config.streams["color"]
            shape = (color.get_height(), color.get_width())
            bgr = np.broadcast_to(fake.color_bgr, (*shape, 3)).copy()
            if color_format == "RGB":
                data = bgr[..., ::-1].copy()
            elif color_format == "MJPG":
                import cv2

                success, data = cv2.imencode(".jpg", bgr)
                assert success, "The fake camera could not encode its JPEG frame"
            elif color_format == "YUYV":
                # Neutral chroma and Y=128 encode a uniform grey image.
                data = np.full((*shape, 2), 128, dtype=np.uint8)
            else:
                data = bgr
            self.color = VideoFrame(color, data)
            self.depth = None
            if "depth" in config.streams and not fake.missing_depth:
                depth = config.streams["depth"]
                data = np.full(
                    (depth.get_height(), depth.get_width()),
                    DEPTH_FAR,
                    dtype=np.uint16,
                )
                data[:, : depth.get_width() // 2] = DEPTH_NEAR
                data[0, 0] = 0
                self.depth = DepthFrame(depth, data)

        def get_color_frame(self):
            return self.color

        def get_depth_frame(self):
            fake.depth_reads += 1
            return self.depth

        def as_frame_set(self):
            return self

    class Pipeline:
        def __init__(self, device: Any):
            self.device = device
            self.started = False
            self.stops = 0
            self.config = None
            self.wait_timeouts = []
            fake.pipelines.append(self)

        def get_stream_profile_list(self, sensor: str):
            if fake.fail_profiles:
                raise RuntimeError("device profile query failed")
            return ProfileList(sensor)

        def enable_frame_sync(self):
            pass

        def start(self, config: Config):
            self.config = config
            self.started = True
            if fake.fail_start:
                raise RuntimeError("device pipeline start failed")

        def wait_for_frames(self, timeout_ms: int):
            self.wait_timeouts.append(timeout_ms)
            if not self.started:
                raise RuntimeError("device pipeline is not started")
            if fake.no_frames or fake.transient_misses:
                fake.transient_misses = max(0, fake.transient_misses - 1)
                time.sleep(min(timeout_ms / 1000, 0.01))
                return None
            if fake.missing_depth:
                time.sleep(min(timeout_ms / 1000, 0.01))
            return Frames(self.config)

        def stop(self):
            self.started = False
            self.stops += 1

    class AlignFilter:
        def __init__(self, align_to_stream: str):
            assert align_to_stream == "color", "Depth must align to color"
            self.match_target_resolution = True
            self.config = {"TargetDistortion": 0.0}
            fake.align_filters.append(self)

        def set_match_target_resolution(self, enabled: bool):
            self.match_target_resolution = enabled

        def set_config_value(self, name: str, value: float):
            self.config[name] = value

        def process(self, frames: Frames):
            fake.aligned += 1
            expected = float(fake.distortion_model != "NONE")
            if frames.depth is not None and self.config["TargetDistortion"] != expected:
                # Represent the difference between ideal and distorted color
                # coordinates with a one-pixel shift at the depth boundary.
                frames.depth.data = np.roll(frames.depth.data, 1, axis=1)
            return frames

    fake.Context = lambda: types.SimpleNamespace(
        query_devices=DeviceList,
        enable_net_device_enumeration=lambda enabled: None,
    )
    fake.Config = Config
    fake.Pipeline = Pipeline
    fake.AlignFilter = AlignFilter
    fake.OBSensorType = types.SimpleNamespace(
        COLOR_SENSOR="color", DEPTH_SENSOR="depth"
    )
    fake.OBStreamType = types.SimpleNamespace(
        COLOR_STREAM="color", DEPTH_STREAM="depth"
    )
    fake.OBFormat = types.SimpleNamespace(
        RGB="RGB",
        BGR="BGR",
        MJPG="MJPG",
        YUYV="YUYV",
        YUY2="YUYV",
        Y16="Y16",
        Z16="Z16",
    )
    fake.OBFrameAggregateOutputMode = types.SimpleNamespace(
        FULL_FRAME_REQUIRE="full", COLOR_FRAME_REQUIRE="color"
    )
    fake.OBCameraDistortionModel = types.SimpleNamespace(
        NONE="NONE",
        BROWN_CONRADY="BROWN_CONRADY",
        BROWN_CONRADY_K6="BROWN_CONRADY_K6",
        KANNALA_BRANDT4="KANNALA_BRANDT4",
    )
    return fake


#: A ZED reports a numeric serial, and the driver casts it, so the fake's has
#: to be one too.
ZED_SERIAL = "12345678"


def zed(width: int = 64, height: int = 48) -> dict[str, types.ModuleType]:
    """Return a ``pyzed.sl`` module that captures synthetic frames."""
    opened: list[Any] = []

    class Mat:
        def __init__(self):
            self._count = 0

        def get_data(self):
            self._count += 1
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            frame[0, 0, 0] = self._count % 256
            return frame

    class Camera:
        def __init__(self):
            self.opened = False

        def open(self, params):
            self.opened = True
            opened.append(getattr(params, "serial", None))
            return "SUCCESS"

        def grab(self, _runtime):
            return "SUCCESS" if self.opened else "ERROR"

        def retrieve_image(self, mat, _view):
            return mat

        def retrieve_measure(self, mat, _measure):
            return mat

        def close(self):
            self.opened = False

        def get_camera_information(self):
            return types.SimpleNamespace(
                camera_configuration=types.SimpleNamespace(
                    resolution=types.SimpleNamespace(width=width, height=height)
                )
            )

    class InitParameters:
        def __init__(self):
            self.camera_resolution = None
            self.camera_fps = None
            self.depth_mode = None

        def set_from_serial_number(self, serial):
            self.serial = serial

    sl = module(
        "pyzed.sl",
        Camera=Camera,
        Mat=Mat,
        InitParameters=InitParameters,
        RuntimeParameters=lambda: types.SimpleNamespace(),
        ERROR_CODE=types.SimpleNamespace(SUCCESS="SUCCESS"),
        VIEW=types.SimpleNamespace(LEFT="LEFT"),
        MEASURE=types.SimpleNamespace(DEPTH="DEPTH"),
        DEPTH_MODE=types.SimpleNamespace(ULTRA="ULTRA", NONE="NONE"),
        RESOLUTION=types.SimpleNamespace(
            HD2K="HD2K", HD1080="HD1080", HD720="HD720", VGA="VGA"
        ),
    )
    sl.opened = opened
    parent = module("pyzed")
    parent.sl = sl
    return {"pyzed": parent, "pyzed.sl": sl}


#: Native dimensions and format required by the LUMOS driver.
_LUMOS_W = _LUMOS_H = 1280
_YU12 = 0x32315559


def opencv() -> types.ModuleType:
    """Return a ``cv2`` module with a synthetic XVisio V4L2 capture."""
    opens: list[Any] = []
    # Captured before the fake is installed, so delegation reaches the real
    # module rather than importing this one back and recursing.
    try:
        import cv2 as real_cv2
    except ImportError:  # pragma: no cover - a node may have no OpenCV
        real_cv2 = None

    class VideoCapture:
        def __init__(self, path: Any, api: Any = None):
            opens.append(path)
            self.path = path
            self.released = False
            self._properties: dict[int, float] = {}

        def isOpened(self):
            return not self.released

        def set(self, prop, value):
            self._properties[prop] = value
            return True

        def get(self, prop):
            # Report back what the device really supports, not what was asked
            # for: the driver checks these and refuses a mismatch.
            fixed = {
                _Props.FOURCC: float(_YU12),
                _Props.FRAME_WIDTH: float(_LUMOS_W),
                _Props.FRAME_HEIGHT: float(_LUMOS_H),
            }
            return fixed.get(prop, self._properties.get(prop, 0.0))

        def read(self):
            if self.released:
                return False, None
            # One I420 plane set, flat, exactly as V4L2 delivers it.
            return True, np.zeros(_LUMOS_W * _LUMOS_H * 3 // 2, dtype=np.uint8)

        def release(self):
            self.released = True

    class _Props:
        FOURCC = 6
        CONVERT_RGB = 16
        FRAME_WIDTH = 3
        FRAME_HEIGHT = 4
        FPS = 5
        BUFFERSIZE = 38

    def cvtColor(image, code):
        return np.zeros((_LUMOS_H, _LUMOS_W, 3), dtype=np.uint8)

    def resize(image, size, interpolation=None):
        # Environments resize their own observations with this, so it has to
        # be the real thing wherever OpenCV is installed: a stand-in that
        # answers in colour turns a depth map into an image. Without OpenCV,
        # keep at least the shape and dtype the caller asked for.
        if real_cv2 is not None:
            if interpolation is None:
                return real_cv2.resize(image, size)
            return real_cv2.resize(image, size, interpolation=interpolation)
        image = np.asarray(image)
        width, height = size
        shape = (height, width) if image.ndim == 2 else (height, width, image.shape[2])
        return np.zeros(shape, dtype=image.dtype)

    class _OpenCV(types.ModuleType):
        """Delegate to real OpenCV except for device capture."""

        def __getattr__(self, name: str) -> Any:
            if real_cv2 is None:
                raise AttributeError(name)
            return getattr(real_cv2, name)

    fake = _OpenCV("cv2")
    fake.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    for key, value in {
        "CAP_V4L2": 200,
        "CAP_PROP_FOURCC": _Props.FOURCC,
        "CAP_PROP_CONVERT_RGB": _Props.CONVERT_RGB,
        "CAP_PROP_FRAME_WIDTH": _Props.FRAME_WIDTH,
        "CAP_PROP_FRAME_HEIGHT": _Props.FRAME_HEIGHT,
        "CAP_PROP_FPS": _Props.FPS,
        "CAP_PROP_BUFFERSIZE": _Props.BUFFERSIZE,
        "COLOR_YUV2BGR_I420": 101,
        "INTER_AREA": 3,
        "VideoCapture": VideoCapture,
        "VideoWriter_fourcc": lambda *_chars: _YU12,
        "cvtColor": cvtColor,
        "resize": resize,
    }.items():
        setattr(fake, key, value)
    #: What the fake opened, so a test can prove construction touched nothing.
    fake.opens = opens
    return fake


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return fake camera SDKs keyed by import name."""
    made = {"pyrealsense2": realsense(), "pyorbbecsdk": orbbec(), "cv2": opencv()}
    made.update(zed())
    return made
