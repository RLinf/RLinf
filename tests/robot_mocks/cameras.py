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

"""Fake RealSense, ZED, and OpenCV camera interfaces.

Each generated frame increments its first pixel so tests can detect fresh
capture-thread output.
"""

from __future__ import annotations

import importlib.machinery
import types
from typing import Any

import numpy as np

from ._fakes import module

#: Camera serials available to multi-camera tests.
SERIALS = ("MOCK0001", "MOCK0002", "MOCK0003")
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
        image = np.zeros(self._shape, dtype=np.uint16)
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
            return types.SimpleNamespace(get_device=lambda: None)

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
        width, height = size
        return np.zeros((height, width, 3), dtype=np.uint8)

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
    made = {"pyrealsense2": realsense(), "cv2": opencv()}
    made.update(zed())
    return made
