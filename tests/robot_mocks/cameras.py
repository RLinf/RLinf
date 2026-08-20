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

"""Fake RealSense and ZED SDKs.

Frames carry a rising counter in the first pixel, so a test can tell one from
the next and prove the capture thread is running rather than replaying.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np

from ._fakes import module

#: Serials the fake reports. A robot may carry several cameras, and each is
#: opened by serial, so one device would leave the others unopenable.
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
    """A ``pyrealsense2`` that yields frames without a camera."""
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


def zed(width: int = 64, height: int = 48) -> dict[str, types.ModuleType]:
    """A ``pyzed.sl`` that opens and grabs without a camera."""

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

        def open(self, _params):
            self.opened = True
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
        RESOLUTION=types.SimpleNamespace(HD720="HD720", VGA="VGA"),
    )
    parent = module("pyzed")
    parent.sl = sl
    return {"pyzed": parent, "pyzed.sl": sl}


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Every camera SDK, by the name a part imports it as."""
    made = {"pyrealsense2": realsense()}
    made.update(zed())
    return made
