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

"""Check LUMOS camera connectivity and frame capture.

Usage::

    python toolkits/realworld_check/test_lumos_camera.py [--serial SERIAL] [--steps 20]

Requires ``opencv-python`` (cv2) and a LUMOS V4L2 USB camera.
"""

import argparse
import time

import numpy as np

from rlinf.envs.realworld.common.camera.base_camera import CameraInfo
from rlinf.envs.realworld.common.camera.lumos_camera import LumosCamera


def main():
    parser = argparse.ArgumentParser(description="LUMOS camera hardware check")
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial number (by-id filename or 'videoN') of the LUMOS camera "
        "to test. If not specified, all connected cameras are listed and the "
        "first is used.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of frames to capture"
    )
    parser.add_argument("--fps", type=int, default=15, help="Requested camera FPS")
    args = parser.parse_args()

    devices = LumosCamera.get_device_serial_numbers()
    if not devices:
        print("[ERROR] No LUMOS cameras detected.")
        return

    print(f"[INFO] Found {len(devices)} V4L2 camera device(s):")
    for dev in devices:
        print(f"  serial={dev}")

    serial = args.serial or devices[0]
    print(f"\n[INFO] Testing camera serial={serial}")

    camera_info = CameraInfo(
        name="lumos_check",
        serial_number=serial,
        camera_type="lumos",
        fps=args.fps,
    )
    camera = LumosCamera(camera_info)
    camera.open()
    print("[INFO] Camera opened successfully.")

    for step in range(args.steps):
        frame = camera.get_frame(timeout=5)
        print(
            f"  step {step}: shape={frame.shape}, dtype={frame.dtype}, "
            f"mean={np.mean(frame):.1f}"
        )
        time.sleep(1.0 / args.fps)

    camera.close()
    print("[INFO] LUMOS camera check completed.")


if __name__ == "__main__":
    main()
