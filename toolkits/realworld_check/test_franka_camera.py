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

"""Check RealSense camera connectivity and frame capture.

Usage::

    python -m toolkits.realworld_check.test_franka_camera [--serial SERIAL] [--steps 20]
"""

import argparse
import time

import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(description="RealSense camera hardware check")
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial number to test. If omitted, list devices and use the first one.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of frames to capture"
    )
    parser.add_argument("--fps", type=int, default=15, help="Requested color FPS")
    args = parser.parse_args()

    devices = list(rs.context().devices)
    if not devices:
        print("[ERROR] No RealSense cameras detected by pyrealsense2.")
        print("  USB may still show the cameras (lsusb), but the SDK cannot open them.")
        print("  Common fix:")
        print("    1) sudo usermod -aG video,plugdev $USER")
        print("    2) install librealsense udev rules, then: sudo udevadm trigger")
        print("    3) re-login (or newgrp video) and retry")
        print("  Temporary workaround: sudo -E $(which python) -m toolkits.realworld_check.test_franka_camera")
        return

    print(f"[INFO] Found {len(devices)} RealSense camera(s):")
    serials = []
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"  name={name} serial={serial}")

    serial_number = args.serial or serials[0]
    if serial_number not in serials:
        print(f"[ERROR] Serial {serial_number} not in detected list: {serials}")
        return

    print(f"\n[INFO] Testing camera serial={serial_number}")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.fps)
    pipeline.start(config)
    print("[INFO] Pipeline started.")

    try:
        for step in range(args.steps):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            ok = color is not None
            print(f"  step {step}: color={'ok' if ok else 'missing'}")
            time.sleep(0.05)
    finally:
        pipeline.stop()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
