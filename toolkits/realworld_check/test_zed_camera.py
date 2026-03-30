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

"""Check ZED camera connectivity and frame capture.

Usage::

    python toolkits/realworld_check/test_zed_camera.py [--serial SERIAL] [--steps 20]

Requires the Stereolabs ZED SDK (``pyzed``) to be installed.
"""

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="ZED camera hardware check")
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial number of the ZED camera to test. "
        "If not specified, all connected cameras are listed and the first is used.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of frames to capture"
    )
    parser.add_argument("--fps", type=int, default=15, help="Requested camera FPS")
    args = parser.parse_args()

    import pyzed.sl as sl

    devices = sl.Camera.get_device_list()
    if not devices:
        print("[ERROR] No ZED cameras detected.")
        return

    print(f"[INFO] Found {len(devices)} ZED camera(s):")
    for dev in devices:
        print(f"  serial={dev.serial_number}")

    serial = args.serial or str(devices[0].serial_number)
    print(f"\n[INFO] Testing camera serial={serial}")

    camera = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(int(serial))
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = args.fps

    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open ZED camera: {status}")
        return
    print("[INFO] Camera opened successfully.")

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    for step in range(args.steps):
        err = camera.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"  step {step}: grab failed ({err})")
            continue
        camera.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]
        print(
            f"  step {step}: shape={frame.shape}, dtype={frame.dtype}, "
            f"mean={np.mean(frame):.1f}"
        )
        time.sleep(1.0 / args.fps)

    camera.close()
    print("[INFO] ZED camera check completed.")


if __name__ == "__main__":
    main()
