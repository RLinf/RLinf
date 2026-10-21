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

"""Read twenty color frames from every attached RealSense camera.

Run ``python toolkits/realworld_check/test_franka_camera.py`` on the robot host.
Cameras are discovered automatically and streamed one at a time. Each stream
is stopped after its reads, including when a read fails.
"""

import time

import pyrealsense2 as rs


def main() -> None:
    """Discover cameras, read each stream, and release it."""
    serials = sorted(
        device.get_info(rs.camera_info.serial_number) for device in rs.context().devices
    )
    if not serials:
        raise RuntimeError("No RealSense cameras are connected.")

    for serial_number in serials:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            15,
        )
        pipeline.start(config)
        try:
            for step in range(20):
                time.sleep(0.1)
                pipeline.wait_for_frames()
                print(f"{serial_number}: frame {step + 1}/20")
        finally:
            pipeline.stop()


if __name__ == "__main__":
    main()
