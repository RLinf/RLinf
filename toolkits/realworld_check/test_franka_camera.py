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

import time

import pyrealsense2 as rs


def main():
    for device in rs.context().devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(serial_number)
        device.get_info(rs.camera_info.serial_number)

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

    for step in range(20):
        print(step)
        time.sleep(0.1)
        pipeline.wait_for_frames()


if __name__ == "__main__":
    main()
