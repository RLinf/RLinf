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

"""Check Robotiq 2F gripper connectivity and basic motion.

Usage::

    python toolkits/realworld_check/test_robotiq_gripper.py --port /dev/ttyUSB0

Requires ``pymodbus`` to be installed.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Robotiq gripper hardware check")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the USB-RS485 adapter (default: /dev/ttyUSB0)",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Modbus baud rate")
    args = parser.parse_args()

    from rlinf.envs.realworld.common.gripper.robotiq_gripper import RobotiqGripper

    print(f"[INFO] Connecting to Robotiq gripper on {args.port} ...")
    gripper = RobotiqGripper(port=args.port, baudrate=args.baudrate)

    if not gripper.is_ready():
        print("[ERROR] Gripper activation failed.")
        return
    print("[INFO] Gripper activated successfully.")

    print(f"[INFO] Current position: {gripper.position:.4f} m")

    print("[INFO] Opening gripper ...")
    gripper.open()
    time.sleep(2.0)
    print(f"  position after open: {gripper.position:.4f} m, is_open={gripper.is_open}")

    print("[INFO] Closing gripper ...")
    gripper.close()
    time.sleep(2.0)
    print(
        f"  position after close: {gripper.position:.4f} m, is_open={gripper.is_open}"
    )

    print("[INFO] Moving to mid position (128) ...")
    gripper.move(128)
    time.sleep(2.0)
    print(f"  position after move: {gripper.position:.4f} m, is_open={gripper.is_open}")

    gripper.cleanup()
    print("[INFO] Robotiq gripper check completed.")


if __name__ == "__main__":
    main()
