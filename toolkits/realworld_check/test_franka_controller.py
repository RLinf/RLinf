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


import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
from rlinf.robotics.parts.end_effectors import EndEffector


def _parse_args():
    parser = argparse.ArgumentParser(description="Check Franka controller state.")
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("FRANKA_ROBOT_IP", None),
        help="Franka robot IP. Defaults to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--end-effector-type",
        default="franka_gripper",
        choices=["franka_gripper", "robotiq_gripper", "ruiyan_hand"],
        help="Mounted end-effector type.",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Serial port for Ruiyan hand, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--hand-baudrate",
        type=int,
        default=460800,
        help="Serial baudrate for Ruiyan hand.",
    )
    parser.add_argument(
        "--hand-motor-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Motor IDs for Ruiyan hand.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    robot_ip = args.robot_ip
    assert robot_ip is not None, "Please set the FRANKA_ROBOT_IP environment variable."

    end_effector_config = {}
    if args.end_effector_type == "ruiyan_hand":
        if args.hand_port is None:
            raise ValueError("--hand-port is required when using ruiyan_hand.")
        end_effector_config = {
            "port": args.hand_port,
            "baudrate": args.hand_baudrate,
            "motor_ids": tuple(args.hand_motor_ids),
        }

    # The arm and the end effector open their own connections, so build and
    # connect each one.
    controller = FrankaROSArm(robot_ip=robot_ip, node_rank=0)
    end_effector = EndEffector.of(
        args.end_effector_type,
        robot_ip=robot_ip,
        node_rank=0,
        **end_effector_config,
    )
    controller.connect()
    end_effector.connect()

    start_time = time.time()
    while not controller.is_robot_up():
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(
                f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
            )
    while True:
        try:
            cmd_str = input("Please input cmd:")
            if cmd_str == "q":
                break
            elif cmd_str == "getpos":
                print(controller.get_state().tcp_pose)
            elif cmd_str == "getpos_euler":
                tcp_pose = controller.get_state().tcp_pose
                r = R.from_quat(tcp_pose[3:].copy())
                euler = r.as_euler("xyz")
                print(np.concatenate([tcp_pose[:3], euler]))
            elif cmd_str == "getstate":
                state = controller.get_state()
                print(state.to_dict())
            elif cmd_str == "gethand":
                print(end_effector.get_detailed_state())
            else:
                print(f"Unknown cmd: {cmd_str}")
        except KeyboardInterrupt:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
