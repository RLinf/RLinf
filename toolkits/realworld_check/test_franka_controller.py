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
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.arms import Arm
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.robotics.robots import FrankaRobot

# Franka Emika Panda factory "ready" pose, in radians.
HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

COMMANDS = (
    "q | getpos | getpos_euler | getjoint | getstate | gethand | "
    "clear | home | open | close"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Franka controller state.")
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("FRANKA_ROBOT_IP", None),
        help="Franka robot IP. Defaults to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--backend",
        default=FrankaRobot.BACKEND,
        choices=sorted(Arm.backends()),
        help="Arm backend (default: franky). Use franka_ros for the legacy ROS "
        "Noetic stack.",
    )
    parser.add_argument(
        "--realtime-config",
        default=None,
        choices=["enforce", "ignore"],
        help="libfranka real-time mode for --backend franky (default: ignore). "
        "franka_ros takes it from FRANKA_REALTIME_CONFIG at install time.",
    )
    parser.add_argument(
        "--end-effector-type",
        default=None,
        choices=sorted(EndEffector.backends()),
        help="Mounted end-effector type.",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="End-effector serial port, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--hand-baudrate",
        type=int,
        default=None,
        help="End-effector serial baudrate; defaults to the driver setting.",
    )
    parser.add_argument(
        "--hand-motor-ids",
        type=int,
        nargs="+",
        default=None,
        help="End-effector motor IDs; defaults to the driver setting.",
    )
    parser.add_argument(
        "--end-effector-config",
        type=json.loads,
        default={},
        help='Driver settings as a JSON object, e.g. {"port": "/dev/ttyUSB0"}.',
    )
    args = parser.parse_args()
    if not isinstance(args.end_effector_config, dict):
        parser.error("--end-effector-config must be a JSON object")
    return args


def main() -> None:
    args = _parse_args()
    robot_ip = args.robot_ip
    assert robot_ip is not None, "Please set the FRANKA_ROBOT_IP environment variable."

    end_effector_config = dict(args.end_effector_config)
    for key, value in (
        ("port", args.hand_port),
        ("baudrate", args.hand_baudrate),
        (
            "motor_ids",
            tuple(args.hand_motor_ids) if args.hand_motor_ids is not None else None,
        ),
    ):
        if value is not None:
            end_effector_config[key] = value

    arm_settings = {}
    if args.realtime_config is not None:
        arm_settings["realtime_config"] = args.realtime_config

    # The arm and the end effector open their own connections, so build and
    # connect each one.
    controller = FrankaRobot.declare_arm(
        robot_ip, node_rank=0, name="CheckArm", backend=args.backend, **arm_settings
    )
    end_effector = FrankaRobot.declare_end_effector(
        robot_ip,
        node_rank=0,
        name="CheckEndEffector",
        backend=args.backend,
        end_effector_type=args.end_effector_type,
        end_effector_config=end_effector_config,
    )
    robot = FrankaRobot(arm=controller, end_effector=end_effector)
    try:
        robot.connect()

        start_time = time.time()
        while not controller.is_robot_up():
            time.sleep(0.5)
            if time.time() - start_time > 30:
                raise TimeoutError("Franka did not become ready within 30 seconds.")
        print(f"Commands: {COMMANDS}")
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
                elif cmd_str == "getjoint":
                    print(controller.get_state().arm_joint_position)
                elif cmd_str == "getstate":
                    state = controller.get_state()
                    print(state.to_dict())
                elif cmd_str == "gethand":
                    print(end_effector.get_observation())
                elif cmd_str == "clear":
                    controller.clear_errors()
                elif cmd_str == "home":
                    controller.reset_joint(HOME_JOINTS)
                elif cmd_str == "open":
                    end_effector.open()
                elif cmd_str == "close":
                    end_effector.close()
                else:
                    print(f"Unknown cmd: {cmd_str}. Commands: {COMMANDS}")
            except (EOFError, KeyboardInterrupt):
                break
            time.sleep(1.0)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
