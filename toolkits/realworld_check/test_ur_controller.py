#!/usr/bin/env python3
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

"""Read a UR5e and command small translations through its robotics interface.

Start with --receive-only to read RTDE feedback through a receive connection.
Use --read-only to check a control session without sending position targets,
then use the interactive session for deliberate motion tests. Each offset is
measured from the current TCP pose, expressed in the robot base frame, and
interpolated over one second. Control sessions require Remote Control and a
running controller.
"""

import argparse
import time
from collections.abc import Iterable, Iterator

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics import UR5eRobot
from rlinf.robotics.parts.arms.ur_rtde import URRTDEArm
from rlinf.utils.logging import get_logger

logger = get_logger()

HELP = """Commands:
  where            read measured joints and TCP pose
  offset X Y Z     translate by X/Y/Z millimetres in the base frame, at most 5 mm
  help             show commands
  quit             stop the control script and disconnect (also EOF or Ctrl-C)
"""


def receive_only(robot_ip: str, frequency: float = 10.0, samples: int = 10) -> None:
    """Read a bounded sequence of RTDE samples without opening control.

    Args:
        robot_ip: Controller IP with RTDE enabled and reachable.
        frequency: RTDE subscription and sampling frequency in Hz.
        samples: Number of feedback snapshots to log before disconnecting.
    """
    if not np.isfinite(frequency) or not 0 < frequency <= 500:
        raise ValueError("UR receive frequency must be finite and in (0, 500] Hz.")
    if samples < 1:
        raise ValueError("UR receive sample count must be a positive integer.")

    from rtde_receive import RTDEReceiveInterface

    variables = [
        "actual_q",
        "actual_qd",
        "actual_TCP_pose",
        "actual_TCP_speed",
        "robot_mode",
        "safety_mode",
    ]
    receive = RTDEReceiveInterface(robot_ip, frequency, variables)
    try:
        for index in range(samples):
            if not receive.isConnected():
                raise RuntimeError(
                    "UR receive connection disconnected during the check."
                )
            logger.info(
                "RTDE sample %d/%d: robot_mode=%d, safety_mode=%d",
                index + 1,
                samples,
                receive.getRobotMode(),
                receive.getSafetyMode(),
            )
            for name, value in {
                "Joint positions (rad)": receive.getActualQ(),
                "Joint velocities (rad/s)": receive.getActualQd(),
                "TCP pose (m, rotation vector rad)": receive.getActualTCPPose(),
                "TCP speed (m/s, rad/s)": receive.getActualTCPSpeed(),
            }.items():
                logger.info("%s: %s", name, value)
            if index + 1 < samples:
                time.sleep(1.0 / frequency)
    except KeyboardInterrupt:
        logger.info("Interrupted; disconnecting the UR receive session.")
    finally:
        receive.disconnect()


def drive(robot: UR5eRobot, commands: Iterable[str]) -> None:
    """Read terminal commands and send bounded Cartesian translations.

    Args:
        robot: Connected UR5e with an arm named ``arm``.
        commands: Terminal input or a finite sequence of commands.
    """
    for command in commands:
        words = command.lower().split()
        if not words:
            continue
        if words == ["quit"]:
            return
        if words == ["help"]:
            logger.info(HELP)
            continue
        if words == ["where"]:
            reading = robot.get_observation()["arm"]
            logger.info("Measured joints (rad): %s", reading["arm_joint_position"])
            logger.info("TCP pose (m, xyzw): %s", reading["tcp_pose"])
            pose = reading["tcp_pose"]
            logger.info(
                "TCP pose (m, xyz Euler rad): %s",
                np.concatenate([pose[:3], R.from_quat(pose[3:]).as_euler("xyz")]),
            )
            continue
        try:
            if len(words) != 4 or words[0] != "offset":
                raise ValueError("Unknown command; type help for the accepted forms.")
            offset = np.asarray([float(value) for value in words[1:]])
            if not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > 5:
                raise ValueError("Use a finite translation with length at most 5 mm.")
        except ValueError as error:
            logger.warning("%s", error)
            continue

        start = np.asarray(robot.get_observation()["arm"]["tcp_pose"], dtype=float)
        # This is the low-rate target stream; the arm owns the servo cycle.
        for step in range(1, 21):
            target = start.copy()
            target[:3] += offset * (0.001 * step / 20)
            robot.send_action({"arm": {"tcp_pose": target}})
            time.sleep(0.05)
        logger.info("Target sent. Use where to read the measured state.")


def terminal_commands() -> Iterator[str]:
    """Yield terminal commands until EOF or Ctrl-C."""
    while True:
        try:
            yield input("ur5e> ")
        except (EOFError, KeyboardInterrupt):
            return


def main() -> None:
    """Connect a UR5e, run the requested check, and release its connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="UR5e controller IP address")
    parser.add_argument(
        "--frequency", type=float, default=500.0, help="servo or receive rate in Hz"
    )
    parser.add_argument("--lookahead-time", type=float, default=0.1, help="seconds")
    parser.add_argument("--gain", type=float, default=300.0, help="UR servo gain")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--receive-only", action="store_true", help="read RTDE without opening control"
    )
    modes.add_argument(
        "--read-only",
        action="store_true",
        help="open control and read without position targets",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="receive-only sample count"
    )
    args = parser.parse_args()
    if args.receive_only:
        receive_only(args.robot_ip, args.frequency, args.samples)
        return
    robot = UR5eRobot(
        arm=URRTDEArm.declare(
            args.robot_ip,
            frequency=args.frequency,
            lookahead_time=args.lookahead_time,
            gain=args.gain,
        )
    )
    try:
        robot.connect()
        drive(robot, ["where"])
        if not args.read_only:
            logger.info(HELP)
            drive(robot, terminal_commands())
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping the UR servo session.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
