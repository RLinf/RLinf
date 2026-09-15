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

"""Read and command a local Piper through the robotics interface.

Bring the CAN interface up at 1 Mbit/s before connecting. Start with a state
read, then omit --read-only for an interactive session::

    python -m toolkits.realworld_check.test_piper_controller --channel can0 --read-only
    python -m toolkits.realworld_check.test_piper_controller --mock

Connection enables the motors but sends no position target or reset. Motion
commands use measured joint positions, so a previous clipped target cannot
accumulate into the next move. Disconnect leaves the motors holding position.
"""

import argparse
import contextlib
import signal
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from rlinf.robotics import PiperRobot
from rlinf.robotics.parts.arms.piper import PiperArm
from rlinf.utils.logging import get_logger

logger = get_logger()

HELP = """Commands:
  where             read measured joints, tool pose, and gripper
  joint N DEGREES   move joint 1..6 by -5..5 degrees from its measured position
  grip FRACTION    set gripper opening: 0 closed, 1 open
  help              show commands
  quit              disconnect (also EOF or Ctrl-C)
"""


def drive(robot: PiperRobot, commands: Iterable[str]) -> None:
    """Read commands for a connected robot, reporting invalid input.

    Args:
        robot: Connected local Piper robot, with an arm named ``arm``.
        commands: Terminal input or a finite sequence of commands.
    """
    arm = robot.child("arm", PiperArm)
    for command in commands:
        words = command.lower().split()
        if not words:
            continue
        if words == ["quit"]:
            return
        if words == ["help"]:
            logger.info(HELP)
            continue
        reading = robot.get_observation()["arm"]
        if words == ["where"]:
            logger.info(
                "Measured joints (deg): %s", np.rad2deg(reading["arm_joint_position"])
            )
            logger.info("Tool pose (m, xyzw): %s", reading["tcp_pose"])
            if "end_effector" in reading:
                logger.info("Gripper opening: %s", reading["end_effector"]["state"])
            continue
        try:
            if len(words) == 3 and words[0] == "joint":
                joint, degrees = int(words[1]), float(words[2])
                if not 1 <= joint <= 6 or not np.isfinite(degrees) or abs(degrees) > 5:
                    raise ValueError(
                        "Use joint 1..6 and an increment in [-5, 5] degrees."
                    )
                target = np.array(reading["arm_joint_position"], dtype=float)
                target[joint - 1] += np.deg2rad(degrees)
                action = {"joint_position": target}
            elif len(words) == 2 and words[0] == "grip":
                opening = float(words[1])
                if not np.isfinite(opening) or not 0 <= opening <= 1:
                    raise ValueError("Use a gripper opening in [0, 1].")
                if "end_effector" not in reading:
                    raise ValueError("This arm was declared without a gripper.")
                action = {"end_effector": {"target": np.array([opening])}}
            else:
                raise ValueError("Unknown command; type help for the accepted forms.")
        except ValueError as error:
            logger.warning("%s", error)
            continue
        robot.send_action({"arm": action})
        if "joint_position" in action:
            arm.wait_until_still()
        logger.info("Command sent. Use where to read the measured state.")


def terminal_commands() -> Iterator[str]:
    """Yield terminal commands until EOF or Ctrl-C."""
    while True:
        try:
            yield input("piper> ")
        except (EOFError, KeyboardInterrupt):
            return


def main() -> None:
    """Connect the selected Piper, run the check, and release its connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0", help="CAN channel (default: can0)")
    parser.add_argument("--interface", default="socketcan", help="python-can interface")
    parser.add_argument(
        "--model", choices=("piper", "piper_h", "piper_l", "piper_x"), default="piper"
    )
    parser.add_argument(
        "--firmware", default=None, help="firmware profile; default: auto-detect"
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=10,
        help="motion speed, 1..100 (default: 10)",
    )
    parser.add_argument(
        "--gripper-max-width",
        type=float,
        default=0.07,
        help="full gripper stroke in metres (default: 0.07)",
    )
    parser.add_argument(
        "--no-gripper", action="store_true", help="arm has no AgxGripper"
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="connect, read state, and exit without sending targets",
    )
    parser.add_argument(
        "--mock", action="store_true", help="use mock vendor SDKs without hardware"
    )
    args = parser.parse_args()
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1, 100].")
    if not np.isfinite(args.gripper_max_width) or args.gripper_max_width <= 0:
        parser.error("--gripper-max-width must be a positive finite width in metres.")

    context = contextlib.nullcontext()
    if args.mock:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        from robot_mocks import mocked_sdks

        context = mocked_sdks()

    with context:
        robot = PiperRobot(
            arm=PiperArm.declare(
                args.channel,
                interface=args.interface,
                model=args.model,
                firmware=args.firmware,
                speed_percent=args.speed_percent,
                gripper_max_width=args.gripper_max_width,
                with_gripper=not args.no_gripper,
            )
        )
        robot.connect()
        try:
            drive(robot, ["where"])
            if not args.read_only:
                logger.info(HELP)
                drive(robot, terminal_commands())
        except KeyboardInterrupt:
            logger.info("Interrupted; disconnecting the Piper.")
        finally:
            # Let cleanup finish even if another Ctrl-C arrives during teardown.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                robot.disconnect()
            finally:
                signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    main()
