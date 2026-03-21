#!/usr/bin/env python3
# Copyright 2026 Ying-Chun Lee
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

"""Minimum YAM follower server with zero-torque support."""

from __future__ import annotations

import argparse

import portal
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a YAM follower server compatible with RLinf."
    )
    parser.add_argument("--can_channel", type=str, default="can0")
    parser.add_argument(
        "--gripper",
        type=str,
        default="linear_4310",
        choices=GripperType.available_grippers(),
    )
    parser.add_argument("--mode", type=str, default="follower")
    parser.add_argument("--server_port", type=int, default=11333)
    parser.add_argument("--gripper_open_limit", type=float, default=None)
    parser.add_argument("--gripper_close_limit", type=float, default=None)
    return parser.parse_args()


def build_gripper_limits(
    gripper_type: GripperType,
    open_limit: float | None,
    close_limit: float | None,
):
    default_limits = gripper_type.get_gripper_limits()
    if open_limit is None and close_limit is None:
        return default_limits
    if default_limits is None:
        return None

    resolved_open = default_limits[0] if open_limit is None else open_limit
    resolved_close = default_limits[1] if close_limit is None else close_limit
    return [resolved_open, resolved_close]


def main() -> None:
    args = parse_args()
    if args.mode != "follower":
        raise ValueError(
            "scripts/yam_follower_server.py currently supports only --mode follower."
        )

    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(
        channel=args.can_channel,
        gripper_type=gripper_type,
        gripper_limits=build_gripper_limits(
            gripper_type=gripper_type,
            open_limit=args.gripper_open_limit,
            close_limit=args.gripper_close_limit,
        ),
    )

    server = portal.Server(args.server_port)
    print(
        f"Robot Sever Binding to {args.server_port}, Robot: {robot}",
        flush=True,
    )
    server.bind("num_dofs", robot.num_dofs)
    server.bind("get_joint_pos", robot.get_joint_pos)
    server.bind("command_joint_pos", robot.command_joint_pos)
    server.bind("command_joint_state", robot.command_joint_state)
    server.bind("get_observations", robot.get_observations)
    server.bind("zero_torque_mode", robot.zero_torque_mode)
    server.start()


if __name__ == "__main__":
    main()
