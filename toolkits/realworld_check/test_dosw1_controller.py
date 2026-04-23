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
import math
import time

import numpy as np

from rlinf.envs.realworld.dosw1.dosw1_env import DOSW1Config
from rlinf.envs.realworld.dosw1.dosw1_sdk import DOSW1SDKAdapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate DOSW1 follower arms with leader arms and stream states."
    )
    parser.add_argument("--robot-url", type=str, default="localhost")
    parser.add_argument("--left-arm-port", type=int, default=50051)
    parser.add_argument("--right-arm-port", type=int, default=50053)
    parser.add_argument("--left-lead-port", type=int, default=50050)
    parser.add_argument("--right-lead-port", type=int, default=50052)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--print-hz", type=float, default=30.0)
    parser.add_argument("--joint-limit-min", type=float, default=-3.14)
    parser.add_argument("--joint-limit-max", type=float, default=3.14)
    parser.add_argument("--max-joint-delta", type=float, default=math.inf)
    parser.add_argument("--gripper-factor", type=float, default=0.07 / 0.048)
    parser.add_argument("--gripper-teleop-scale", type=float, default=5.0)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DOSW1Config:
    return DOSW1Config(
        robot_url=args.robot_url,
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        left_lead_port=args.left_lead_port,
        right_lead_port=args.right_lead_port,
        is_dummy=False,
        enable_human_in_loop=True,
    )


def _compute_teleop_targets(
    lead_left: np.ndarray,
    lead_right: np.ndarray,
    cur_follow_left: np.ndarray,
    cur_follow_right: np.ndarray,
    init_lead_left: np.ndarray,
    init_lead_right: np.ndarray,
    init_follow_left: np.ndarray,
    init_follow_right: np.ndarray,
    joint_limit_min: np.ndarray,
    joint_limit_max: np.ndarray,
    max_joint_delta: float,
    gripper_scale: float,
    gripper_min: float,
    gripper_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta_left_joint = lead_left[:6] - init_lead_left[:6]
    delta_left_gripper = gripper_scale * (lead_left[6] - init_lead_left[6])
    left_joint = np.clip(
        init_follow_left[:6] + delta_left_joint, joint_limit_min, joint_limit_max
    )
    left_joint = np.clip(
        left_joint,
        cur_follow_left[:6] - max_joint_delta,
        cur_follow_left[:6] + max_joint_delta,
    )
    left_gripper = float(
        np.clip(init_follow_left[6] + delta_left_gripper, gripper_min, gripper_max)
    )

    delta_right_joint = lead_right[:6] - init_lead_right[:6]
    delta_right_gripper = gripper_scale * (lead_right[6] - init_lead_right[6])
    right_joint = np.clip(
        init_follow_right[:6] + delta_right_joint, joint_limit_min, joint_limit_max
    )
    right_joint = np.clip(
        right_joint,
        cur_follow_right[:6] - max_joint_delta,
        cur_follow_right[:6] + max_joint_delta,
    )
    right_gripper = float(
        np.clip(init_follow_right[6] + delta_right_gripper, gripper_min, gripper_max)
    )

    left_target = np.concatenate(
        [left_joint, np.array([left_gripper], dtype=np.float64)]
    )
    right_target = np.concatenate(
        [right_joint, np.array([right_gripper], dtype=np.float64)]
    )
    return left_target, right_target


def _fmt(arr: np.ndarray) -> str:
    return np.array2string(np.asarray(arr), precision=4, floatmode="fixed")


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)
    sdk = DOSW1SDKAdapter(cfg)

    control_dt = 1.0 / max(args.control_hz, 1e-6)
    print_dt = 1.0 / max(args.print_hz, 1e-6)

    joint_limit_min = np.full(6, args.joint_limit_min, dtype=np.float64)
    joint_limit_max = np.full(6, args.joint_limit_max, dtype=np.float64)
    max_joint_delta = float(args.max_joint_delta)
    gripper_scale = float(args.gripper_teleop_scale) * float(args.gripper_factor)
    gripper_min = float(cfg.gripper_width_min)
    gripper_max = float(cfg.gripper_width_max)

    print("Connecting DOSW1 SDK...")
    sdk.connect()
    print("Connected. Keep moving leader arms to teleoperate follower arms.")
    print("Press Ctrl+C to exit.")

    try:
        init_lead_left = sdk.get_left_lead_joint()
        init_lead_right = sdk.get_right_lead_joint()
        init_follow_left = sdk.get_left_joint()
        init_follow_right = sdk.get_right_joint()

        next_print_time = time.time()
        while True:
            loop_start = time.time()

            lead_left = sdk.get_left_lead_joint()
            lead_right = sdk.get_right_lead_joint()
            cur_follow_left = sdk.get_left_joint()
            cur_follow_right = sdk.get_right_joint()

            left_target, right_target = _compute_teleop_targets(
                lead_left=lead_left,
                lead_right=lead_right,
                cur_follow_left=cur_follow_left,
                cur_follow_right=cur_follow_right,
                init_lead_left=init_lead_left,
                init_lead_right=init_lead_right,
                init_follow_left=init_follow_left,
                init_follow_right=init_follow_right,
                joint_limit_min=joint_limit_min,
                joint_limit_max=joint_limit_max,
                max_joint_delta=max_joint_delta,
                gripper_scale=gripper_scale,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )

            sdk.left_go_joint(left_target[:6].tolist(), float(left_target[6]))
            sdk.right_go_joint(right_target[:6].tolist(), float(right_target[6]))

            now = time.time()
            if now >= next_print_time:
                state = sdk.get_state()
                left_pose = sdk.get_left_pose()
                right_pose = sdk.get_right_pose()
                print(
                    f"[{now:.3f}] "
                    f"left_joint={_fmt(np.concatenate([state.left_joint_positions, [state.left_gripper]]))} "
                    f"right_joint={_fmt(np.concatenate([state.right_joint_positions, [state.right_gripper]]))}"
                )
                print(
                    f"[{now:.3f}] left_eef={_fmt(left_pose)} right_eef={_fmt(right_pose)}"
                )
                next_print_time = now + print_dt

            elapsed = time.time() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\nExit requested by user.")
    finally:
        sdk.disconnect()
        print("DOSW1 SDK disconnected.")


if __name__ == "__main__":
    main()
