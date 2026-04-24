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

"""GimArm env-level hardware test.

Tests the RLinf GimArmController integration layer on real hardware:
feedforward control thread, Butterworth filtering, smooth reset,
move_joints, get_state (FK), and gripper — the same code path that
GimArmEnv.step() and reset() use.

The SDK itself is assumed to work. This validates the RLinf wrapper.

Usage::

    python toolkits/realworld_check/test_gim_arm_env.py --can can0
    python toolkits/realworld_check/test_gim_arm_env.py --can can0 --variant gim_arm --no-gripper
    python toolkits/realworld_check/test_gim_arm_env.py --can can0 --gripper-type single_side

Requires ``gim_arm_control`` SDK and ``pinocchio`` to be installed.
"""

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="GimArm env-level hardware test")
    parser.add_argument(
        "--can",
        type=str,
        default="can0",
        help="CAN socket interface name (default: can0)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="gim_arm_xl",
        choices=["gim_arm", "gim_arm_xl"],
        help="Arm variant (default: gim_arm_xl)",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Disable gripper",
    )
    parser.add_argument(
        "--gripper-type",
        type=str,
        default="parallel",
        choices=["parallel", "single_side"],
        help="Gripper type (default: parallel)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation before sending motion commands "
        "(use for CI / repeat runs on a known-safe setup).",
    )
    args = parser.parse_args()

    from rlinf.envs.realworld.gim_arm.gim_arm_controller import GimArmController

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}  {detail}")

    # ── 1. Launch controller via distributed Worker ──────────────────────────
    print(f"\n[1] Launching GimArmController on '{args.can}' ...")
    controller = GimArmController.launch_controller(
        can_interface=args.can,
        arm_variant=args.variant,
        enable_gripper=not args.no_gripper,
        gripper_type=args.gripper_type,
    )
    print("  Controller launched (SDK + feedforward thread started)")

    try:
        # ── 2. is_robot_up ───────────────────────────────────────────────────
        print("\n[2] Checking is_robot_up() ...")
        up = controller.is_robot_up().wait()[0]
        check("is_robot_up() returns True", up)

        # ── 3. get_state — verify shapes and print ──────────────────────────
        print("\n[3] Checking get_state() ...")
        state = controller.get_state().wait()[0]

        check(
            "tcp_pose shape (7,)",
            state.tcp_pose.shape == (7,),
            f"got {state.tcp_pose.shape}",
        )
        check(
            "tcp_vel shape (6,)",
            state.tcp_vel.shape == (6,),
            f"got {state.tcp_vel.shape}",
        )
        check(
            "arm_joint_position shape (6,)",
            state.arm_joint_position.shape == (6,),
            f"got {state.arm_joint_position.shape}",
        )
        check(
            "arm_joint_velocity shape (6,)",
            state.arm_joint_velocity.shape == (6,),
            f"got {state.arm_joint_velocity.shape}",
        )
        check(
            "tcp_force shape (3,)",
            state.tcp_force.shape == (3,),
            f"got {state.tcp_force.shape}",
        )
        check(
            "tcp_torque shape (3,)",
            state.tcp_torque.shape == (3,),
            f"got {state.tcp_torque.shape}",
        )
        check(
            "arm_jacobian shape (6, 6)",
            state.arm_jacobian.shape == (6, 6),
            f"got {state.arm_jacobian.shape}",
        )
        check("tcp_pose has no NaN", not np.any(np.isnan(state.tcp_pose)))
        check("arm_jacobian has no NaN", not np.any(np.isnan(state.arm_jacobian)))

        print(f"  Joint positions: {np.round(state.arm_joint_position, 4)}")
        print(f"  TCP position:    {np.round(state.tcp_pose[:3], 4)}")
        print(f"  TCP quaternion:  {np.round(state.tcp_pose[3:], 4)}")
        print(f"  Gripper open:    {state.gripper_open}")

        # ── 4. move_joints — feedforward thread test ─────────────────────────
        # Safety precondition: the ramp below assumes the arm starts near the
        # mechanical home (all joints within 0.1 rad of zero) on a clear bench.
        # Refuse to run small motions from a non-home pose to avoid collisions.
        current_q = state.arm_joint_position
        if np.max(np.abs(current_q)) > 0.1:
            raise RuntimeError(
                f"Arm is not near the home position (max |q| = "
                f"{np.max(np.abs(current_q)):.3f} rad > 0.1 rad). Manually move "
                f"the arm to its mechanical home before running this test."
            )

        # Interactive confirmation gate — skip with --yes / -y.
        if not args.yes:
            print("\n--- About to send joint motion commands to the robot ---")
            print("  Ramp: J1 → 0.3 rad, J3 → 0.3 rad, all → 0 (2 s each)")
            print(
                "  Verify the arm is at its mechanical home and the workspace is clear."
            )
            try:
                input("  Press ENTER to continue, or Ctrl+C to abort... ")
            except KeyboardInterrupt:
                print("\n  Aborted by user.")
                return

        num_steps = 200
        step_dt = 0.01  # 100 Hz

        def ramp_to(label, target_q):
            """Linearly interpolate from current position to target_q at 100 Hz."""
            current_q = controller.get_state().wait()[0].arm_joint_position.copy()
            target_q = np.array(target_q, dtype=np.float64)
            for i in range(1, num_steps + 1):
                alpha = i / num_steps
                interp = current_q + alpha * (target_q - current_q)
                controller.move_joints(interp).wait()
                time.sleep(step_dt)
            state_after = controller.get_state().wait()[0]
            max_err = np.max(np.abs(state_after.arm_joint_position - target_q))
            check(
                f"{label} (max_err={max_err:.4f})",
                max_err < 0.1,
                f"actual={np.round(state_after.arm_joint_position, 4)}",
            )
            print(f"  Joint positions: {np.round(state_after.arm_joint_position, 4)}")

        # Small motion on joints 1 and 3 individually, holding previous positions.
        target = np.zeros(6)
        for j in [0, 2]:
            target[j] = 0.3
            print(f"\n[4.{j + 1}] Moving J{j + 1} to 0.3 rad ...")
            ramp_to(f"J{j + 1} reached 0.3 rad", target.copy())

        # Return to zero.
        print("\n[4.3] Moving all joints back to zero ...")
        ramp_to("All joints at zero", np.zeros(6))

        # ── 5. reset_joint — smooth interpolation test ───────────────────────
        print("\n[5] Testing reset_joint() — smooth return to zero (3s) ...")
        controller.reset_joint([0.0] * 6, duration=3.0).wait()
        time.sleep(0.5)

        state_after_reset = controller.get_state().wait()[0]
        max_error = np.max(np.abs(state_after_reset.arm_joint_position))
        check(
            f"All joints near zero after reset (max_err={max_error:.4f})",
            max_error < 0.1,
            f"positions={np.round(state_after_reset.arm_joint_position, 4)}",
        )
        print(f"  Joint positions: {np.round(state_after_reset.arm_joint_position, 4)}")

        # ── 6. Gripper ───────────────────────────────────────────────────────
        if not args.no_gripper:
            print("\n[6] Testing gripper ...")

            controller.open_gripper().wait()
            time.sleep(1.5)
            state_open = controller.get_state().wait()[0]
            check("Gripper reports open after open_gripper()", state_open.gripper_open)
            print(f"  Gripper position: {state_open.gripper_position:.4f} rad")

            controller.close_gripper().wait()
            time.sleep(1.5)
            state_closed = controller.get_state().wait()[0]
            check(
                "Gripper reports closed after close_gripper()",
                not state_closed.gripper_open,
            )
            print(f"  Gripper position: {state_closed.gripper_position:.4f} rad")

            controller.open_gripper().wait()
            time.sleep(1.5)
            state_reopen = controller.get_state().wait()[0]
            check("Gripper reports open after re-open", state_reopen.gripper_open)
            print(f"  Gripper position: {state_reopen.gripper_position:.4f} rad")
        else:
            print("\n[6] Gripper test skipped (--no-gripper)")

        # ── Summary ──────────────────────────────────────────────────────────
        total = passed + failed
        print(f"\n{'=' * 40}")
        print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
        if failed == 0:
            print("All checks passed.")
        else:
            print("Some checks FAILED — review output above.")
    finally:
        # ── 7. Cleanup ───────────────────────────────────────────────────────
        print("\n[7] Stopping controller ...")
        controller.stop().wait()


if __name__ == "__main__":
    main()
