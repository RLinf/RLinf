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

"""Interactive smoke test for :class:`FrankyController`. Type ``help`` at
the prompt for the command list. Set ``FRANKA_ROBOT_IP`` (and optionally
``FRANKA_GRIPPER_TYPE`` / ``FRANKA_GRIPPER_PORT``) before running.

Only one client can hold a libfranka session — running this while an env
is already connected to the same arm will fail.
"""

import os
import time

# Silence Ray actor stdout capture BEFORE any ray import — otherwise the
# actor's logger writes are forwarded to the driver and interleave with
# the interactive prompt.
import ray  # noqa: E402

if not ray.is_initialized():
    try:
        ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    except Exception:
        ray.init(log_to_driver=False, logging_level="ERROR")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from rlinf.envs.realworld.franka.franky_controller import (  # noqa: E402
    FrankyController,
)

# Franka Emika Panda factory "ready" pose.
HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def _print_help() -> None:
    print(
        "commands: q | getpos | getpos_euler | getjoint | home | "
        "nudge <i> <d> | stream <i> <d> <n> | hold <secs> | "
        "open | close | grip <0-255> | impedance <7 ints>"
    )


def main() -> None:
    robot_ip = os.environ.get("FRANKA_ROBOT_IP")
    assert robot_ip is not None, "Please set the FRANKA_ROBOT_IP environment variable."

    gripper_type = os.environ.get("FRANKA_GRIPPER_TYPE", "robotiq")
    gripper_connection = os.environ.get("FRANKA_GRIPPER_PORT")

    controller = FrankyController.launch_controller(
        robot_ip=robot_ip,
        gripper_type=gripper_type,
        gripper_connection=gripper_connection,
    )

    # Wait for the controller to publish a valid state.
    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(f"Waited {time.time() - start_time:.1f}s for Franka to be ready")
            break

    print(f"Connected to Franka at {robot_ip}")
    _print_help()

    while True:
        try:
            cmd_str = input("cmd> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd_str:
            continue

        parts = cmd_str.split()
        cmd = parts[0].lower()

        try:
            if cmd == "q":
                break
            elif cmd == "help":
                _print_help()
            elif cmd == "getpos":
                pose = controller.get_state().wait()[0].tcp_pose
                print(pose)
            elif cmd == "getpos_euler":
                pose = controller.get_state().wait()[0].tcp_pose
                euler = R.from_quat(pose[3:].copy()).as_euler("xyz")
                print(np.concatenate([pose[:3], euler]))
            elif cmd == "getjoint":
                state = controller.get_state().wait()[0]
                print(state.arm_joint_position)
            elif cmd == "home":
                print(f"Resetting to home: {HOME_JOINTS}")
                controller.reset_joint(HOME_JOINTS).wait()
                print("Home reached")
            elif cmd == "nudge":
                if len(parts) != 3:
                    print("usage: nudge <joint_index 1..7> <delta_rad>")
                    continue
                idx = int(parts[1]) - 1
                delta = float(parts[2])
                assert 0 <= idx < 7, "joint index must be 1..7"
                current = controller.get_state().wait()[0].arm_joint_position
                target = current.copy()
                target[idx] += delta
                print(f"move_joints: {target}")
                controller.move_joints(target).wait()
            elif cmd == "stream":
                if len(parts) != 4:
                    print("usage: stream <joint_index 1..7> <delta_per_tick> <n_ticks>")
                    continue
                idx = int(parts[1]) - 1
                delta = float(parts[2])
                n = int(parts[3])
                assert 0 <= idx < 7, "joint index must be 1..7"
                total = abs(delta) * n
                if total > 0.5:
                    print(
                        f"refusing stream: total displacement "
                        f"{total:.3f} rad > 0.5 rad safety cap "
                        f"(reduce delta or n)"
                    )
                    continue
                current = controller.get_state().wait()[0].arm_joint_position
                print(
                    f"starting from joint {idx + 1} = {current[idx]:+.4f} rad; "
                    f"total planned displacement {delta * n:+.4f} rad"
                )
                target = current.copy()
                t0 = time.time()
                for _ in range(n):
                    target[idx] += delta
                    controller.move_joints(target).wait()
                    time.sleep(0.001)
                elapsed = time.time() - t0
                end = controller.get_state().wait()[0].arm_joint_position
                print(
                    f"streamed {n} ticks in {elapsed:.3f}s "
                    f"(~{n / elapsed:.0f} Hz); "
                    f"joint {idx + 1} now = {end[idx]:+.4f} rad"
                )
            elif cmd == "hold":
                secs = float(parts[1]) if len(parts) == 2 else 30.0
                print(f"holding for {secs:.1f}s — listen for buzz")
                time.sleep(secs)
                state = controller.get_state().wait()[0]
                print(
                    f"joint_vel rms = "
                    f"{np.sqrt(np.mean(state.arm_joint_velocity**2)):.5f} rad/s"
                )
            elif cmd == "open":
                controller.open_gripper().wait()
                print("gripper opened")
            elif cmd == "close":
                controller.close_gripper().wait()
                print("gripper closed")
            elif cmd == "grip":
                if len(parts) != 2:
                    print("usage: grip <0-255>")
                    continue
                pos = int(parts[1])
                controller.move_gripper(pos).wait()
                print(f"gripper moved to {pos}")
            elif cmd == "impedance":
                if len(parts) != 8:
                    print("usage: impedance <k1 k2 k3 k4 k5 k6 k7>")
                    continue
                Kq = [float(x) for x in parts[1:]]
                controller.reconfigure_compliance_params({"Kq": Kq}).wait()
                print(f"impedance updated to {Kq}")
            else:
                print(f"unknown cmd: {cmd_str}")
                _print_help()
        except Exception as e:
            print(f"command failed: {e}")

        time.sleep(0.05)

    print("shutting down...")
    try:
        controller.cleanup().wait()
    except Exception:
        pass


if __name__ == "__main__":
    main()
