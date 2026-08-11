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

"""Interactive smoke test for single-arm :class:`FrankyController`.

Set ``FRANKA_ROBOT_IP`` (and optionally gripper / Ruiyan hand options)
before running. Only one client can hold a libfranka session.

XYZ motion examples (meters, robot base frame)::

    x 0.02          # +2 cm along x
    y -0.01         # -1 cm along y
    z 0.03          # +3 cm along z
    move 0.02 0 0   # relative dx dy dz
    goto 0.4 0 0.3  # absolute TCP xyz, keep orientation
"""

import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.franka.franky_controller import FrankyController

# Per-command safety cap for interactive relative/absolute moves.
_MAX_DELTA_M = float(os.environ.get("FRANKA_TEST_MAX_DELTA_M", 0.15))
_CART_MAX_STEP_M = float(os.environ.get("RLINF_CART_MAX_STEP_M", 0.10))
_MOVE_DT_S = float(os.environ.get("FRANKA_TEST_MOVE_DT_S", 0.05))
_SETTLE_S = float(os.environ.get("FRANKA_TEST_SETTLE_S", 0.4))


def _parse_args():
    parser = argparse.ArgumentParser(description="Check FrankyController state.")
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
        "--gripper-connection",
        default=os.environ.get("FRANKA_GRIPPER_PORT"),
        help="Serial port for Robotiq grippers, e.g. /dev/ttyUSB0.",
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


def _print_help() -> None:
    print(
        "commands: q | help | getpos | getpos_euler | getstate | gethand | "
        "open | close | "
        "x <m> | y <m> | z <m> | move <dx> <dy> <dz> | goto <x> <y> <z>"
    )


def _move_tcp_to(controller, target_pose: np.ndarray, label: str) -> None:
    """Stream ``move_tcp_pose`` until the slewed target reaches ``target_pose``."""
    target_pose = np.asarray(target_pose, dtype=np.float64).copy()
    assert target_pose.shape == (7,)
    target_pose[3:] /= np.linalg.norm(target_pose[3:])

    start = controller.get_state().wait()[0].tcp_pose.copy()
    delta = target_pose[:3] - start[:3]
    dist = float(np.linalg.norm(delta))
    if dist > _MAX_DELTA_M:
        raise ValueError(
            f"refusing move: ||delta||={dist:.3f} m > {_MAX_DELTA_M:.3f} m "
            f"safety cap (set FRANKA_TEST_MAX_DELTA_M to raise)"
        )

    n_steps = max(1, int(np.ceil(dist / max(_CART_MAX_STEP_M, 1e-6))) + 2)
    print(
        f"{label}: {start[:3]} -> {target_pose[:3]} "
        f"(||d||={dist:.4f} m, {n_steps} steps)"
    )
    for _ in range(n_steps):
        controller.move_tcp_pose(target_pose).wait()
        time.sleep(_MOVE_DT_S)
    time.sleep(_SETTLE_S)
    end = controller.get_state().wait()[0].tcp_pose
    print(f"done: tcp xyz={end[:3]}  err={np.linalg.norm(end[:3] - target_pose[:3]):.4f} m")


def _move_relative(controller, dxyz: np.ndarray) -> None:
    dxyz = np.asarray(dxyz, dtype=np.float64).reshape(3)
    pose = controller.get_state().wait()[0].tcp_pose.copy()
    target = pose.copy()
    target[:3] += dxyz
    _move_tcp_to(controller, target, label=f"move relative {dxyz}")


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

    controller = FrankyController.launch_controller(
        robot_ip=robot_ip,
        end_effector_type=args.end_effector_type,
        end_effector_config=end_effector_config,
        gripper_connection=args.gripper_connection,
    )

    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(
                f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
            )
            break

    print(f"Connected to Franka at {robot_ip} via FrankyController")
    _print_help()

    while True:
        try:
            cmd_str = input("Please input cmd:").strip()
            if not cmd_str:
                continue
            parts = cmd_str.split()
            cmd = parts[0].lower()

            if cmd == "q":
                break
            elif cmd == "help":
                _print_help()
            elif cmd == "getpos":
                print(controller.get_state().wait()[0].tcp_pose)
            elif cmd == "getpos_euler":
                tcp_pose = controller.get_state().wait()[0].tcp_pose
                r = R.from_quat(tcp_pose[3:].copy())
                euler = r.as_euler("xyz")
                print(np.concatenate([tcp_pose[:3], euler]))
            elif cmd == "getstate":
                state = controller.get_state().wait()[0]
                print(state.to_dict())
            elif cmd == "gethand":
                print(controller.get_hand_detailed_state().wait()[0])
            elif cmd == "open":
                controller.open_gripper().wait()
                print("gripper opened")
            elif cmd == "close":
                controller.close_gripper().wait()
                print("gripper closed")
            elif cmd in ("x", "y", "z"):
                if len(parts) != 2:
                    print(f"usage: {cmd} <delta_m>")
                    continue
                delta = float(parts[1])
                dxyz = np.zeros(3, dtype=np.float64)
                dxyz[{"x": 0, "y": 1, "z": 2}[cmd]] = delta
                _move_relative(controller, dxyz)
            elif cmd == "move":
                if len(parts) != 4:
                    print("usage: move <dx> <dy> <dz>   # meters, relative")
                    continue
                dxyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                _move_relative(controller, dxyz)
            elif cmd == "goto":
                if len(parts) != 4:
                    print("usage: goto <x> <y> <z>   # meters, absolute TCP")
                    continue
                pose = controller.get_state().wait()[0].tcp_pose.copy()
                pose[:3] = [float(parts[1]), float(parts[2]), float(parts[3])]
                _move_tcp_to(controller, pose, label="goto")
            else:
                print(f"Unknown cmd: {cmd_str}")
                _print_help()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"command failed: {e}")
        time.sleep(0.05)

    try:
        controller.cleanup().wait()
    except Exception:
        pass


if __name__ == "__main__":
    main()
