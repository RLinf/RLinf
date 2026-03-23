# Copyright 2025 The RLinf Authors.
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

"""Interactive diagnostic tool for FrankaController and dexterous hands.

Environment variables
---------------------
FRANKA_ROBOT_IP          Robot arm IP (required).
FRANKA_END_EFFECTOR_TYPE End-effector type: ``franka_gripper`` (default),
                         ``aoyi_hand``, or ``ruiyan_hand``.
FRANKA_HAND_PORT         Serial port for the dexterous hand
                         (default ``/dev/ttyUSB0``).
FRANKA_HAND_BAUDRATE     Baudrate for the dexterous hand
                         (default ``460800``).

Example
-------
.. code-block:: bash

   export FRANKA_ROBOT_IP=172.16.0.2
   export FRANKA_END_EFFECTOR_TYPE=ruiyan_hand
   python -m toolkits.realworld_check.test_franka_controller
"""

import os
import time

import numpy as np
import ray
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.franka.franka_controller import FrankaController

# ── ANSI colour helpers ──────────────────────────────────────────────
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# ── Finger name translations (for pretty printing) ──────────────────
_FINGER_EN = {
    "thumb_rotation": "thumb rotation",
    "thumb_bend": "thumb bend",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "pinky": "pinky",
}

_HELP_TEXT = f"""\
{_BOLD}Available commands:{_RESET}
  {_GREEN}getpos{_RESET}          Get TCP pose (quaternion)
  {_GREEN}getpos_euler{_RESET}    Get TCP pose (Euler angles)
  {_GREEN}getjoints{_RESET}       Get joint positions (7-DOF)
  {_GREEN}getvel{_RESET}          Get TCP velocity (6-DOF)
  {_GREEN}getforce{_RESET}        Get TCP force/torque
  {_CYAN}gethand{_RESET}         Get finger positions (normalised [0,1])
  {_CYAN}gethand_detail{_RESET}  Get detailed hand state (pos/vel/current/status)
  {_CYAN}handinfo{_RESET}        Show hand config (type/port/baudrate/DOF)
  {_GREEN}state{_RESET}           Show full robot state
  {_GREEN}help{_RESET}            Show this help
  {_GREEN}q{_RESET}               Quit
"""


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple ASCII table."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    print(sep)


def _fmt_arr(arr, decimals: int = 4) -> str:
    """Format a numpy array to a compact string."""
    return np.array2string(np.asarray(arr), precision=decimals, suppress_small=True)


def _handle_gethand(controller) -> None:
    """Print normalised hand finger positions."""
    hand_type = controller.get_hand_type().wait()[0]
    if hand_type == "franka_gripper":
        state = controller.get_state().wait()[0]
        print(f"  Gripper pos: {state.gripper_position}  ({'open' if state.gripper_open else 'closed'})")
        return
    hand_state = controller.get_hand_state().wait()[0]
    finger_names = controller.get_hand_finger_names().wait()[0]
    rows = []
    for i, (name, pos) in enumerate(zip(finger_names, hand_state)):
        en = _FINGER_EN.get(name, name)
        rows.append([str(i + 1), name, en, f"{pos:.4f}"])
    _print_table(["#", "DOF", "Name", "Position [0,1]"], rows)


def _handle_gethand_detail(controller) -> None:
    """Print detailed per-motor diagnostics."""
    hand_type = controller.get_hand_type().wait()[0]
    if hand_type == "franka_gripper":
        print("  Current end-effector is Franka gripper; no detailed motor info.")
        return
    detail = controller.get_hand_detailed_state().wait()[0]
    finger_names = detail.get("finger_names", [])
    positions = detail.get("positions", [])
    velocities = detail.get("velocities", [])
    currents = detail.get("currents", [])
    statuses = detail.get("statuses", [])
    motor_ids = detail.get("motor_ids", [])
    targets = detail.get("target_positions", [])

    rows = []
    for i in range(len(finger_names)):
        zh = _FINGER_EN.get(finger_names[i], finger_names[i])
        status_str = f"{_GREEN}OK{_RESET}" if statuses[i] == 0 else f"{_RED}ERR({statuses[i]}){_RESET}"
        rows.append([
            str(motor_ids[i] if i < len(motor_ids) else "?"),
            finger_names[i],
            zh,
            f"{positions[i]:.4f}",
            f"{targets[i]:.4f}" if i < len(targets) else "-",
            f"{velocities[i]:.0f}",
            f"{currents[i]:.0f}",
            status_str,
        ])
    _print_table(
        ["Motor ID", "DOF", "Name", "Position", "Target", "Velocity", "Current", "Status"],
        rows,
    )


def _handle_handinfo(controller) -> None:
    """Print hand configuration metadata."""
    hand_type = controller.get_hand_type().wait()[0]
    detail = controller.get_hand_detailed_state().wait()[0]
    info_rows = [
        ["End-effector type", hand_type],
        ["DOF count", str(len(detail.get("finger_names", detail.get("positions", []))))],
    ]
    if "port" in detail:
        info_rows.append(["Serial port", detail["port"]])
    if "baudrate" in detail:
        info_rows.append(["Baudrate", str(detail["baudrate"])])
    if "motor_ids" in detail:
        info_rows.append(["Motor IDs", str(detail["motor_ids"])])
    if "finger_names" in detail:
        for i, name in enumerate(detail["finger_names"]):
            en = _FINGER_EN.get(name, name)
            info_rows.append([f"  DOF {i+1}", f"{name} ({en})"])
    _print_table(["Property", "Value"], info_rows)


def main():
    robot_ip = os.environ.get("FRANKA_ROBOT_IP", None)
    assert robot_ip is not None, (
        "Please set FRANKA_ROBOT_IP, e.g.: export FRANKA_ROBOT_IP=172.16.0.2"
    )

    end_effector_type = os.environ.get("FRANKA_END_EFFECTOR_TYPE", "franka_gripper")
    hand_port = os.environ.get("FRANKA_HAND_PORT", "/dev/ttyUSB0")
    hand_baudrate = int(os.environ.get("FRANKA_HAND_BAUDRATE", "460800"))

    # Build end-effector config for dexterous hands
    end_effector_config = {}
    if end_effector_type in ("aoyi_hand", "ruiyan_hand"):
        end_effector_config["port"] = hand_port
        if end_effector_type == "ruiyan_hand":
            end_effector_config["baudrate"] = hand_baudrate

    print(f"{_BOLD}Starting FrankaController ...{_RESET}")
    print(f"  Robot IP:        {robot_ip}")
    print(f"  End-effector:    {end_effector_type}")
    if end_effector_type != "franka_gripper":
        print(f"  Hand port:       {hand_port}")
        print(f"  Baudrate:        {hand_baudrate}")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    controller = FrankaController.launch_controller(
        robot_ip=robot_ip,
        end_effector_type=end_effector_type,
        end_effector_config=end_effector_config,
    )

    # Wait for robot to come up
    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(
                f"{_YELLOW}Waited {time.time() - start_time:.0f}s, Franka still not ready ...{_RESET}"
            )

    print(f"{_GREEN}Franka ready! Type help for available commands.{_RESET}\n")

    while True:
        try:
            cmd_str = input(f"{_BOLD}cmd> {_RESET}").strip()
            if not cmd_str:
                continue
            elif cmd_str == "q":
                break
            elif cmd_str == "help":
                print(_HELP_TEXT)
            elif cmd_str == "getpos":
                print(f"  TCP (quat): {_fmt_arr(controller.get_state().wait()[0].tcp_pose)}")
            elif cmd_str == "getpos_euler":
                tcp_pose = controller.get_state().wait()[0].tcp_pose
                r = R.from_quat(tcp_pose[3:].copy())
                euler = r.as_euler("xyz")
                print(f"  TCP (euler): {_fmt_arr(np.concatenate([tcp_pose[:3], euler]))}")
            elif cmd_str == "getjoints":
                state = controller.get_state().wait()[0]
                print(f"  Joint pos:  {_fmt_arr(state.arm_joint_position)}")
                print(f"  Joint vel:  {_fmt_arr(state.arm_joint_velocity)}")
            elif cmd_str == "getvel":
                state = controller.get_state().wait()[0]
                print(f"  TCP vel:    {_fmt_arr(state.tcp_vel)}")
            elif cmd_str == "getforce":
                state = controller.get_state().wait()[0]
                print(f"  TCP force:  {_fmt_arr(state.tcp_force)}")
                print(f"  TCP torque: {_fmt_arr(state.tcp_torque)}")
            elif cmd_str == "gethand":
                _handle_gethand(controller)
            elif cmd_str == "gethand_detail":
                _handle_gethand_detail(controller)
            elif cmd_str == "handinfo":
                _handle_handinfo(controller)
            elif cmd_str == "state":
                state = controller.get_state().wait()[0]
                tcp_pose = state.tcp_pose
                r = R.from_quat(tcp_pose[3:].copy())
                euler = r.as_euler("xyz")
                print(f"  TCP (quat):  {_fmt_arr(tcp_pose)}")
                print(f"  TCP (euler): {_fmt_arr(np.concatenate([tcp_pose[:3], euler]))}")
                print(f"  TCP vel:       {_fmt_arr(state.tcp_vel)}")
                print(f"  TCP force:     {_fmt_arr(state.tcp_force)}")
                print(f"  TCP torque:    {_fmt_arr(state.tcp_torque)}")
                print(f"  Joint pos:     {_fmt_arr(state.arm_joint_position)}")
                print(f"  Joint vel:     {_fmt_arr(state.arm_joint_velocity)}")
                hand_type = controller.get_hand_type().wait()[0]
                if hand_type == "franka_gripper":
                    print(f"  Gripper pos:   {state.gripper_position}  ({'open' if state.gripper_open else 'closed'})")
                else:
                    print(f"  Hand type:     {hand_type}")
                    _handle_gethand(controller)
            else:
                print(f"{_YELLOW}Unknown command: {cmd_str}  (type help for available commands){_RESET}")
        except KeyboardInterrupt:
            print()
            break
        time.sleep(0.2)

    print("Exiting.")


if __name__ == "__main__":
    main()
