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

"""GELLO ↔ Franka teleop tools. Run with ``--help`` for subcommands.

Environment variables::

    FRANKA_ROBOT_IP   Franka FCI IP (default: 172.16.0.2)
    GELLO_PORT        GELLO Dynamixel by-id path (MUST be set; placeholder shown in default is not a real device)
    GELLO_BAUDRATE    Dynamixel baudrate (default: 1000000) [calibrate only]
    ALIGN_HOME        Comma-separated 7 floats overriding HOME_JOINTS
                      [align-sequential only]

The Robotiq gripper port is auto-resolved from the local FTDI USB-RS485
adapter (each dual-Franka node has exactly one).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import Sequence

import numpy as np
import ray

if not ray.is_initialized():
    ray.init(log_to_driver=False, logging_level="ERROR")

from gello.dynamixel.driver import DynamixelDriver  # noqa: E402

from rlinf.envs.realworld.common.gello.gello_joint_expert import (  # noqa: E402
    GelloJointExpert,
)
from rlinf.envs.realworld.franka.franky_controller import (  # noqa: E402
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    FrankyController,
)
from rlinf.envs.realworld.franka.utils import wrap_to_pi  # noqa: E402

# ───────────────────────── shared helpers ──────────────────────────────


def _resolve_local_robotiq_port() -> str:
    """Pick the single FTDI USB-RS485 adapter on this node (left or right)."""
    matches = sorted(glob.glob("/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            "No FTDI USB-RS485 adapter found under /dev/serial/by-id/. "
            "Check that the Robotiq is powered and the FTDI driver is loaded."
        )
    raise RuntimeError(
        "Multiple FTDI USB-RS485 adapters found, ambiguous: " + ", ".join(matches)
    )


def colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def print_banner(title: str) -> None:
    print()
    print(colour("─" * 64, "36"))
    print(colour(f" {title}", "36;1"))
    print(colour("─" * 64, "36"))


def confirm_or_exit(prompt: str = "Proceed? [y/N]: ") -> None:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)
    if ans not in ("y", "yes"):
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)


def _format_angle(rad: float) -> str:
    """Pretty-print a radian value with both deg and π/4 multiple form."""
    deg = math.degrees(rad)
    k = int(round(rad / (math.pi / 4)))
    if abs(rad - k * math.pi / 4) < 1e-6:
        if k == 0:
            sym = "0"
        elif abs(k) == 4:
            sym = ("−" if k < 0 else "") + "π"
        elif abs(k) == 2:
            sym = ("−" if k < 0 else "") + "π/2"
        else:
            sym = ("−" if k < 0 else "+") + (f"{abs(k)}π/4" if abs(k) != 1 else "π/4")
    else:
        sym = f"{rad:+.4f} rad"
    return f"{rad:+.4f}  ({sym},  {deg:+7.2f}°)"


def fmt_joints(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"


def fmt_deg(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+.1f}°" for v in q) + "]"


def setup_franky() -> FrankyController:
    """Connect to the local Franka via FrankyController and wait until it is up."""
    robot_ip = os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2")
    gripper_port = _resolve_local_robotiq_port()
    print(f"Connecting to Franka at {robot_ip} ...", flush=True)
    controller = FrankyController.launch_controller(
        robot_ip=robot_ip,
        env_idx=0,
        node_rank=0,
        worker_rank=0,
        gripper_type="robotiq",
        gripper_connection=gripper_port,
    )
    for _ in range(60):
        if controller.is_robot_up().wait()[0]:
            break
        time.sleep(0.5)
    else:
        print("ERROR: robot did not come up", file=sys.stderr)
        sys.exit(1)
    print("Franka ready.", flush=True)
    return controller


def setup_gello_expert() -> GelloJointExpert:
    """Open the calibrated GELLO joint expert and wait for first reading."""
    gello_port = os.environ.get(
        "GELLO_PORT",
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_YOURGELLO-if00-port0",
    )
    print(f"Connecting to GELLO at {gello_port} ...", flush=True)
    gello = GelloJointExpert(port=gello_port)
    for _ in range(50):
        if gello.ready:
            break
        time.sleep(0.1)
    if not gello.ready:
        print("ERROR: GELLO did not start producing readings", file=sys.stderr)
        sys.exit(1)
    print("GELLO ready.\n", flush=True)
    return gello


def safe_reset_to(controller: FrankyController, q_target: Sequence[float]) -> None:
    """Move robot to ``q_target`` via the slow safe path with actionable errors.

    Bails out with a hint if the robot is too far from ``q_target`` (the
    1.5 rad ``max_joint_delta`` guard would refuse the move) or if FCI is
    offline.
    """
    try:
        q_now = controller.get_state().wait()[0].arm_joint_position
        delta = np.asarray(q_target) - np.asarray(q_now)
        max_d = float(np.max(np.abs(delta)))
        worst = int(np.argmax(np.abs(delta)))
        print(f"  current: {[round(float(x), 3) for x in q_now]}", flush=True)
        print(
            f"  target : {[round(float(x), 3) for x in q_target]}  "
            f"(max Δ={max_d:.3f} rad on J{worst + 1})",
            flush=True,
        )
        if max_d > 1.5:
            print(
                colour(
                    f"\n  J{worst + 1} would need to move {max_d:.3f} rad — "
                    "exceeds reset_joint safety guard (1.5 rad).",
                    "31;1",
                )
            )
            print(
                "  Manually jog the Franka closer to the target pose using\n"
                "  Desk's hand-guide mode (white button on the arm), then\n"
                "  re-run this script.\n"
            )
            sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        # Probe failed — let reset_joint surface the real error below.
        pass

    print("  → calling reset_joint (slow, ~4.5% dynamics) ...", flush=True)
    try:
        controller.reset_joint(list(q_target)).wait()
    except Exception as e:
        msg = str(e)
        print(colour(f"\n  reset_joint failed: {msg}", "31;1"), file=sys.stderr)
        if "FCI" in msg or "Connection" in msg:
            print(
                "\n  The Franka FCI session is not active.  Open\n"
                "    http://172.16.0.2/desk/\n"
                "  release the User Stop button if pressed, click the\n"
                "  three-dot menu → 'Activate FCI', and unlock the joints\n"
                "  (white LED → blue).  Then re-run this script.",
                file=sys.stderr,
            )
        elif "reflex" in msg.lower() or "discontinuity" in msg.lower():
            print(
                "\n  libfranka reflex tripped during the move — the planned\n"
                "  trajectory likely passed too close to a joint limit or\n"
                "  the workspace boundary.  Try jogging the robot to a more\n"
                "  open configuration first.",
                file=sys.stderr,
            )
        elif "max_joint_delta" in msg:
            print(
                "\n  Manually jog the Franka closer to the target pose using\n"
                "  Desk's hand-guide mode, then re-run.",
                file=sys.stderr,
            )
        sys.exit(2)
    time.sleep(0.5)


# ───────────────────── subcommand: align-check ────────────────────────


ALIGN_CHECK_THRESHOLD = 0.5  # rad — same default as GelloJointIntervention
ALIGN_CHECK_REFRESH_HZ = 5.0


def _fmt_align_check_row(i: int, q_robot: float, q_gello: float) -> str:
    raw_delta = q_gello - q_robot
    wrapped = float(wrap_to_pi(np.array([raw_delta]))[0])
    abs_w = abs(wrapped)

    if abs_w < 0.05:
        c, label, hint = "32", "ok", ""
    elif abs_w <= ALIGN_CHECK_THRESHOLD:
        c, label, hint = "33", "~~", ""
    else:
        c, label = "31", "xx"
        direction = -wrapped
        hint = (
            f"  → turn GELLO J{i + 1} by {direction:+.3f} rad "
            f"({math.degrees(direction):+.1f}°)"
        )

    delta_str = colour(f"Δ={wrapped:+.3f}", c)
    raw_note = f"  raw={raw_delta:+.3f}" if abs(raw_delta - wrapped) > 1e-6 else ""

    return (
        f"  {colour(f'[{label}]', c)} J{i + 1}  "
        f"robot={q_robot:+.3f}  gello={q_gello:+.3f}  "
        f"{delta_str}{raw_note}{hint}"
    )


def run_align_check(_args: argparse.Namespace) -> None:
    """Live full-arm alignment dashboard. NO motion is issued."""
    controller = setup_franky()
    gello = setup_gello_expert()

    period = 1.0 / ALIGN_CHECK_REFRESH_HZ
    n_lines_drawn = 0  # how many lines our last frame occupied

    try:
        while True:
            try:
                q_robot = controller.get_state().wait()[0].arm_joint_position
                q_gello, _grip = gello.get_action()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"read error: {e}", flush=True)
                time.sleep(period)
                continue

            wrapped = wrap_to_pi(np.asarray(q_gello) - np.asarray(q_robot))
            max_abs = float(np.max(np.abs(wrapped)))
            worst = int(np.argmax(np.abs(wrapped)))

            if max_abs <= ALIGN_CHECK_THRESHOLD:
                summary = colour(
                    f"ALIGNED  (max Δ={max_abs:.3f} rad on J{worst + 1})", "32;1"
                )
            else:
                summary = colour(
                    f"NOT ALIGNED  (max Δ={wrapped[worst]:+.3f} rad on J{worst + 1}; "
                    f"threshold={ALIGN_CHECK_THRESHOLD} rad)",
                    "31;1",
                )

            rows = [_fmt_align_check_row(i, q_robot[i], q_gello[i]) for i in range(7)]

            frame_lines = [summary, *rows]
            new_n = len(frame_lines)

            if n_lines_drawn:
                sys.stdout.write(f"\033[{n_lines_drawn}F")
            for line in frame_lines:
                sys.stdout.write("\r\033[2K" + line + "\n")
            n_lines_drawn = new_n
            sys.stdout.flush()

            time.sleep(period)
    except KeyboardInterrupt:
        print("\nexiting (no motion was issued).")


# ─────────────────── subcommand: align-sequential ─────────────────────


ALIGN_SEQ_TOL = 0.10  # rad — joint considered aligned when |Δ| < this
ALIGN_SEQ_STABLE_TICKS = 8  # consecutive frames inside tol → advance
ALIGN_SEQ_REFRESH_HZ = 10.0

ALIGN_SEQ_HOME_JOINTS_DEFAULT = [
    math.pi / 4,  # J1: base rotated 45°
    0.0,  # J2: upper arm vertical
    0.0,  # J3: no shoulder roll
    -math.pi / 2,  # J4: elbow at right angle
    0.0,  # J5: no forearm roll
    math.pi / 2,  # J6: wrist at right angle
    0.0,  # J7: no flange roll
]


def _parse_align_home_env() -> list[float]:
    raw = os.environ.get("ALIGN_HOME", "").strip()
    if not raw:
        return list(ALIGN_SEQ_HOME_JOINTS_DEFAULT)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 7:
        print(
            f"ERROR: ALIGN_HOME must contain exactly 7 comma-separated "
            f"floats, got {len(parts)}: {raw!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        print(f"ERROR: cannot parse ALIGN_HOME={raw!r}: {e}", file=sys.stderr)
        sys.exit(2)


def _progress_bar(delta: float, tol: float, width: int = 30) -> str:
    """Return a horizontal bar showing how far the joint is from aligned."""
    cap = math.pi / 2
    x = max(-cap, min(cap, delta))
    centre = width // 2
    pos = centre + int(round(x / cap * centre))
    pos = max(0, min(width - 1, pos))

    bar_chars = ["·"] * width
    bar_chars[centre] = "│"
    bar_chars[pos] = "●"
    bar = "".join(bar_chars)

    inside = abs(delta) <= tol
    return colour(f"[{bar}]", "32" if inside else "33")


def _print_align_seq_home_pose(home: list[float]) -> None:
    print(colour("Alignment HOME pose:", "36;1"))
    for i, v in enumerate(home):
        print(f"  J{i + 1} = {_format_angle(v)}")
    print()
    print(
        "  Geometric intent: J2 vertical, J4 at right angle (forearm\n"
        "  perpendicular to upper arm), J6 at right angle (wrist 90°\n"
        "  from forearm), and J1 rotated 45° as an asymmetric reference.\n"
        "  Override via the ALIGN_HOME env var if you want a different pose."
    )


def run_align_sequential(_args: argparse.Namespace) -> None:
    """Sequentially walk the operator through aligning J1..J7 against HOME."""
    home_joints: list[float] = _parse_align_home_env()
    controller = setup_franky()
    gello = setup_gello_expert()

    print()
    _print_align_seq_home_pose(home_joints)
    print()
    confirm_or_exit("Proceed with motion to HOME? [y/N]: ")
    print(colour("Moving robot to HOME pose before alignment ...", "36;1"))
    safe_reset_to(controller, home_joints)
    q_at_home = controller.get_state().wait()[0].arm_joint_position
    print(
        colour(
            f"  at home: {[round(float(x), 3) for x in q_at_home]}",
            "32;1",
        )
    )

    period = 1.0 / ALIGN_SEQ_REFRESH_HZ
    print()
    print("Aligning joints sequentially against the HOME pose.")
    print("Move ONLY the indicated joint until its delta stays within")
    print(
        f"±{ALIGN_SEQ_TOL:.3f} rad for {ALIGN_SEQ_STABLE_TICKS} frames, "
        "then it auto-advances.\n"
    )

    try:
        for j in range(7):
            stable = 0
            entered = time.time()
            while True:
                try:
                    q_robot = controller.get_state().wait()[0].arm_joint_position
                    q_gello, _ = gello.get_action()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\nread error: {e}", flush=True)
                    time.sleep(period)
                    continue

                delta = wrap_to_pi(float(q_gello[j]) - float(q_robot[j]))
                deg = math.degrees(delta)
                hint = "→ turn GELLO BACK" if delta > 0 else "→ turn GELLO FORWARD"
                if abs(delta) <= ALIGN_SEQ_TOL:
                    hint = colour("INSIDE TOL", "32;1")
                    stable += 1
                else:
                    stable = 0

                bar = _progress_bar(delta, ALIGN_SEQ_TOL)
                line = (
                    f"  J{j + 1}  robot={q_robot[j]:+.3f}  gello={q_gello[j]:+.3f}  "
                    f"Δ={delta:+.3f} rad ({deg:+6.1f}°)  {bar}  "
                    f"stable={stable}/{ALIGN_SEQ_STABLE_TICKS}  {hint}    "
                )
                sys.stdout.write("\r" + line)
                sys.stdout.flush()

                if stable >= ALIGN_SEQ_STABLE_TICKS:
                    elapsed = time.time() - entered
                    sys.stdout.write(
                        "\r"
                        + colour(
                            f"  J{j + 1} aligned (Δ={delta:+.3f} rad, "
                            f"took {elapsed:.1f}s)",
                            "32;1",
                        )
                        + " " * 80
                        + "\n"
                    )
                    sys.stdout.flush()
                    break

                time.sleep(period)
    except KeyboardInterrupt:
        print("\n\naborted by user; no motion was issued.", flush=True)
        return

    # Final sanity print
    q_robot = controller.get_state().wait()[0].arm_joint_position
    q_gello, _ = gello.get_action()
    deltas = [wrap_to_pi(float(q_gello[i]) - float(q_robot[i])) for i in range(7)]
    max_d = max(abs(d) for d in deltas)
    worst = int(np.argmax(np.abs(np.asarray(deltas))))
    print()
    print(colour("ALL JOINTS ALIGNED", "32;1"))
    print(f"  per-joint Δ (rad): {[f'{d:+.3f}' for d in deltas]}")
    print(
        f"  max |Δ| = {max_d:.3f} rad on J{worst + 1} "
        f"(stream gate threshold = 0.5 rad — well under)"
    )
    print()
    print("You can now Ctrl-C and start collect_data.sh.  Hold the GELLO")
    print("steady so the alignment doesn't drift before the env wrapper")
    print("opens its own stream loop.")
    print()
    print("Holding here so you don't lose the alignment — Ctrl-C to exit.")
    try:
        while True:
            q_robot = controller.get_state().wait()[0].arm_joint_position
            q_gello, _ = gello.get_action()
            deltas = [
                wrap_to_pi(float(q_gello[i]) - float(q_robot[i])) for i in range(7)
            ]
            md = max(abs(d) for d in deltas)
            worst = int(np.argmax(np.abs(np.asarray(deltas))))
            line = f"  hold: max |Δ| = {md:.3f} rad on J{worst + 1}    " + (
                "OK " if md < 0.5 else "DRIFT "
            )
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nexiting.", flush=True)


# ───────────────────── subcommand: calibrate ──────────────────────────


PI = np.pi
CALIB_POSE_A = np.array([0.0, -PI / 4, 0.0, -3 * PI / 4, 0.0, PI / 2, PI / 4])
CALIB_POSE_B = np.array([-PI / 4, 0.0, -PI / 4, -PI / 2, PI / 4, 3 * PI / 4, 0.0])
CALIB_JOINT_IDS = (1, 2, 3, 4, 5, 6, 7)
CALIB_GRIPPER_ID = 8
CALIB_NUM_ARM = len(CALIB_JOINT_IDS)


def _calib_describe_pose(name: str, q: np.ndarray) -> None:
    """Print a pose as both numeric values and human angle units."""
    print(f"  {name}:")
    for i, v in enumerate(q):
        deg = math.degrees(float(v))
        k = int(round(float(v) / (PI / 4)))
        if abs(float(v) - k * PI / 4) < 1e-6:
            if k == 0:
                sym = "0"
            elif abs(k) == 4:
                sym = ("−" if k < 0 else "") + "π"
            elif abs(k) == 2:
                sym = ("−" if k < 0 else "") + "π/2"
            else:
                sign = "−" if k < 0 else " "
                num = abs(k)
                sym = f"{sign}{num}π/4" if num != 1 else f"{sign}π/4"
        else:
            sym = f"{v:+.4f} rad"
        print(f"    J{i + 1} = {float(v):+.4f}  ({sym},  {deg:+7.2f}°)")


def _calib_setup_gello_raw():
    """Open the GELLO Dynamixel chain in RAW mode (signs and offsets bypassed)."""
    gello_port = os.environ.get(
        "GELLO_PORT",
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_YOURGELLO-if00-port0",
    )
    baudrate = int(os.environ.get("GELLO_BAUDRATE", "1000000"))
    print(
        f"Opening GELLO Dynamixel chain at {gello_port} (baud {baudrate}) ...",
        flush=True,
    )
    driver = DynamixelDriver(
        list(CALIB_JOINT_IDS) + [CALIB_GRIPPER_ID],
        port=gello_port,
        baudrate=baudrate,
    )
    # Warm up — first few reads may be noisy.
    for _ in range(10):
        driver.get_joints()
        time.sleep(0.01)
    print("GELLO ready.", flush=True)
    return driver, gello_port


def _calib_read_raw_arm(driver, n_samples: int = 30) -> np.ndarray:
    """Median several raw motor reads on the 7 arm joints."""
    samples = []
    for _ in range(n_samples):
        q = driver.get_joints()
        samples.append(np.asarray(q[:CALIB_NUM_ARM], dtype=np.float64))
        time.sleep(0.01)
    return np.median(np.stack(samples, axis=0), axis=0)


def _calib_read_raw_gripper(driver, n_samples: int = 30) -> float:
    samples = []
    for _ in range(n_samples):
        q = driver.get_joints()
        samples.append(float(q[CALIB_NUM_ARM]))
        time.sleep(0.01)
    return float(np.median(np.asarray(samples)))


def _calib_wait_for_enter(prompt: str, driver=None) -> None:
    """Block on ENTER while optionally streaming raw GELLO values."""
    print()
    print(prompt)
    if driver is not None:
        print("(raw motor positions stream below — press ENTER when GELLO matches)")
    try:
        if driver is None:
            input("  press ENTER to continue: ")
            return
        import select

        last_print = 0.0
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                sys.stdin.readline()
                return
            now = time.time()
            if now - last_print > 0.2:
                q = driver.get_joints()
                arm = np.asarray(q[:CALIB_NUM_ARM], dtype=np.float64)
                line = "  raw motors: " + " ".join(
                    f"J{i + 1}={arm[i]:+.3f}" for i in range(CALIB_NUM_ARM)
                )
                sys.stdout.write("\r\033[2K" + line)
                sys.stdout.flush()
                last_print = now
    except KeyboardInterrupt:
        print()
        print("aborted by user; no calibration was written.", file=sys.stderr)
        sys.exit(130)


def _calib_snap_to_half_pi(x: float) -> float:
    """Round x to the nearest k * π/2 in [-8π, 8π]."""
    grid = np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1)
    return float(grid[int(np.argmin(np.abs(grid - x)))])


def _calib_half_pi_label(x: float) -> str:
    """Return a 'k * np.pi/2' style symbolic label for x."""
    k = int(round(x / (np.pi / 2)))
    return f"{k} * np.pi / 2"


def run_calibrate(_args: argparse.Namespace) -> None:
    """Two-pose sign + offset calibration. Moves the robot via reset_joint."""
    controller = setup_franky()
    driver, gello_port = _calib_setup_gello_raw()

    print_banner("Calibration plan — review before any motion")
    _calib_describe_pose("POSE A (Franka home)", CALIB_POSE_A)
    print()
    _calib_describe_pose("POSE B (calibration target)", CALIB_POSE_B)
    print()
    print(
        "  per-joint Δ (B − A): "
        + ", ".join(
            f"J{i + 1}={CALIB_POSE_B[i] - CALIB_POSE_A[i]:+.3f}"
            for i in range(CALIB_NUM_ARM)
        )
    )
    print(
        f"  max |Δ| = {float(np.max(np.abs(CALIB_POSE_B - CALIB_POSE_A))):.3f} rad "
        "(safety guard = 1.5 rad)"
    )
    print()
    print(
        "The robot will be moved twice (A → wait → B → wait) using\n"
        "reset_joint at ~4.5% effective dynamics.  Each move is gated by\n"
        "the 1.5 rad max_joint_delta guard."
    )
    print()
    confirm_or_exit("Proceed with calibration? [y/N]: ")

    print_banner("Step 1 — move robot to POSE A and align GELLO")
    safe_reset_to(controller, CALIB_POSE_A)
    q_robot_A = controller.get_state().wait()[0].arm_joint_position
    print(f"  robot now at A: {[round(float(x), 3) for x in q_robot_A]}", flush=True)
    _calib_wait_for_enter(
        "Physically pose the GELLO leader so it visually matches the Franka.\n"
        "Hold it steady, then press ENTER (with a free finger).",
        driver,
    )
    raw_A = _calib_read_raw_arm(driver)
    grip_A = _calib_read_raw_gripper(driver)
    print(f"\n  raw motor A: {[round(float(x), 4) for x in raw_A]}")
    print(f"  raw gripper A: {grip_A:.4f}")

    print_banner("Step 2 — move robot to POSE B and re-align GELLO")
    safe_reset_to(controller, CALIB_POSE_B)
    q_robot_B = controller.get_state().wait()[0].arm_joint_position
    print(f"  robot now at B: {[round(float(x), 3) for x in q_robot_B]}", flush=True)
    _calib_wait_for_enter(
        "Re-pose the GELLO leader so it visually matches the new Franka pose.\n"
        "Press ENTER when ready.",
        driver,
    )
    raw_B = _calib_read_raw_arm(driver)
    grip_B = _calib_read_raw_gripper(driver)
    print(f"\n  raw motor B: {[round(float(x), 4) for x in raw_B]}")
    print(f"  raw gripper B: {grip_B:.4f}")

    print_banner("Step 3 — solve signs and offsets")
    dq_robot = q_robot_B - q_robot_A
    dq_motor = raw_B - raw_A

    signs = np.where(dq_robot * dq_motor > 0, 1, -1).astype(int)
    print(
        "  Δq_robot:  "
        + " ".join(f"J{i + 1}={dq_robot[i]:+.3f}" for i in range(CALIB_NUM_ARM))
    )
    print(
        "  Δq_motor:  "
        + " ".join(f"J{i + 1}={dq_motor[i]:+.3f}" for i in range(CALIB_NUM_ARM))
    )
    print(
        "  signs:     "
        + " ".join(f"J{i + 1}={int(signs[i]):+d}" for i in range(CALIB_NUM_ARM))
    )

    ambiguous = [
        i
        for i in range(CALIB_NUM_ARM)
        if abs(dq_motor[i]) < 0.05 or abs(dq_robot[i]) < 0.1
    ]
    if ambiguous:
        print(
            colour(
                "  WARNING: ambiguous sign on joints "
                + ", ".join(f"J{i + 1}" for i in ambiguous)
                + " (Δ too small).  Re-run with poses that move that joint more.",
                "33;1",
            )
        )

    # offset[i] = raw_A[i] - sign[i] * q_robot_A[i], snapped to k*pi/2.
    offsets_raw = raw_A - signs * q_robot_A
    offsets = np.array([_calib_snap_to_half_pi(o) for o in offsets_raw])

    print()
    print("  raw offsets (before snapping):")
    print("   ", [round(float(o), 4) for o in offsets_raw])
    print("  snapped to k * π/2:")
    print("   ", [round(float(o), 4) for o in offsets])

    # Sanity: re-derive q_gello at both poses with the new calibration.
    def calib(raw: np.ndarray) -> np.ndarray:
        return signs * (raw - offsets)

    q_calib_A = calib(raw_A)
    q_calib_B = calib(raw_B)
    res_A = q_calib_A - q_robot_A
    res_B = q_calib_B - q_robot_B
    print()
    print("  residual at A: " + " ".join(f"{r:+.3f}" for r in res_A))
    print("  residual at B: " + " ".join(f"{r:+.3f}" for r in res_B))
    max_res = max(float(np.max(np.abs(res_A))), float(np.max(np.abs(res_B))))
    if max_res > 0.15:
        print(
            colour(
                f"  max residual {max_res:.3f} rad — calibration is loose.\n"
                "    Most likely you didn't physically match the robot pose closely;\n"
                "    re-pose the GELLO and re-run the offending step.",
                "33;1",
            )
        )
    else:
        print(colour(f"  max residual {max_res:.3f} rad — looks clean", "32;1"))

    grip_open_deg = math.degrees(grip_B) - 0.2
    grip_close_deg = math.degrees(grip_B) - 42.0
    print()
    print(f"  gripper at pose B raw: {grip_B:.4f} rad ({math.degrees(grip_B):.2f}°)")
    print(
        f"  suggested gripper_config:  ({CALIB_GRIPPER_ID}, "
        f"{int(round(grip_open_deg))}, {int(round(grip_close_deg))})"
    )
    print(
        "  (after pasting, manually open/close the GELLO gripper and tweak "
        "the open/close degree numbers if the binary state is wrong)"
    )

    print()
    print(colour("─" * 64, "36"))
    print(
        colour(" Paste this into gello/agents/gello_agent.py PORT_CONFIG_MAP:", "36;1")
    )
    print(colour("─" * 64, "36"))
    sign_tuple = ", ".join(str(int(s)) for s in signs)
    print(
        f'    "{gello_port}": DynamixelRobotConfig(\n'
        f"        joint_ids={tuple(CALIB_JOINT_IDS)},\n"
        f"        joint_offsets=(\n"
        + "".join(f"            {_calib_half_pi_label(o)},\n" for o in offsets)
        + f"        ),\n"
        f"        joint_signs=({sign_tuple}),\n"
        f"        gripper_config=({CALIB_GRIPPER_ID}, "
        f"{int(round(grip_open_deg))}, {int(round(grip_close_deg))}),\n"
        f"        baudrate=1_000_000,\n"
        f"    ),"
    )
    print()
    print("After pasting, restart any process that imports gello (no need to")
    print("reinstall — it's an editable install).  Then re-run align-sequential")
    print("to verify J1..J7 all stay green when you move the GELLO around.")


# ──────────────────── subcommand: reset-to-gello ──────────────────────


RESET_ALIGN_TOL = 0.08  # rad — considered aligned when all |Δ| < this


def _reset_check_limits(q: np.ndarray) -> bool:
    """Return True if all joints are within Franka limits."""
    return bool(np.all(q >= JOINT_LIMITS_LOWER) and np.all(q <= JOINT_LIMITS_UPPER))


def run_reset_to_gello(_args: argparse.Namespace) -> None:
    """Move the Franka arm to match the current GELLO joint positions."""
    controller = setup_franky()
    gello = setup_gello_expert()

    print("Reading GELLO position (averaging 20 samples) ...", flush=True)
    samples = []
    for _ in range(20):
        q, _ = gello.get_action()
        samples.append(q.copy())
        time.sleep(0.02)
    target = np.mean(samples, axis=0)

    state = controller.get_state().wait()[0]
    current = state.arm_joint_position

    delta = target - current
    max_delta = float(np.max(np.abs(delta)))

    print("─" * 64)
    print(f"  GELLO target : {fmt_joints(target)}")
    print(f"                 {fmt_deg(target)}")
    print(f"  Robot current: {fmt_joints(current)}")
    print(f"                 {fmt_deg(current)}")
    print()

    for i in range(7):
        d = delta[i]
        status = (
            colour("OK", "32")
            if abs(d) < RESET_ALIGN_TOL
            else colour(f"Δ={d:+.3f}", "33")
        )
        print(f"  J{i + 1}: {current[i]:+.3f} → {target[i]:+.3f}  ({status})")
    print()
    print(f"  Max |Δ| = {max_delta:.3f} rad ({math.degrees(max_delta):.1f}°)")
    print("─" * 64)

    if max_delta < RESET_ALIGN_TOL:
        print(colour("\n  Already aligned! No motion needed.\n", "32"))
        return

    if not _reset_check_limits(target):
        print(
            colour("\n  ERROR: GELLO target is outside Franka joint limits!", "31"),
            file=sys.stderr,
        )
        for i in range(7):
            lo, hi = JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]
            if target[i] < lo or target[i] > hi:
                print(
                    f"    J{i + 1}: {target[i]:+.3f} not in [{lo:+.3f}, {hi:+.3f}]",
                    file=sys.stderr,
                )
        sys.exit(1)

    if max_delta > 1.5:
        print(
            colour(
                f"\n  WARNING: Max delta is {max_delta:.2f} rad "
                f"({math.degrees(max_delta):.0f}°).\n"
                f"  This is a large motion. Make sure the workspace is clear!\n",
                "33",
            )
        )

    ans = input("  Move robot to GELLO position? [y/N]: ").strip().lower()
    if ans != "y":
        print("  Aborted.")
        return

    print("\n  Moving robot (slow, ~4.5% dynamics) ...", flush=True)
    controller.reset_joint(target.tolist()).wait()
    time.sleep(0.5)

    state = controller.get_state().wait()[0]
    final = state.arm_joint_position
    final_delta = target - final
    max_err = float(np.max(np.abs(final_delta)))

    print()
    for i in range(7):
        d = final_delta[i]
        status = (
            colour("ok", "32")
            if abs(d) < RESET_ALIGN_TOL
            else colour(f"err={d:+.3f}", "31")
        )
        print(f"  J{i + 1}: target={target[i]:+.3f}  actual={final[i]:+.3f}  {status}")
    print()

    if max_err < RESET_ALIGN_TOL:
        print(colour("  Aligned successfully! Ready for data collection.\n", "32"))
    else:
        print(
            colour(
                f"  Alignment residual {max_err:.3f} rad > {RESET_ALIGN_TOL} tol.\n"
                f"  You may need to re-run or check for collisions.\n",
                "33",
            )
        )


# ────────────────────────────── main ──────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gello",
        description="GELLO ↔ Franka teleop tools (alignment, calibration, reset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for environment variables.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")
    sub.add_parser(
        "align-check",
        help="Live full-arm alignment dashboard (read-only, no motion).",
    )
    sub.add_parser(
        "align-sequential",
        help="Walk operator through J1..J7 alignment (moves robot to HOME first).",
    )
    sub.add_parser(
        "calibrate",
        help="Two-pose sign + offset calibration (moves robot).",
    )
    sub.add_parser(
        "reset-to-gello",
        help="Move Franka to current GELLO pose (moves robot).",
    )

    args = parser.parse_args()
    handlers = {
        "align-check": run_align_check,
        "align-sequential": run_align_sequential,
        "calibrate": run_calibrate,
        "reset-to-gello": run_reset_to_gello,
    }
    try:
        handlers[args.cmd](args)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
