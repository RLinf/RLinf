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

"""Compute the SO101 end-effector pose via forward kinematics.

Two modes:

* **Live (default)** — connect to a real SO101 arm, read the current joint
  angles, and print the end-effector ``(x, y, z)`` in metres.
* **Offline** — supply joint angles on the command line and get the EE pose
  without any hardware.

Typical usage (live)
--------------------

.. code-block:: bash

   # Calibrated arm with default URDF
   python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0

   # Custom URDF (e.g. from lerobot-calibrate)
   python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0 \\
       --urdf ~/.cache/huggingface/lerobot/calibration/robots/so_follower/…/so101.urdf

Typical usage (offline)
-----------------------

.. code-block:: bash

   python toolkits/lerobot/compute_ee_pose.py \\
       --joints 30.0 -60.0 120.0 0.0 30.0

   python toolkits/lerobot/compute_ee_pose.py \\
       --joints 0.0 -45.0 90.0 0.0 0.0
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional, Sequence


_SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
_NUM_ARM_JOINTS = len(_SO101_ARM_JOINTS)
_DEFAULT_URDF = "pack://lerobot/robots/so_follower/so101.urdf"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute SO101 end-effector pose via forward kinematics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples
          # Live — read current joint angles from the arm
          python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0

          # Offline — compute from known joint angles
          python toolkits/lerobot/compute_ee_pose.py --joints 10.0 20.0 90.0 0.0 30.0
        """),
    )
    live_group = p.add_argument_group("live (real hardware)")
    live_group.add_argument(
        "--port",
        default=None,
        help="Serial port of the follower arm (e.g. /dev/ttyACM0).",
    )
    live_group.add_argument(
        "--calibration-id",
        default="default",
        help="Calibration lookup id (matches the id field in "
        "SO101FollowerConfig; lerobot resolves this to "
        "~/.cache/huggingface/lerobot/calibration/robots/so_follower/"
        "<id>/calibration.json).  Default: %(default)s.",
    )

    offline_group = p.add_argument_group("offline")
    offline_group.add_argument(
        "--joints",
        type=float,
        nargs=_NUM_ARM_JOINTS,
        metavar="DEG",
        help=f"Joint angles in degrees ({' '.join(_SO101_ARM_JOINTS)}).",
    )

    p.add_argument(
        "--urdf",
        default=_DEFAULT_URDF,
        help="Path or pack:// URI for the SO101 URDF (default: %(default)s).",
    )
    p.add_argument(
        "--no-fk-check",
        action="store_true",
        help="Skip the FK-placo import check (useful in CI with no placo installed).",
    )
    return p


def _connect_robot(port: str, calibration_id: str) -> "tuple[object, list[float]]":
    """Connect to the follower arm and return ``(robot, joint_angles_deg)``."""
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    config = SO101FollowerConfig(port=port, id=calibration_id)
    robot = SO101Follower(config)
    # calibrate=True loads the saved calibration JSON at
    # ~/.cache/.../calibration/robots/so_follower/<id>/calibration.json
    # if one exists, or runs the interactive wizard for a first-time
    # setup.  calibrate=False skips JSON loading entirely and causes
    # `get_observation()` to fail.
    robot.connect(calibrate=True)

    obs = robot.get_observation()
    joints = [
        float(obs[f"{name}.pos"])
        for name in _SO101_ARM_JOINTS
    ]
    return robot, joints


def _run_fk(joint_angles_deg: Sequence[float], urdf_path: str) -> "tuple[float, float, float]":
    """Forward kinematics → ``(x, y, z)`` in metres."""
    import numpy as np

    from lerobot.model.kinematics import RobotKinematics

    # Resolve pack:// URIs against the installed lerobot package.
    resolved = _resolve_urdf(str(urdf_path))

    fk = RobotKinematics(
        urdf_path=resolved,
        target_frame_name="gripper_frame_link",
    )
    T = fk.forward_kinematics(np.asarray(joint_angles_deg, dtype=np.float64))
    x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    return x, y, z


def _resolve_urdf(urdf_path: str) -> str:
    """Resolve a ``pack://lerobot/...`` URI to a local path."""
    if urdf_path.startswith("pack://"):
        parts = urdf_path[len("pack://"):].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid pack:// URI: {urdf_path!r}")
        pkg_name, rel_path = parts
        from importlib.resources import files
        return str(files(pkg_name) / rel_path)

    if (p := Path(urdf_path).expanduser()).is_file():
        return str(p.resolve())

    raise FileNotFoundError(f"URDF not found: {urdf_path!r}")


def _check_fk_available() -> Optional[str]:
    """Return ``None`` if FK can be imported, or an error message."""
    try:
        from lerobot.model.kinematics import RobotKinematics  # noqa: F401
        return None
    except ImportError as e:
        return f"Cannot import lerobot.model.kinematics: {e}"


def main() -> None:
    args = _build_parser().parse_args()

    is_live = args.port is not None
    is_offline = args.joints is not None

    if not is_live and not is_offline:
        print(
            "Either --port (live) or --joints (offline) is required.  "
            "Use -h for help."
        )
        raise SystemExit(2)

    if is_live and is_offline:
        print("--port and --joints are mutually exclusive.")
        raise SystemExit(2)

    if args.no_fk_check:
        # Offline-only convenience — print angles without touching
        # lerobot or placo.  Handy for CI / smoke-testing the CLI.
        if is_live:
            print("--no-fk-check is only meaningful with --joints (offline mode).")
            raise SystemExit(2)
        joints = list(args.joints)  # type: ignore[arg-type]
        print(f"Joint angles: {[round(j, 1) for j in joints]}")
        print("(FK skipped via --no-fk-check)")
        return

    err = _check_fk_available()
    if err is not None:
        print(f"ERROR: {err}")
        print("Hint: install lerobot with `pip install lerobot` and then `pip install placo`.")
        raise SystemExit(3)

    # ---- Live ----
    if is_live:
        robot, joints = _connect_robot(args.port, args.calibration_id)
        try:
            x, y, z = _run_fk(joints, args.urdf)
        finally:
            robot.disconnect()
        print(f"Joint angles: {[round(j, 1) for j in joints]}")
        print(f"EE pose (m):  ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"\nYAML snippet:\n  target_ee_pose: [{x:.3f}, {y:.3f}, {z:.3f}]")
        return

    # ---- Offline ----
    joints = list(args.joints)  # type: ignore[arg-type]
    print(f"Joint angles: {[round(j, 1) for j in joints]}")
    x, y, z = _run_fk(joints, args.urdf)
    print(f"EE pose (m):  ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"\nYAML snippet:\n  target_ee_pose: [{x:.3f}, {y:.3f}, {z:.3f}]")


if __name__ == "__main__":
    main()
