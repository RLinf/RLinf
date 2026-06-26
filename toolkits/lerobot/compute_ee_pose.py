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

   # Calibrated arm (download the URDF first — see --urdf help)
   python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0 \\
       --calibration-id my_awesome_follower_arm \\
       --urdf ~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf

   # Or use --calibration-file with a flat calibration JSON
   python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0 \\
       --calibration-file ~/.cache/huggingface/lerobot/calibration/.../my_arm.json \\
       --urdf ~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf

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
_DEFAULT_URDF = "~/.cache/huggingface/lerobot/urdf/so101.urdf"


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
    live_group.add_argument(
        "--calibration-file",
        default=None,
        metavar="PATH",
        help="Absolute path to a calibration JSON file.  Use this when "
        "the calibration was saved as a flat file rather than the "
        "standard lerobot <id>/calibration.json directory layout.  "
        "Takes precedence over --calibration-id.",
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
        help="Path to the SO101 URDF file.  The SO101 URDF is not bundled "
        "with lerobot; download it from the SO-ARM100 repo: "
        "https://github.com/TheRobotStudio/SO-ARM100/blob/main/"
        "Simulation/SO101/so101_new_calib.urdf  "
        "Default: %(default)s.",
    )
    p.add_argument(
        "--no-fk-check",
        action="store_true",
        help="Skip the FK-placo import check (useful in CI with no placo installed).",
    )
    return p


def _connect_robot(
    port: str, calibration_id: str, *, calibration_file: str | None = None
) -> "tuple[object, list[float]]":
    """Connect to the follower arm and return ``(robot, joint_angles_deg)``."""
    from pathlib import Path
    import shutil

    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    if calibration_file is not None:
        # lerobot always looks for
        #   <calibration_dir>/<id>/calibration.json
        # If the user has a flat .json file we create the expected
        # directory layout on the fly.
        cf = Path(calibration_file).expanduser().resolve()
        calib_id = cf.stem  # "my_awesome_follower_arm"
        calib_dir = cf.parent  # the so_follower/ dir
        target = calib_dir / calib_id / "calibration.json"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cf, target)
            print(f"Staged calibration: {target}")
        config = SO101FollowerConfig(
            port=port, id=calib_id, calibration_dir=calib_dir,
        )
    else:
        config = SO101FollowerConfig(port=port, id=calibration_id)

    robot = SO101Follower(config)
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
        joint_names=list(_SO101_ARM_JOINTS),
    )
    T = fk.forward_kinematics(np.asarray(joint_angles_deg, dtype=np.float64))
    x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    return x, y, z


def _resolve_urdf(urdf_path: str) -> str:
    """Resolve a URDF path, falling back to directory listing."""
    p = Path(urdf_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    # placo also accepts a directory containing robot.urdf.
    if p.is_dir() and (p / "robot.urdf").is_file():
        return str(p.resolve())
    raise FileNotFoundError(
        f"URDF not found at {p}\n"
        "The SO101 URDF is not bundled with lerobot. Download it:\n"
        "  - Option 1: git clone https://github.com/TheRobotStudio/SO-ARM100\n"
        "    then pass --urdf <path>/SO-ARM100/Simulation/SO101/so101_new_calib.urdf\n"
        "  - Option 2: download just the URDF file:\n"
        "    curl -L -o so101_new_calib.urdf https://raw.githubusercontent.com/"
        "TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf\n"
        "    then pass --urdf ./so101_new_calib.urdf"
    )


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
        robot, joints = _connect_robot(
            args.port, args.calibration_id, calibration_file=args.calibration_file,
        )
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
