"""Read the installed LoopInfra configuration; never instantiate its SDK owner."""

import argparse
import json
import os
import sys
from pathlib import Path


def inspect(root: Path, speed: str) -> dict:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import yaml
    from loopinfra.client.policy_action import contact_policy_capability
    from loopinfra.pico_commissioning import (
        digest,
        load_commissioning,
        precision_pose_config,
        resolve_contact_hold_policy,
    )
    from loopinfra.pose_teleop import pose_profile_id
    from loopinfra.reset_pose import load_reset_pose
    from types import SimpleNamespace

    commissioning = load_commissioning(root / "config/tianji_pico_commissioning.json")
    pose = precision_pose_config(
        commissioning,
        ("A", "B"),
        speed,
        control_mode="joint_position",
        elbow_guard=True,
        backend_profile="wuji_compat",
        contact_assist=True,
        feedback_recovery_mode="feedback_brake_v1",
        hold_policy=resolve_contact_hold_policy(
            True, None, backend_profile="wuji_compat"
        ),
    )
    contact = pose["contact_assist"]
    capability = contact_policy_capability(
        SimpleNamespace(
            joint_compliance=True,
            joint_pd=False,
            pose_config=pose,
            sdk_control_mode=contact["native_mode"],
            contact_config=contact,
        )
    )
    if capability != "finite_cartesian_impedance_v1":
        raise RuntimeError(
            "LoopInfra finite contact policy compatibility is unavailable"
        )
    collection = yaml.safe_load((root / "config/tianji_collection.yaml").read_text())
    reset_path = (root / collection["reset_pose"]).resolve()
    reset = load_reset_pose(
        reset_path, backend="hardware", commissioning_id=digest(commissioning)
    )
    app = (root / "loopinfra/server/app.py").read_text()
    for endpoint in ("/api/tianji/execution", "/api/collection", "/api/commands"):
        if endpoint not in app:
            raise RuntimeError("Required LoopInfra endpoint missing: " + endpoint)
    launcher = (root / "loopinfra/collector_launcher.py").read_text()
    for flag in ("--client-config", "--collection-config"):
        if flag not in launcher:
            raise RuntimeError(
                "LoopInfra collector configuration override missing: " + flag
            )
    protocol = (root / "loopinfra/policy_action.py").read_text()
    if "loopinfra.policy_pose.v1" not in protocol:
        raise RuntimeError("Required policy_pose.v1 protocol missing")
    return {
        "backend_profile": pose["backend_profile"],
        "sdk_control_mode": "joint_position",
        "actual_sdk_control_mode": contact["native_mode"],
        "contact_assist_policy": contact["policy"],
        "policy_pose_contact": capability,
        "hold_policy": pose["hold_policy"],
        "feedback_recovery_mode": "feedback_brake_v1",
        "sync_frequency_hz": int(float(os.environ.get("TIANJI_CONTROL_HZ", "10"))),
        "collection_profile": "rlt_synchronized_observations",
        "sdk_default_collection_mode": collection.get("mode", "synchronized"),
        "speed_profile": speed,
        "pose_config_id": pose_profile_id(pose),
        "commissioning_id": digest(commissioning),
        "reset_pose_id": reset["id"],
        "reset_pose_file": str(reset_path),
        "camera_roles": sorted(collection["camera_roles"]),
        "hardware_contacted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--speed", choices=("trial45",), default="trial45")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root.resolve(), args.speed), ensure_ascii=False))
