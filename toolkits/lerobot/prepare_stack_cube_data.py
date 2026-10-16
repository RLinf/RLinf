# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic Stack Cube requests and assemble audited LeRobot splits.

This CPU-only command preserves rejected requests and copies both camera videos.
Its profiles align with the installed IsaacLab task configuration, not a proven
reconstruction of the public dataset's undisclosed producer distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

OBJECTS = ("blue", "red", "green")
BASE_POSITIONS = np.array([[0.4, 0, 0.0234], [0.55, 0.05, 0.0234], [0.6, -0.1, 0.0234]])
PROFILES = (
    "smoke",
    "position",
    "position_yaw",
    "formal",
    "ood_position",
    "ood_yaw",
    "ood_visual",
)


def prepare_panda(source: Path, output: Path) -> Path:
    """Resolve bundled Panda meshes and add the control TCP to a private copy."""
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    package = source.parent.parent
    for mesh in tree.iter("mesh"):
        uri = mesh.attrib["filename"]
        resolved = package / uri.removeprefix("package://franka_description/")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        target = output / "meshes" / resolved.relative_to(package / "meshes")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(resolved, target)
        mesh.set("filename", str(target.relative_to(output)))
    # Copy material textures referenced by DAE files as well.
    shutil.copytree(package / "meshes", output / "meshes", dirs_exist_ok=True)
    robot = tree.getroot()
    if robot.attrib.get("name") != "panda":
        raise ValueError("Expected the Panda asset, not an FR3 robot.")
    # The task drives both finger joints explicitly. Avoid also imposing the
    # URDF importer's mimic constraint on the same actuated pair.
    for finger in robot.findall("joint"):
        if finger.attrib["name"] in {"panda_finger_joint1", "panda_finger_joint2"}:
            mimic = finger.find("mimic")
            if mimic is not None:
                finger.remove(mimic)
    ET.SubElement(robot, "link", {"name": "panda_tcp"})
    joint = ET.SubElement(robot, "joint", {"name": "panda_tcp_joint", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "panda_hand"})
    ET.SubElement(joint, "child", {"link": "panda_tcp"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0.107", "rpy": "0 0 0"})
    destination = output / "panda.urdf"
    tree.write(destination, encoding="unicode", xml_declaration=True)
    return destination


def sample_request(seed: int, split: str, profile: str) -> dict:
    """Sample one initial state; reject invalid layouts instead of clipping."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown spatial profile: {profile}")
    if 42 <= seed <= 49:
        raise ValueError(
            "Seeds 42–49 belong exclusively to the preview regression set."
        )
    rng = np.random.default_rng(seed)
    positions = BASE_POSITIONS.copy()
    yaw = np.zeros(3)
    if profile == "smoke":
        positions[:, :2] += rng.uniform(-0.005, 0.005, (3, 2))
    else:
        lower, upper = (
            ([0.35, -0.15], [0.65, 0.15])
            if profile == "ood_position"
            else ([0.4, -0.1], [0.6, 0.1])
        )
        for _ in range(10000):
            xy = np.empty((3, 2))
            for index in range(3):
                for _attempt in range(10000):
                    candidate = rng.uniform(lower, upper)
                    if all(
                        np.linalg.norm(candidate - xy[previous]) > 0.1
                        for previous in range(index)
                    ):
                        xy[index] = candidate
                        break
                else:
                    raise RuntimeError("Unable to sample a separated object position.")
            if profile == "ood_position" and not np.any(
                (xy < [0.4, -0.1]) | (xy > [0.6, 0.1])
            ):
                continue
            positions[:, :2] = xy
            break
        else:
            raise RuntimeError("Unable to sample an out-of-distribution layout.")
        if profile != "position":
            yaw = rng.uniform(-1.0, 1.0, 3)
        if profile == "ood_yaw":
            yaw[int(rng.integers(3))] = rng.choice([-1, 1]) * rng.uniform(1.0, 1.4)
    offsets = (
        rng.normal(0, 0.02, 7)
        if profile in ("formal", "ood_position", "ood_yaw", "ood_visual")
        else np.zeros(7)
    )
    request = {
        "request_id": f"{profile}-{split}-{seed}",
        "seed": seed,
        "split": split,
        "profile": profile,
        "objects": {
            name: {"position": positions[i].tolist(), "yaw_rad": float(yaw[i])}
            for i, name in enumerate(OBJECTS)
        },
        "arm_joint_offsets": offsets.tolist(),
        "visual": None,
    }
    if profile == "ood_visual":
        request["visual"] = {
            "light_intensity": float(rng.uniform(1.5, 4.5)),
            "table_rgb": rng.uniform(0.05, 0.45, 3).tolist(),
        }
    return request


def create_manifest(
    output: Path, profile: str, split: str, seed: int, episodes: int
) -> None:
    """Freeze a request list before observing any expert or policy outcome."""
    if output.exists():
        raise FileExistsError(output)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    requests = [sample_request(seed + i, split, profile) for i in range(episodes)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "distribution_basis": "installed IsaacLab task configuration; public dataset producer unverified",
                "position_frame": "robot_base",
                "object_order": list(OBJECTS),
                "preview_seeds_excluded": list(range(42, 50)),
                "requests": requests,
            },
            indent=2,
        )
        + "\n"
    )


def validate_episode(root: Path, episode_index: int = 0) -> dict:
    """Validate frame timing, numeric fields and the decoded two-camera videos."""
    frame = pd.read_parquet(
        root
        / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
    )
    count = len(frame)
    lengths = {
        row["episode_index"]: row["length"]
        for row in (
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        )
    }
    if count == 0 or count != lengths[episode_index]:
        raise ValueError(f"Invalid episode frame count: {root}")
    if not np.array_equal(frame.frame_index, np.arange(count)) or not np.allclose(
        frame.timestamp, np.arange(count) / 20, atol=1e-7
    ):
        raise ValueError(f"Invalid timing: {root}")
    for key, dimension in (("observation.state", 8), ("action", 7)):
        values = np.stack(frame[key])
        if values.shape != (count, dimension) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}: {root}")
    if not np.isin(np.stack(frame.action)[:, -1], [-1, 1]).all():
        raise ValueError(f"Non-binary gripper: {root}")
    videos = {}
    for camera in ("front", "wrist"):
        path = (
            root
            / f"videos/chunk-{episode_index // 1000:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
        )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if (
            stream["width"],
            stream["height"],
            int(stream["nb_read_frames"]),
            stream["r_frame_rate"],
        ) != (256, 256, count, "20/1"):
            raise ValueError(f"Camera timing or shape mismatch: {path}: {stream}")
        videos[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"frames": count, "video_sha256": videos}
    report_path = root / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("request") and "initial_state" in report:
            requested = np.array(
                [report["request"]["objects"][uid]["position"] for uid in OBJECTS]
            )
            actual = np.array(
                [
                    np.asarray(report["initial_state"]["object_poses"][uid])[:3, 3]
                    for uid in OBJECTS
                ]
            )
            error = float(np.abs(requested - actual).max())
            if error > 1e-6:
                raise ValueError(
                    f"Recorded initial pose differs from request: {root}: {error}"
                )
            result["initial_position_max_error_m"] = error
            measurements = json.loads((root / "measurements.json").read_text())
            result["first_step_max_xy_displacement_m"] = float(
                np.abs(
                    np.asarray(measurements[0]["positions"])[:, :2] - actual[:, :2]
                ).max()
            )
    return result


def aggregate_stats(episodes: list[dict]) -> dict:
    """Combine per-episode moments with frame-count weighting."""
    result = {}
    for key in episodes[0]:
        parts = [episode[key] for episode in episodes]
        count = sum(part["count"][0] for part in parts)
        mean = (
            sum(np.asarray(part["mean"]) * part["count"][0] for part in parts) / count
        )
        second = (
            sum(
                (np.asarray(part["std"]) ** 2 + np.asarray(part["mean"]) ** 2)
                * part["count"][0]
                for part in parts
            )
            / count
        )
        result[key] = {
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(second - mean**2, 0)).tolist(),
            "min": np.min([part["min"] for part in parts], axis=0).tolist(),
            "max": np.max([part["max"] for part in parts], axis=0).tolist(),
            "count": [count],
        }
    return result


def assemble_dataset(
    manifest: Path, collection: Path, output: Path, split: str
) -> dict:
    """Copy qualified requests into one split and preserve every attempt in audit."""
    if output.exists():
        raise FileExistsError(output)
    run_manifest = collection / "run_manifest.json"
    if run_manifest.is_file():
        status = json.loads(run_manifest.read_text())
        if status.get("camera_superseded") or status.get("training_ready") is False:
            raise ValueError(
                "Collection is not training-ready; its camera or task contract was superseded."
            )
    requests = json.loads(manifest.read_text())["requests"]
    requested = {
        request["request_id"]: request
        for request in requests
        if request["split"] == split
    }
    if not requested:
        raise ValueError(f"No requests for split {split}")
    selected = [request for request in requests if request["split"] == split]
    if len(requested) != len(selected) or len(
        {request["seed"] for request in selected}
    ) != len(selected):
        raise ValueError("Split requests must have unique IDs and seeds.")
    if any(42 <= request["seed"] <= 49 for request in selected):
        raise ValueError(
            "Preview regression seeds cannot enter a training or test split."
        )
    seen, accepted, audit = set(), [], []
    for path in sorted(collection.glob("episode_*/report.json")):
        report = json.loads(path.read_text())
        request = report.get("request")
        if request is None or request["request_id"] not in requested:
            continue
        key = request["request_id"]
        if key in seen or request != requested[key]:
            raise ValueError(f"Duplicate or modified request: {key}")
        seen.add(key)
        validation = validate_episode(path.parent)
        qualified = bool(report["passed_450"])
        audit.append(
            {
                "request_id": key,
                "seed": request["seed"],
                "accepted": qualified,
                "report_path": str(path.resolve()),
                **validation,
            }
        )
        if qualified:
            accepted.append((path.parent, report))
    missing = sorted(set(requested) - seen)
    if missing:
        raise ValueError(f"Unexecuted requests; refuse an incomplete split: {missing}")
    if not accepted:
        raise ValueError("No qualified demonstrations; inspect the collection reports.")
    output.mkdir(parents=True)
    meta = output / "meta"
    meta.mkdir()
    info = json.loads((accepted[0][0] / "meta/info.json").read_text())
    episode_rows, stats_rows, all_stats = [], [], []
    offset = 0
    for index, (source, report) in enumerate(accepted):
        frame = pd.read_parquet(source / "data/chunk-000/episode_000000.parquet")
        frame["episode_index"] = index
        frame["index"] = np.arange(offset, offset + len(frame))
        offset += len(frame)
        destination = (
            output / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        for camera in ("front", "wrist"):
            key = f"observation.images.{camera}"
            target = (
                output
                / f"videos/chunk-{index // 1000:03d}/{key}/episode_{index:06d}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / f"videos/chunk-000/{key}/episode_000000.mp4", target)
        task = json.loads((source / "meta/tasks.jsonl").read_text())["task"]
        episode_rows.append(
            {
                "episode_index": index,
                "tasks": [task],
                "length": len(frame),
                "request_id": report["request"]["request_id"],
                "seed": report["seed"],
            }
        )
        stats = json.loads((source / "meta/stats.json").read_text())
        all_stats.append(stats)
        stats_rows.append({"episode_index": index, "stats": stats})
    info.update(
        total_episodes=len(accepted),
        total_frames=offset,
        total_videos=2 * len(accepted),
        total_chunks=(len(accepted) + 999) // 1000,
        splits={"train": f"0:{len(accepted)}"},
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    shutil.copy2(accepted[0][0] / "meta/tasks.jsonl", meta / "tasks.jsonl")
    for name, rows in (("episodes", episode_rows), ("episodes_stats", stats_rows)):
        (meta / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (meta / "stats.json").write_text(json.dumps(aggregate_stats(all_stats), indent=2))
    result = {
        "split": split,
        "requested": len(requested),
        "accepted": len(accepted),
        "frames": offset,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "attempts": audit,
        "normalization_source_allowed": split == "train",
    }
    (meta / "collection_audit.json").write_text(json.dumps(result, indent=2))
    shutil.copy2(manifest, meta / "request_manifest.json")
    return result


def split_reference(source: Path, output: Path, seed: int = 0) -> None:
    """Freeze and copy 120/27 reference episodes with independent statistics."""
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((source / "meta/info.json").read_text())
    if info["total_episodes"] != 147:
        raise ValueError("Expected the complete 147-episode reference dataset.")
    episodes = {
        row["episode_index"]: row
        for row in (
            json.loads(line)
            for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        )
    }
    episode_stats = {
        row["episode_index"]: row["stats"]
        for row in (
            json.loads(line)
            for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines()
        )
    }
    permutation = np.random.default_rng(seed).permutation(147)
    validated = {index: validate_episode(source, index) for index in range(147)}
    for split, indices in (("train", permutation[:120]), ("val", permutation[120:])):
        root = output / split
        meta = root / "meta"
        meta.mkdir(parents=True)
        offset, rows, stats_rows, stats = 0, [], [], []
        for index, original in enumerate(indices.tolist()):
            data = pd.read_parquet(
                source / f"data/chunk-000/episode_{original:06d}.parquet"
            )
            data["episode_index"] = index
            data["index"] = np.arange(offset, offset + len(data))
            offset += len(data)
            target = root / f"data/chunk-000/episode_{index:06d}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(target, index=False)
            for camera in ("front", "wrist"):
                key = f"observation.images.{camera}"
                target = root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    source / f"videos/chunk-000/{key}/episode_{original:06d}.mp4",
                    target,
                )
            rows.append(
                episodes[original]
                | {"episode_index": index, "source_episode_index": original}
            )
            stats.append(episode_stats[original])
            stats_rows.append(
                {"episode_index": index, "stats": episode_stats[original]}
            )
        updated = info | {
            "total_episodes": len(indices),
            "total_frames": offset,
            "total_videos": 2 * len(indices),
            "total_chunks": 1,
            "splits": {"train": f"0:{len(indices)}"},
        }
        (meta / "info.json").write_text(json.dumps(updated, indent=2))
        for name, values in (("episodes", rows), ("episodes_stats", stats_rows)):
            (meta / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in values)
            )
        (meta / "stats.json").write_text(json.dumps(aggregate_stats(stats), indent=2))
        shutil.copy2(source / "meta/tasks.jsonl", meta / "tasks.jsonl")
        if (source / "meta/modality.json").exists():
            shutil.copy2(source / "meta/modality.json", meta / "modality.json")
        (meta / "collection_audit.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "normalization_source_allowed": split == "train",
                    "source": str(source.resolve()),
                    "source_revision": json.loads(
                        (source / "source_revision.json").read_text()
                    ),
                    "split_seed": seed,
                    "source_episode_indices": indices.tolist(),
                    "accepted": len(indices),
                    "frames": offset,
                    "episodes": [
                        validated[index] | {"source_episode_index": index}
                        for index in indices.tolist()
                    ],
                },
                indent=2,
            )
        )


def main() -> None:
    """Prepare requests or assemble a complete collection without starting GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--profile", choices=PROFILES, required=True)
    manifest.add_argument("--split", required=True)
    manifest.add_argument("--seed", type=int, required=True)
    manifest.add_argument("--episodes", type=int, required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--manifest", type=Path, required=True)
    assemble.add_argument("--collection", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--split", required=True)
    panda = commands.add_parser("prepare-panda")
    panda.add_argument("--source", type=Path, required=True)
    panda.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser("split-reference")
    reference.add_argument("--source", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "manifest":
        create_manifest(**args)
    elif command == "assemble":
        assemble_dataset(**args)
    elif command == "prepare-panda":
        prepare_panda(**args)
    else:
        split_reference(**args)


if __name__ == "__main__":
    main()
