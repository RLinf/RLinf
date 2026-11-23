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

"""Bridge strictly selected Isaac Lab episodes to RLinf's LeRobot schema."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

CAMERAS = {"image": "base_camera", "wrist_image": "wrist_camera"}


def file_sha256(path: Path) -> str:
    """Hash raw evidence without loading camera arrays into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawEpisode:
    """Read one hash-bound episode with observations before executed actions."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.source = Path(record["source"])
        self.length = int(record["length"])
        required = {
            "trajectory.npz",
            *(f"cameras/{name}.npy" for name in CAMERAS.values()),
        }
        if not required.issubset(record["source_sha256"]):
            raise ValueError("Selection must bind the trajectory and both camera files")
        for name, expected in record["source_sha256"].items():
            path = self.source / name
            if not path.resolve().is_relative_to(self.source.resolve()):
                raise ValueError(f"Evidence path leaves the episode: {name}")
            if file_sha256(path) != expected:
                raise ValueError(f"Source evidence changed: {path}")
        with np.load(self.source / "trajectory.npz", allow_pickle=False) as raw:
            self.qpos = raw["qpos"]
            self.actions = raw["action_targets"]
            targets = raw["position_targets"]
            times = raw["sim_time"]
        n = self.length
        if n < 1 or self.qpos.shape != (n + 1, 9) or self.actions.shape != (n, 9):
            raise ValueError(
                f"Expected N+1 joint observations and N actions: {self.source}"
            )
        if not np.isfinite(self.qpos).all() or not np.isfinite(self.actions).all():
            raise ValueError(f"Nonfinite joints or actions: {self.source}")
        np.testing.assert_array_equal(self.actions, targets[1:])
        np.testing.assert_allclose(
            times - times[0], np.arange(n + 1) / 60, atol=2e-6, rtol=0
        )
        self.images = {
            key: np.load(
                self.source / "cameras" / f"{camera}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for key, camera in CAMERAS.items()
        }
        if any(
            array.shape != (n + 1, 256, 256, 3) or array.dtype != np.uint8
            for array in self.images.values()
        ):
            raise ValueError(
                f"Expected N+1 dual 256x256 uint8 RGB observations: {self.source}"
            )

    @staticmethod
    def joint_state(values: np.ndarray) -> np.ndarray:
        """Map seven arm joints and two finger coordinates to total opening."""
        return np.concatenate((values[:7], [values[7:9].sum()])).astype(np.float32)

    def frames(self) -> Iterator[dict[str, Any]]:
        """Yield each pre-action observation and its actual position target."""
        for t in range(self.length):
            yield {
                "state": self.joint_state(self.qpos[t]),
                "actions": self.joint_state(self.actions[t]),
                "task": self.record["task"],
                "done": np.array([t == self.length - 1], dtype=bool),
                "is_success": np.array([True], dtype=bool),
                **{
                    key: np.array(array[t], copy=True)
                    for key, array in self.images.items()
                },
            }

    def save_evidence(self, destination: Path) -> None:
        """Keep physical receipts and terminal observations alongside training data."""
        destination.mkdir(parents=True)
        for name in self.record["source_sha256"]:
            if not name.startswith("cameras/"):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.source / name, target)
        for key, array in self.images.items():
            np.save(destination / f"terminal_{key}.npy", array[-1], allow_pickle=False)


def verify_selection(selection: dict[str, Any]) -> None:
    """Require the runtime's strict selector and bind its validation inputs."""
    if (
        selection["status"] != "selected_before_export"
        or selection.get("static_and_motion_verified") is not True
        or not selection["episodes"]
    ):
        raise ValueError(
            "Export requires nonempty, strictly verified episode selection"
        )
    for name, expected in selection["original_inputs_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Selection input changed: {name}")


def audit_dataset(selection: dict[str, Any], root: Path, repo_id: str) -> int:
    """Read every official-loader row and compare it with its raw source."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data = LeRobotDataset(repo_id, root=root)
    offset = 0
    for episode_index, record in enumerate(selection["episodes"]):
        episode = RawEpisode(record)
        for frame_index, expected in enumerate(episode.frames()):
            row = data[offset]
            for key in ("state", "actions", "done", "is_success"):
                np.testing.assert_array_equal(
                    row[key].numpy().reshape(-1), expected[key]
                )
            for key in CAMERAS:
                actual = (
                    torch.round(row[key].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
                )
                np.testing.assert_array_equal(actual, expected[key])
            if (
                row["task"] != record["task"]
                or int(row["episode_index"]) != episode_index
                or int(row["frame_index"]) != frame_index
            ):
                raise ValueError(
                    f"Dataset task or episode boundary mismatch at row {offset}"
                )
            np.testing.assert_allclose(
                float(row["timestamp"]), frame_index / 60, atol=2e-6, rtol=0
            )
            offset += 1
    if len(data) != offset:
        raise ValueError(
            f"Expected {offset} rows; official loader returned {len(data)}"
        )
    return offset


def export_selection(selection_path: Path, output: Path, repo_id: str) -> None:
    """Publish a local LeRobot dataset only after complete source readback.

    Interrupted exports retain a sibling ``.partial`` directory. A fresh output
    path is required for another export; the completed directory is atomic.
    """
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    selection = json.loads(selection_path.read_text())
    verify_selection(selection)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(
            f"Export destination or partial export already exists: {output}"
        )
    partial.mkdir(parents=True)
    shutil.copy2(selection_path, partial / "SELECTION.json")
    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=repo_id,
        root=partial / "dataset",
        robot_type="panda",
        fps=60,
        state_dim=8,
        action_dim=8,
        wrist_image_keys={"wrist_image": (256, 256, 3)},
        has_intervene_flag=False,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    try:
        for index, record in enumerate(selection["episodes"]):
            episode = RawEpisode(record)
            for frame in episode.frames():
                writer.add_frame(frame)
            writer.save_episode()
            episode.save_evidence(partial / "sidecars" / f"episode_{index:06d}")
    finally:
        writer.finalize()
    rows = audit_dataset(selection, partial / "dataset", repo_id)
    audit = {
        "status": "passed",
        "episodes": len(selection["episodes"]),
        "frames": rows,
        "selection_sha256": file_sha256(selection_path),
        "state_action": "seven arm joints (rad), total finger opening (m); absolute position targets",
        "observation_action_alignment": "observation[t], executed_target[t]; terminal observation retained in sidecars",
        "fps": 60,
        "physics_hz": 480,
        "readback": "all states, actions, images, tasks, timestamps, done/success flags, and episode/frame indices",
    }
    (partial / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    partial.rename(output)
