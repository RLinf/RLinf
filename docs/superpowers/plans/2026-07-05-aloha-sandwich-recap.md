# ALOHA Sandwich RECAP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an RLinf-native RECAP path for ALOHA sandwich off-policy fine-tuning from raw HITL HDF5 data and a converted pi0.5 SFT checkpoint.

**Architecture:** Keep the merged RECAP four-stage pipeline intact and add `aloha` as another robot type at the data conversion, return labeling, value-transform, advantage-inference, and CFG config boundaries. Store RECAP labels as LeRobot sidecars and metadata so the raw dataset semantics remain ALOHA-specific rather than disguised as LIBERO or Franka. Convert the OpenPI JAX checkpoint outside RLinf before CFG training and fail fast when the configured PyTorch checkpoint or HF value backbones are missing.

**Tech Stack:** Python, Hydra/OmegaConf, PyArrow/Parquet, pandas, NumPy, h5py, LeRobot v2.1, OpenPI ALOHA transforms, PyTorch/FSDP, Sphinx RST.

---

## File Map

- Create: `examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py`
  - CLI and testable helpers to convert raw ALOHA HDF5 episodes into LeRobot v2.1 with `is_success`, `teleop_mask`, optional velocity/effort, and `meta/hil_segments.json`.
- Create: `tests/unit_tests/test_aloha_recap_converter.py`
  - Lightweight HDF5 helper tests for reward, teleop mask, episode loading, and audit metadata.
- Create: `tests/unit_tests/test_aloha_recap_returns.py`
  - HITL-aware return split tests for successful rescued episodes and failed intervention episodes.
- Create: `tests/unit_tests/test_aloha_recap_transforms.py`
  - Unit tests for ALOHA value dataset transforms, checkpoint transforms, `build_obs`, and teleop label override.
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
  - Add `data.hitl_aware_returns`, optional `teleop_mask` reading, and successful-episode split logic.
- Modify: `rlinf/data/datasets/recap/value_dataset.py`
  - Add `_REPACK_KEYS["aloha"]` and `aloha_policy.AlohaInputs` support.
- Modify: `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py`
  - Add `env_type == "aloha"` value-checkpoint inference transforms.
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`
  - Add ALOHA observation construction, carry `teleop_mask`, override only boolean labels, and emit `teleop_positive`.
- Create: `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`
  - Step 1 config for the converted sandwich rollout dataset.
- Create: `examples/offline_rl/config/aloha_sandwich_recap_value_model_sft.yaml`
  - Step 2 config for ALOHA value SFT.
- Create: `examples/offline_rl/config/aloha_sandwich_recap_compute_advantages.yaml`
  - Step 3 config for ALOHA advantage inference.
- Create: `examples/offline_rl/config/aloha_sandwich_cfg_rl_openpi.yaml`
  - Step 4 CFG config using `pi05_aloha_robotwin`.
- Modify: `docs/source-en/rst_source/examples/embodied/recap.rst`
  - Add an ALOHA sandwich subsection with conversion, checkpoint conversion, and four-stage commands.
- Modify: `docs/source-zh/rst_source/examples/embodied/recap.rst`
  - Add the matching Chinese subsection.

## Task 1: ALOHA Converter Helper Tests

**Files:**
- Create: `tests/unit_tests/test_aloha_recap_converter.py`
- Create later in this task: `examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py`

- [ ] **Step 1: Write the failing converter helper tests**

Create `tests/unit_tests/test_aloha_recap_converter.py` with this content:

```python
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.offline_rl.data.convert_aloha_hdf5_to_lerobot_v21 import (
    _aloha_features,
    _build_hil_segments_entry,
    _episode_success_array,
    _load_episode_payload,
    _teleop_mask,
)


def _write_episode(path: Path, reward: float = 1.0) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("action", data=np.arange(56, dtype=np.float32).reshape(4, 14))
        f.create_dataset("reward", data=np.asarray(reward, dtype=np.float32))
        f.create_dataset(
            "teleop_segments",
            data=np.asarray([[1, 3]], dtype=np.int64),
        )

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.ones((4, 14), dtype=np.float32))
        obs.create_dataset("qvel", data=np.full((4, 14), 2.0, dtype=np.float32))
        obs.create_dataset("effort", data=np.full((4, 14), 3.0, dtype=np.float32))

        images = obs.create_group("images")
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            images.create_dataset(
                name,
                data=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            )


def test_teleop_mask_maps_half_open_segments() -> None:
    segments = np.asarray([[1, 3], [4, 5]], dtype=np.int64)
    mask = _teleop_mask(segments, num_frames=6, episode_name="episode_0.hdf5")

    np.testing.assert_array_equal(mask, np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64))


def test_teleop_mask_rejects_invalid_segment() -> None:
    with pytest.raises(ValueError, match="invalid teleop segment"):
        _teleop_mask(
            np.asarray([[2, 7]], dtype=np.int64),
            num_frames=6,
            episode_name="episode_bad.hdf5",
        )


def test_episode_success_array_uses_scalar_reward() -> None:
    np.testing.assert_array_equal(
        _episode_success_array(1.0, num_frames=3),
        np.asarray([True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        _episode_success_array(0.0, num_frames=3),
        np.asarray([False, False, False], dtype=bool),
    )


def test_aloha_features_include_required_recap_columns() -> None:
    features = _aloha_features(
        image_shape=(8, 8, 3),
        include_velocity=True,
        include_effort=True,
    )

    assert features["observation.images.cam_high"]["dtype"] == "image"
    assert features["observation.images.cam_left_wrist"]["dtype"] == "image"
    assert features["observation.images.cam_right_wrist"]["dtype"] == "image"
    assert features["observation.state"]["shape"] == (14,)
    assert features["observation.velocity"]["shape"] == (14,)
    assert features["observation.effort"]["shape"] == (14,)
    assert features["action"]["shape"] == (14,)
    assert features["is_success"]["dtype"] == "bool"
    assert features["teleop_mask"]["dtype"] == "int64"


def test_load_episode_payload_reads_sandwich_hdf5(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_0.hdf5"
    _write_episode(episode_path, reward=1.0)

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")

    assert payload.episode_name == "episode_0.hdf5"
    assert payload.raw_reward == 1.0
    assert payload.num_frames == 4
    assert payload.frames[0]["task"] == "Assemble a sandwich."
    assert payload.frames[0]["observation.state"].shape == (14,)
    assert payload.frames[0]["action"].shape == (14,)
    assert payload.frames[1]["teleop_mask"] == 1
    assert payload.frames[3]["teleop_mask"] == 0
    assert payload.frames[0]["is_success"] is True


def test_build_hil_segments_entry_is_json_serializable(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_0.hdf5"
    _write_episode(episode_path, reward=0.0)

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")
    entry = _build_hil_segments_entry(payload)

    assert entry == {
        "episode_index": 0,
        "episode_name": "episode_0.hdf5",
        "num_frames": 4,
        "raw_reward": 0.0,
        "is_success": False,
        "teleop_segments": [[1, 3]],
    }
    json.dumps(entry)
```

- [ ] **Step 2: Run the new tests and verify the expected import failure**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_converter.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples.offline_rl.data'` or `ImportError` for the new converter functions.

- [ ] **Step 3: Add the converter module package directory**

Run:

```bash
mkdir -p examples/offline_rl/data
```

Create `examples/offline_rl/data/__init__.py` with this content:

```python
"""Offline RL data conversion utilities."""
```

- [ ] **Step 4: Implement testable converter helpers**

Create `examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py` with this initial content:

```python
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

"""Convert ALOHA sandwich HITL HDF5 episodes to LeRobot v2.1."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


ALOHA_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_TASK = "Assemble a sandwich."


@dataclass(frozen=True)
class EpisodePayload:
    """Loaded ALOHA episode ready to write into LeRobot."""

    episode_index: int
    episode_name: str
    raw_reward: float
    teleop_segments: list[list[int]]
    frames: list[dict[str, Any]]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def is_success(self) -> bool:
        return self.raw_reward > 0.0


def _aloha_features(
    image_shape: tuple[int, int, int],
    include_velocity: bool,
    include_effort: bool,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action"],
        },
        "is_success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_success"],
        },
        "teleop_mask": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["teleop_mask"],
        },
    }

    if include_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": ["velocity"],
        }
    if include_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": ["effort"],
        }

    for camera in ALOHA_CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    return features


def _episode_success_array(raw_reward: float, num_frames: int) -> np.ndarray:
    return np.full(num_frames, raw_reward > 0.0, dtype=bool)


def _teleop_mask(
    segments: np.ndarray,
    num_frames: int,
    episode_name: str,
) -> np.ndarray:
    if segments.size == 0:
        return np.zeros(num_frames, dtype=np.int64)
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError(
            f"{episode_name}: teleop_segments must have shape (N, 2), got {segments.shape}"
        )

    mask = np.zeros(num_frames, dtype=np.int64)
    for raw_start, raw_end in segments.astype(np.int64):
        start = int(raw_start)
        end = int(raw_end)
        if start < 0 or end < start or end > num_frames:
            raise ValueError(
                f"{episode_name}: invalid teleop segment [{start}, {end}) "
                f"for {num_frames} frames"
            )
        mask[start:end] = 1
    return mask


def _read_scalar_reward(ep: h5py.File, episode_name: str) -> float:
    if "reward" not in ep:
        raise ValueError(f"{episode_name}: missing scalar reward")
    reward = np.asarray(ep["reward"][()])
    if reward.shape not in ((), (1,)):
        raise ValueError(f"{episode_name}: reward must be scalar or shape (1,), got {reward.shape}")
    return float(reward.reshape(-1)[0])


def _read_teleop_segments(ep: h5py.File) -> np.ndarray:
    if "teleop_segments" not in ep:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(ep["teleop_segments"][:], dtype=np.int64)


def _read_images(ep: h5py.File, episode_name: str) -> dict[str, np.ndarray]:
    images_group = ep.get("observations/images")
    if not isinstance(images_group, h5py.Group):
        raise ValueError(f"{episode_name}: missing observations/images group")

    images: dict[str, np.ndarray] = {}
    for camera in ALOHA_CAMERAS:
        if camera not in images_group:
            available = ", ".join(sorted(images_group.keys()))
            raise ValueError(
                f"{episode_name}: missing camera {camera}; available cameras: {available}"
            )
        camera_data = np.asarray(images_group[camera][:])
        if camera_data.ndim != 4 or camera_data.shape[-1] != 3:
            raise ValueError(
                f"{episode_name}: camera {camera} must have shape (T, H, W, 3), "
                f"got {camera_data.shape}"
            )
        images[camera] = camera_data
    return images


def _load_episode_payload(
    episode_path: Path,
    task: str,
    episode_index: int = 0,
) -> EpisodePayload:
    episode_name = episode_path.name
    with h5py.File(episode_path, "r") as ep:
        qpos = np.asarray(ep["observations/qpos"][:], dtype=np.float32)
        action = np.asarray(ep["action"][:], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 14:
            raise ValueError(f"{episode_name}: observations/qpos must have shape (T, 14)")
        if action.shape != qpos.shape:
            raise ValueError(
                f"{episode_name}: action shape {action.shape} must match qpos shape {qpos.shape}"
            )

        qvel = (
            np.asarray(ep["observations/qvel"][:], dtype=np.float32)
            if "observations/qvel" in ep
            else None
        )
        effort = (
            np.asarray(ep["observations/effort"][:], dtype=np.float32)
            if "observations/effort" in ep
            else None
        )
        raw_reward = _read_scalar_reward(ep, episode_name)
        segments = _read_teleop_segments(ep)
        images = _read_images(ep, episode_name)

    num_frames = qpos.shape[0]
    success = _episode_success_array(raw_reward, num_frames)
    teleop = _teleop_mask(segments, num_frames, episode_name)

    frames: list[dict[str, Any]] = []
    for idx in range(num_frames):
        frame: dict[str, Any] = {
            "observation.state": qpos[idx],
            "action": action[idx],
            "task": task,
            "is_success": bool(success[idx]),
            "teleop_mask": int(teleop[idx]),
        }
        if qvel is not None:
            frame["observation.velocity"] = qvel[idx]
        if effort is not None:
            frame["observation.effort"] = effort[idx]
        for camera, camera_frames in images.items():
            frame[f"observation.images.{camera}"] = camera_frames[idx]
        frames.append(frame)

    return EpisodePayload(
        episode_index=episode_index,
        episode_name=episode_name,
        raw_reward=raw_reward,
        teleop_segments=segments.astype(int).tolist(),
        frames=frames,
    )


def _build_hil_segments_entry(payload: EpisodePayload) -> dict[str, Any]:
    return {
        "episode_index": payload.episode_index,
        "episode_name": payload.episode_name,
        "num_frames": payload.num_frames,
        "raw_reward": payload.raw_reward,
        "is_success": payload.is_success,
        "teleop_segments": payload.teleop_segments,
    }
```

- [ ] **Step 5: Run converter helper tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_converter.py -q
```

Expected: PASS for all tests in `test_aloha_recap_converter.py`.

- [ ] **Step 6: Commit helper coverage and module skeleton**

Run:

```bash
git add examples/offline_rl/data/__init__.py examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py tests/unit_tests/test_aloha_recap_converter.py
git commit -s -m "test: cover aloha recap converter helpers"
```

## Task 2: ALOHA Converter CLI and Dataset Writing

**Files:**
- Modify: `examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py`
- Modify: `tests/unit_tests/test_aloha_recap_converter.py`

- [ ] **Step 1: Add a fake LeRobot writer test for converter output**

Append this test code to `tests/unit_tests/test_aloha_recap_converter.py`:

```python
class _FakeLeRobotDataset:
    created_kwargs: dict | None = None

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.episode_lengths: list[int] = []

    @classmethod
    def create(cls, **kwargs):
        cls.created_kwargs = kwargs
        return cls()

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episode_lengths.append(len(self.frames) - sum(self.episode_lengths))


def test_convert_dataset_writes_frames_and_hil_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.offline_rl.data import convert_aloha_hdf5_to_lerobot_v21 as converter

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_episode(raw_dir / "episode_0.hdf5", reward=1.0)
    _write_episode(raw_dir / "episode_1.hdf5", reward=0.0)
    output_dir = tmp_path / "lerobot"

    monkeypatch.setattr(converter, "LeRobotDataset", _FakeLeRobotDataset)

    dataset = converter.convert_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        repo_id="local/aloha_sandwich",
        task="Assemble a sandwich.",
        fps=25,
        overwrite=True,
        image_writer_threads=1,
        image_writer_processes=1,
    )

    assert dataset.episode_lengths == [4, 4]
    assert _FakeLeRobotDataset.created_kwargs["repo_id"] == "local/aloha_sandwich"
    assert _FakeLeRobotDataset.created_kwargs["root"] == output_dir
    assert _FakeLeRobotDataset.created_kwargs["robot_type"] == "aloha"

    metadata_path = output_dir / "meta" / "hil_segments.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["total_episodes"] == 2
    assert metadata["total_frames"] == 8
    assert metadata["successful_episodes"] == 1
    assert metadata["failed_episodes"] == 1
    assert metadata["episodes"][0]["teleop_segments"] == [[1, 3]]
```

- [ ] **Step 2: Run the converter tests and verify the expected missing CLI failure**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_converter.py -q
```

Expected: FAIL with `AttributeError: module ... has no attribute 'convert_dataset'`.

- [ ] **Step 3: Add LeRobot imports, dataset writing, audit metadata, and CLI**

Modify `examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py` by adding the LeRobot import after the existing imports:

```python
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - exercised only when converter deps are absent
    LeRobotDataset = None
```

Append this code to the end of the file:

```python
def _discover_hdf5_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.hdf5"))
    if not files:
        files = sorted(raw_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 episodes found under {raw_dir}")
    return files


def _first_image_shape(first_episode: Path) -> tuple[int, int, int]:
    with h5py.File(first_episode, "r") as ep:
        images = _read_images(ep, first_episode.name)
    first = images[ALOHA_CAMERAS[0]]
    return tuple(int(v) for v in first.shape[1:])


def _has_dataset(first_episode: Path, key: str) -> bool:
    with h5py.File(first_episode, "r") as ep:
        return key in ep


def _write_hil_segments(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    total_frames = sum(int(entry["num_frames"]) for entry in entries)
    successful = sum(1 for entry in entries if bool(entry["is_success"]))
    data = {
        "total_episodes": len(entries),
        "total_frames": total_frames,
        "successful_episodes": successful,
        "failed_episodes": len(entries) - successful,
        "episodes": entries,
    }
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "hil_segments.json", "w") as f:
        json.dump(data, f, indent=2)


def convert_dataset(
    raw_dir: Path,
    output_dir: Path,
    repo_id: str,
    task: str = DEFAULT_TASK,
    fps: int = 25,
    overwrite: bool = False,
    image_writer_threads: int = 5,
    image_writer_processes: int = 10,
):
    if LeRobotDataset is None:
        raise ImportError("lerobot is required to run ALOHA HDF5 conversion")

    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    hdf5_files = _discover_hdf5_files(raw_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    features = _aloha_features(
        image_shape=_first_image_shape(hdf5_files[0]),
        include_velocity=_has_dataset(hdf5_files[0], "observations/qvel"),
        include_effort=_has_dataset(hdf5_files[0], "observations/effort"),
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="aloha",
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    metadata_entries: list[dict[str, Any]] = []
    for episode_index, episode_path in enumerate(tqdm(hdf5_files, desc="Converting ALOHA episodes")):
        payload = _load_episode_payload(
            episode_path,
            task=task,
            episode_index=episode_index,
        )
        for frame in payload.frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        metadata_entries.append(_build_hil_segments_entry(payload))

    _write_hil_segments(output_dir, metadata_entries)
    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="local/aloha_sandwich")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=5)
    parser.add_argument("--image-writer-processes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run converter tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_converter.py -q
```

Expected: PASS.

- [ ] **Step 5: Run converter help**

Run:

```bash
python examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py --help
```

Expected: command exits 0 and prints arguments including `--raw-dir`, `--output-dir`, `--repo-id`, and `--overwrite`.

- [ ] **Step 6: Commit converter CLI**

Run:

```bash
git add examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py tests/unit_tests/test_aloha_recap_converter.py
git commit -s -m "feat: add aloha sandwich lerobot converter"
```

## Task 3: HITL-Aware Return Computation

**Files:**
- Create: `tests/unit_tests/test_aloha_recap_returns.py`
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`

- [ ] **Step 1: Write failing return split tests**

Create `tests/unit_tests/test_aloha_recap_returns.py` with this content:

```python
import numpy as np

from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    compute_hitl_aware_returns_for_episode,
    compute_returns_for_episode,
)


def test_successful_episode_split_at_first_teleop_frame() -> None:
    returns, rewards = compute_hitl_aware_returns_for_episode(
        episode_length=6,
        is_success=True,
        teleop_mask=np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(
        rewards,
        np.asarray([-1.0, -1.0, -300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        returns,
        np.asarray([-302.0, -301.0, -300.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )


def test_successful_episode_without_teleop_uses_standard_returns() -> None:
    hitl_returns, hitl_rewards = compute_hitl_aware_returns_for_episode(
        episode_length=4,
        is_success=True,
        teleop_mask=np.zeros(4, dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )
    normal_returns, normal_rewards = compute_returns_for_episode(
        episode_length=4,
        is_success=True,
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(hitl_returns, normal_returns)
    np.testing.assert_array_equal(hitl_rewards, normal_rewards)


def test_failed_episode_with_teleop_remains_failed() -> None:
    hitl_returns, hitl_rewards = compute_hitl_aware_returns_for_episode(
        episode_length=4,
        is_success=False,
        teleop_mask=np.asarray([0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )
    normal_returns, normal_rewards = compute_returns_for_episode(
        episode_length=4,
        is_success=False,
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(hitl_returns, normal_returns)
    np.testing.assert_array_equal(hitl_rewards, normal_rewards)
```

- [ ] **Step 2: Run the tests and verify the expected missing function failure**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_returns.py -q
```

Expected: FAIL with `ImportError` for `compute_hitl_aware_returns_for_episode`.

- [ ] **Step 3: Add HITL-aware helper and optional column reading**

In `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`, replace:

```python
_READ_COLUMNS = ["episode_index", "frame_index", "is_success", "task_index", "task"]
```

with:

```python
_READ_COLUMNS = [
    "episode_index",
    "frame_index",
    "is_success",
    "teleop_mask",
    "task_index",
    "task",
]
```

Add this function immediately after `compute_returns_for_episode`:

```python
def compute_hitl_aware_returns_for_episode(
    episode_length: int,
    is_success: bool,
    teleop_mask: np.ndarray,
    gamma: float,
    failure_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute returns that mark pre-intervention prefixes as failed.

    Successful HITL episodes are split at the first teleop frame. The
    autonomous prefix receives failed-episode returns and the suffix receives
    successful returns. Failed episodes remain failed even when they contain
    teleop frames.
    """
    if not is_success:
        return compute_returns_for_episode(
            episode_length=episode_length,
            is_success=False,
            gamma=gamma,
            failure_reward=failure_reward,
        )

    mask = np.asarray(teleop_mask).astype(bool)
    if mask.shape[0] != episode_length:
        raise ValueError(
            f"teleop_mask length {mask.shape[0]} does not match episode length {episode_length}"
        )
    teleop_indices = np.flatnonzero(mask)
    if len(teleop_indices) == 0:
        return compute_returns_for_episode(
            episode_length=episode_length,
            is_success=True,
            gamma=gamma,
            failure_reward=failure_reward,
        )

    split = int(teleop_indices[0])
    if split == 0:
        return compute_returns_for_episode(
            episode_length=episode_length,
            is_success=True,
            gamma=gamma,
            failure_reward=failure_reward,
        )

    prefix_returns, prefix_rewards = compute_returns_for_episode(
        episode_length=split,
        is_success=False,
        gamma=gamma,
        failure_reward=failure_reward,
    )
    suffix_returns, suffix_rewards = compute_returns_for_episode(
        episode_length=episode_length - split,
        is_success=True,
        gamma=gamma,
        failure_reward=failure_reward,
    )
    return (
        np.concatenate([prefix_returns, suffix_returns]).astype(np.float32),
        np.concatenate([prefix_rewards, suffix_rewards]).astype(np.float32),
    )
```

- [ ] **Step 4: Thread `hitl_aware_returns` through parquet processing**

Change `_process_single_parquet` signature to:

```python
def _process_single_parquet(
    pq_file: str,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    tasks: dict[int, str],
    hitl_aware_returns: bool = False,
) -> pa.Table | None:
```

After `is_success_col` is resolved, add:

```python
    teleop_col = None
    if hitl_aware_returns and "teleop_mask" in col_names:
        teleop_col = np.asarray(table.column("teleop_mask").to_pylist(), dtype=np.int64)
    elif hitl_aware_returns:
        logger.warning(
            "data.hitl_aware_returns is enabled but teleop_mask is missing in %s; "
            "falling back to standard returns for this file.",
            pq_file,
        )
```

Replace the `ep_returns, ep_rewards = compute_returns_for_episode(...)` block inside the episode loop with:

```python
        if hitl_aware_returns and teleop_col is not None:
            ep_returns, ep_rewards = compute_hitl_aware_returns_for_episode(
                episode_length=ep_length,
                is_success=is_success,
                teleop_mask=teleop_col[ep_start:ep_end],
                gamma=gamma,
                failure_reward=failure_reward,
            )
        else:
            ep_returns, ep_rewards = compute_returns_for_episode(
                episode_length=ep_length,
                is_success=is_success,
                gamma=gamma,
                failure_reward=failure_reward,
            )
```

- [ ] **Step 5: Thread the Hydra config option through `process_dataset` and callers**

Add `hitl_aware_returns: bool = False` to `process_dataset(...)`.

In the serial call to `_process_single_parquet`, pass:

```python
                hitl_aware_returns,
```

In the `pool.submit(...)` call, pass:

```python
                    hitl_aware_returns,
```

In `compute_returns`, read:

```python
    hitl_aware_returns = cfg.data.get("hitl_aware_returns", False)
```

Add `"hitl_aware_returns": entry.get("hitl_aware_returns", hitl_aware_returns),` to each dataset config dictionary. Add `"hitl_aware_returns": hitl_aware_returns,` to the single-dataset config dictionary. Pass `hitl_aware_returns=ds_config["hitl_aware_returns"]` into `process_dataset(...)`.

- [ ] **Step 6: Add config default**

In `examples/offline_rl/config/recap_compute_returns.yaml`, add this under `data:`:

```yaml
  hitl_aware_returns: false
```

- [ ] **Step 7: Run return tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_returns.py -q
```

Expected: PASS.

- [ ] **Step 8: Run import smoke test for compute_returns**

Run:

```bash
python - <<'PY'
from examples.offline_rl.advantage_labeling.recap.process.compute_returns import compute_hitl_aware_returns_for_episode
print(compute_hitl_aware_returns_for_episode.__name__)
PY
```

Expected output:

```text
compute_hitl_aware_returns_for_episode
```

- [ ] **Step 9: Commit HITL-aware returns**

Run:

```bash
git add examples/offline_rl/advantage_labeling/recap/process/compute_returns.py examples/offline_rl/config/recap_compute_returns.yaml tests/unit_tests/test_aloha_recap_returns.py
git commit -s -m "feat: add hitl-aware recap returns"
```

## Task 4: ALOHA Value Dataset and Checkpoint Transforms

**Files:**
- Create: `tests/unit_tests/test_aloha_recap_transforms.py`
- Modify: `rlinf/data/datasets/recap/value_dataset.py`
- Modify: `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py`

- [ ] **Step 1: Write failing transform tests**

Create `tests/unit_tests/test_aloha_recap_transforms.py` with this content:

```python
import numpy as np
import pandas as pd
import pytest


def test_value_dataset_builds_aloha_transform() -> None:
    pytest.importorskip("openpi")

    from rlinf.data.datasets.recap.value_dataset import ValueDataset

    transform = ValueDataset._build_transform(
        robot_type="aloha",
        model_type="pi05",
        action_dim=14,
        default_prompt="Assemble a sandwich.",
    )
    sample = {
        "observation.images.cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.ones((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
        "observation.state": np.zeros((14,), dtype=np.float32),
        "action": np.zeros((10, 14), dtype=np.float32),
        "prompt": "Assemble a sandwich.",
    }

    transformed = transform(sample)

    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape[-1] == 14
    assert transformed["prompt"] == "Assemble a sandwich."


def test_checkpoint_utils_builds_aloha_input_transforms() -> None:
    pytest.importorskip("openpi")

    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        build_input_transforms,
    )

    transforms = build_input_transforms(
        env_type="aloha",
        model_type="pi05",
        action_dim=14,
        default_prompt="Assemble a sandwich.",
        norm_stats=None,
        use_quantile_norm=True,
    )

    names = [type(transform).__name__ for transform in transforms]
    assert names == ["InjectDefaultPrompt", "AlohaInputs", "PadStatesAndActions"]
```

- [ ] **Step 2: Run transform tests and verify the expected unsupported robot failures**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_transforms.py::test_value_dataset_builds_aloha_transform tests/unit_tests/test_aloha_recap_transforms.py::test_checkpoint_utils_builds_aloha_input_transforms -q
```

Expected: FAIL with `Unknown robot type: aloha` or `Unknown environment type: aloha`.

- [ ] **Step 3: Add ALOHA to value dataset transforms**

In `rlinf/data/datasets/recap/value_dataset.py`, replace:

```python
from rlinf.models.embodiment.openpi.policies import franka_policy, libero_policy
```

with:

```python
from rlinf.models.embodiment.openpi.policies import (
    aloha_policy,
    franka_policy,
    libero_policy,
)
```

Add this entry to `_REPACK_KEYS`:

```python
    "aloha": {
        "images": {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        },
        "state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    },
```

In `_build_transform`, after the `libero` branch and before the `franka` branch, add:

```python
        elif robot == "aloha":
            transforms_list.append(
                aloha_policy.AlohaInputs(adapt_to_pi=True)
            )
```

- [ ] **Step 4: Add ALOHA to value checkpoint inference transforms**

In `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py`, replace:

```python
    from rlinf.models.embodiment.openpi.policies import franka_policy, libero_policy
```

with:

```python
    from rlinf.models.embodiment.openpi.policies import (
        aloha_policy,
        franka_policy,
        libero_policy,
    )
```

In `build_input_transforms`, add this branch after the LIBERO branch:

```python
    elif env_type == "aloha":
        input_transforms.append(_openpi_transforms.InjectDefaultPrompt(default_prompt))
        input_transforms.append(aloha_policy.AlohaInputs(adapt_to_pi=True))
```

- [ ] **Step 5: Run transform tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_transforms.py::test_value_dataset_builds_aloha_transform tests/unit_tests/test_aloha_recap_transforms.py::test_checkpoint_utils_builds_aloha_input_transforms -q
```

Expected: PASS or SKIP only when `openpi` is unavailable in the active environment.

- [ ] **Step 6: Commit ALOHA transform support**

Run:

```bash
git add rlinf/data/datasets/recap/value_dataset.py rlinf/models/embodiment/value_model/recap/checkpoint_utils.py tests/unit_tests/test_aloha_recap_transforms.py
git commit -s -m "feat: support aloha recap value transforms"
```

## Task 5: ALOHA Advantage Observations and Teleop Label Override

**Files:**
- Modify: `tests/unit_tests/test_aloha_recap_transforms.py`
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`

- [ ] **Step 1: Add failing tests for ALOHA `build_obs` and teleop override**

Append this code to `tests/unit_tests/test_aloha_recap_transforms.py`:

```python
def test_compute_advantages_build_obs_supports_aloha() -> None:
    from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
        build_obs,
    )

    sample = {
        "observation.images.cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.ones((3, 8, 8), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
        "observation.state": np.arange(14, dtype=np.float32),
        "task_index": np.asarray(0),
    }

    obs = build_obs(sample, robot_type="aloha", tasks={0: "Assemble a sandwich."})

    assert set(obs["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert obs["images"]["cam_high"].shape == (3, 8, 8)
    np.testing.assert_array_equal(obs["state"], np.arange(14, dtype=np.float32))
    assert obs["prompt"] == "Assemble a sandwich."


def test_save_advantages_teleop_override_preserves_continuous_values(tmp_path) -> None:
    from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
        save_advantages_to_dataset,
    )

    dataset_path = tmp_path / "dataset"
    (dataset_path / "meta").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "advantage": [-0.5, 0.2, -0.4],
            "teleop_mask": [0, 1, 1],
        }
    )

    save_advantages_to_dataset(
        dataset_path=dataset_path,
        advantages_df=df,
        threshold=0.0,
        dataset_type="rollout",
        tag="teleop",
    )

    saved = pd.read_parquet(dataset_path / "meta" / "advantages_teleop.parquet")
    assert saved["advantage_continuous"].tolist() == [-0.5, 0.2, -0.4]
    assert saved["advantage"].tolist() == [False, True, True]
    assert saved["teleop_positive"].tolist() == [False, False, True]
```

- [ ] **Step 2: Run the new tests and verify expected failures**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_transforms.py::test_compute_advantages_build_obs_supports_aloha tests/unit_tests/test_aloha_recap_transforms.py::test_save_advantages_teleop_override_preserves_continuous_values -q
```

Expected: FAIL with `Unknown robot_type 'aloha'` and missing `teleop_positive`.

- [ ] **Step 3: Add prompt resolution and ALOHA observation construction**

In `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`, add this helper immediately before `build_obs`:

```python
def _resolve_prompt(sample: dict, tasks: dict) -> str:
    if "prompt" in sample:
        return str(to_scalar(sample["prompt"]))
    if "task" in sample:
        return str(to_scalar(sample["task"]))
    if "task_index" in sample and tasks:
        task_idx = int(to_scalar(sample["task_index"]))
        if task_idx not in tasks:
            raise ValueError(
                f"task_index {task_idx} not found in tasks dict. "
                f"Available task indices: {list(tasks.keys())}."
            )
        return tasks[task_idx]
    raise ValueError(
        "Sample has no prompt, task, or task_index field. "
        "Cannot determine task prompt for value model inference."
    )
```

At the start of `build_obs`, before `if robot_type not in KEY_MAPPINGS`, add:

```python
    if robot_type == "aloha":
        required = (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
            "observation.state",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"ALOHA sample missing required keys: {missing}")
        return {
            "images": {
                "cam_high": to_numpy(sample["observation.images.cam_high"]),
                "cam_left_wrist": to_numpy(sample["observation.images.cam_left_wrist"]),
                "cam_right_wrist": to_numpy(sample["observation.images.cam_right_wrist"]),
            },
            "state": to_numpy(sample["observation.state"]),
            "prompt": _resolve_prompt(sample, tasks),
        }
```

In the generic `task` branch inside `build_obs`, replace the current prompt resolution block with:

```python
            obs[dst_key] = _resolve_prompt(sample, tasks)
```

- [ ] **Step 4: Carry `teleop_mask` through advantage inference metadata**

In `ValueInferenceDataset.__getitem__`, after reward is resolved, add:

```python
        teleop_mask = 0
        if "teleop_mask" in sample:
            teleop_mask = int(to_scalar(sample["teleop_mask"]))
```

Add `"teleop_mask": teleop_mask,` to the returned dictionary.

In `advantage_collate_fn`, add this field to each metadata dictionary:

```python
            "teleop_mask": item.get("teleop_mask", 0),
```

In `compute_advantages_for_dataset`, add this key to `results`:

```python
        "teleop_mask": [],
```

Add a `meta_teleop_mask` array next to `meta_reward`:

```python
    meta_teleop_mask = np.zeros(extended_size, dtype=np.int64)
```

Inside `process_value_batch`, set:

```python
            meta_teleop_mask[local_idx] = int(meta_info.get("teleop_mask", 0))
```

Inside the Phase 2 loop, append:

```python
        results["teleop_mask"].append(int(meta_teleop_mask[i]))
```

- [ ] **Step 5: Override only boolean labels in `save_advantages_to_dataset`**

In `save_advantages_to_dataset`, replace the non-SFT boolean label block with:

```python
        if (dataset_type or "").lower() == "sft":
            save_df["advantage"] = True
            save_df["teleop_positive"] = False
        else:
            value_positive = apply_boolean_label(
                save_df["advantage_continuous"], threshold, inclusive=True
            )
            teleop_mask = (
                save_df["teleop_mask"].astype(bool)
                if "teleop_mask" in save_df.columns
                else pd.Series(False, index=save_df.index)
            )
            save_df["teleop_positive"] = teleop_mask & ~value_positive
            save_df["advantage"] = value_positive | teleop_mask
            if teleop_mask.any():
                overridden = int(save_df["teleop_positive"].sum())
                teleop_total = int(teleop_mask.sum())
                logger.info(
                    "  Teleop positive override: %d/%d teleop frames forced positive (%.2f%% of dataset)",
                    overridden,
                    teleop_total,
                    100.0 * overridden / max(len(save_df), 1),
                )
```

- [ ] **Step 6: Run ALOHA advantage tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_transforms.py::test_compute_advantages_build_obs_supports_aloha tests/unit_tests/test_aloha_recap_transforms.py::test_save_advantages_teleop_override_preserves_continuous_values -q
```

Expected: PASS.

- [ ] **Step 7: Run all ALOHA RECAP unit tests**

Run:

```bash
python -m pytest tests/unit_tests/test_aloha_recap_converter.py tests/unit_tests/test_aloha_recap_returns.py tests/unit_tests/test_aloha_recap_transforms.py -q
```

Expected: PASS, with OpenPI transform tests skipped only if `openpi` is unavailable.

- [ ] **Step 8: Commit ALOHA advantage support**

Run:

```bash
git add examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py tests/unit_tests/test_aloha_recap_transforms.py
git commit -s -m "feat: add aloha recap advantage labeling"
```

## Task 6: Sandwich RECAP Configs

**Files:**
- Create: `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`
- Create: `examples/offline_rl/config/aloha_sandwich_recap_value_model_sft.yaml`
- Create: `examples/offline_rl/config/aloha_sandwich_recap_compute_advantages.yaml`
- Create: `examples/offline_rl/config/aloha_sandwich_cfg_rl_openpi.yaml`

- [ ] **Step 1: Add Step 1 returns config**

Create `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`:

```yaml
# Usage:
#   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh aloha_sandwich_recap_compute_returns

data:
  data_root: null
  train_data_paths:
    - dataset_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21
      type: rollout

  dataset_type: rollout
  gamma: 1.0
  failure_reward: -300.0
  hitl_aware_returns: true
  tag: sandwich_fail300
  num_workers: 128

hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false
```

- [ ] **Step 2: Add Step 2 value SFT config**

Create `examples/offline_rl/config/aloha_sandwich_recap_value_model_sft.yaml`:

```yaml
# Usage:
#   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh aloha_sandwich_recap_value_model_sft

defaults:
  - model/recap_value_model@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - _self_
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:REPO_PATH}/examples/offline_rl/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: sft
  logger:
    log_path: "../results/aloha_sandwich_value"
    project_name: rlinf
    experiment_name: "aloha_sandwich_value"
    logger_backends: ["tensorboard"]

  max_epochs: 30000
  max_steps: -1
  val_check_interval: 500
  save_interval: 3000

data:
  tag: sandwich_fail300

  train_data_paths:
    - dataset_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21
      type: rollout
      weight: 1.0
      robot_type: aloha
      model_type: pi05

  eval_data_paths:
    - dataset_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21
      max_samples: 4096
      robot_type: aloha
      model_type: pi05

  train_num_workers: 8
  eval_num_workers: 4
  prefetch_factor: 4
  persistent_workers: false
  pin_memory: false
  include_state: true
  gamma: 1.0
  action_horizon: 10
  normalize_to_minus_one_zero: true
  include_next_obs: false
  action_dim: 14
  robot_type: aloha
  model_type: pi05
  balance_weights: true
  seed: 42

algorithm:
  adv_type: gae

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 256
  seed: 0

  model:
    precision: bf16
    siglip_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/siglip2-so400m-patch14-224
    gemma3_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m
    tokenizer_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m
    freeze_vlm: false
    action_dim: 14
    action_horizon: 10

  optim:
    lr: 5.0e-5
    value_lr: 1.0e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-8
    weight_decay: 1e-10
    clip_grad: 1.0
    lr_scheduler: "constant"
    lr_warmup_steps: 500
    total_training_steps: 8000

  fsdp_config:
    strategy: "fsdp"
    sharding_strategy: "no_shard"
    use_orig_params: false
    gradient_checkpointing: false
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: false

critic:
  use_critic_model: false
```

- [ ] **Step 3: Add Step 3 advantage config**

Create `examples/offline_rl/config/aloha_sandwich_recap_compute_advantages.yaml`:

```yaml
# Usage:
#   ALOHA_SANDWICH_VALUE_CHECKPOINT=/abs/path/to/value/checkpoint \
#   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh aloha_sandwich_recap_compute_advantages --nproc 1

defaults:
  - model/recap_value_model@advantage.model

advantage:
  value_checkpoint: ${oc.env:ALOHA_SANDWICH_VALUE_CHECKPOINT}
  batch_size: 1024
  flush_interval: 256
  num_dataloader_workers_per_gpu: 12
  prefetch_factor: 2
  discount_next_value: true
  positive_quantile: 0.3
  tag: sandwich_fail300_N10_q30_teleop
  returns_tag: sandwich_fail300

  model:
    critic_expert_variant: "gemma_1m"
    tokenizer_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m
    siglip_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/siglip2-so400m-patch14-224
    gemma3_path: /inspire/hdd/project/robot-reasoning/public/shared/hf-models/google/gemma-3-270m

data:
  model_type: pi05

  train_data_paths:
    - dataset_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21
      robot_type: aloha
      type: rollout
      weight: 1.0

  advantage_lookahead_step: 10
  gamma: 1.0

distributed:
  enabled: true
  backend: "nccl"
  timeout: 3600

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:REPO_PATH}/examples/offline_rl/config/
  job:
    chdir: false
```

- [ ] **Step 4: Add Step 4 CFG config**

Create `examples/offline_rl/config/aloha_sandwich_cfg_rl_openpi.yaml`:

```yaml
# Usage:
#   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh aloha_sandwich_cfg_rl_openpi

defaults:
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - _self_
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:REPO_PATH}/examples/offline_rl/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: sft
  logger:
    log_path: "../results/aloha_sandwich_cfg"
    project_name: rlinf
    experiment_name: "aloha_sandwich_cfg"
    logger_backends: ["tensorboard"]

  max_epochs: 30000
  max_steps: -1
  val_check_interval: -1
  save_interval: 3000

data:
  num_workers: 8
  advantage_tag: sandwich_fail300_N10_q30_teleop
  balance_dataset_weights: true
  seed: 42

  train_data_paths:
    - dataset_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21
      type: rollout
      weight: 1.0

algorithm:
  adv_type: gae

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512
  seed: 0

  model:
    precision: null
    model_path: /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all_pytorch/49999
    model_type: cfg_model
    add_value_head: false
    action_dim: 14
    num_action_chunks: 10
    openpi:
      config_name: pi05_aloha_robotwin
      num_images_in_input: 3
      action_env_dim: 14
      cfgrl_guidance_scale: 1.0
      unconditional_prob: 0.1
      train_expert_only: false
      guidance_type: positive
      positive_only_conditional: true

  optim:
    lr: 1.0e-5
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-8
    weight_decay: 1.0e-10
    clip_grad: 1.0
    lr_scheduler: "cosine"
    lr_warmup_steps: 5000
    total_training_steps: 30000
    min_lr: 0.0

  fsdp_config:
    strategy: "fsdp"
    sharding_strategy: "no_shard"
    use_orig_params: false
    gradient_checkpointing: true
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: false

critic:
  use_critic_model: false
```

- [ ] **Step 5: Validate Hydra config composition without launching training**

Run:

```bash
REPO_PATH="$(pwd)" python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir

config_dir = str(Path("examples/offline_rl/config").resolve())
configs = [
    "aloha_sandwich_recap_compute_returns",
    "aloha_sandwich_recap_value_model_sft",
    "aloha_sandwich_recap_compute_advantages",
    "aloha_sandwich_cfg_rl_openpi",
]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    for name in configs:
        cfg = compose(config_name=name)
        print(name, "ok")
PY
```

Expected output:

```text
aloha_sandwich_recap_compute_returns ok
aloha_sandwich_recap_value_model_sft ok
aloha_sandwich_recap_compute_advantages ok
aloha_sandwich_cfg_rl_openpi ok
```

- [ ] **Step 6: Commit sandwich configs**

Run:

```bash
git add examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml examples/offline_rl/config/aloha_sandwich_recap_value_model_sft.yaml examples/offline_rl/config/aloha_sandwich_recap_compute_advantages.yaml examples/offline_rl/config/aloha_sandwich_cfg_rl_openpi.yaml
git commit -s -m "feat: add aloha sandwich recap configs"
```

## Task 7: Documentation

**Files:**
- Modify: `docs/source-en/rst_source/examples/embodied/recap.rst`
- Modify: `docs/source-zh/rst_source/examples/embodied/recap.rst`

- [ ] **Step 1: Add English ALOHA sandwich section**

In `docs/source-en/rst_source/examples/embodied/recap.rst`, insert this section after the "Pipeline Tag System" table:

```rst
ALOHA Sandwich HITL Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf also supports running RECAP on an ALOHA sandwich HITL rollout dataset.
This path keeps the data as ``robot_type: aloha`` and uses the existing
``pi05_aloha_robotwin`` OpenPI configuration.

The expected raw data and checkpoint locations are:

.. code:: text

   /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl
   /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999

Convert the raw HDF5 episodes to LeRobot v2.1:

.. code:: bash

   python examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py \
      --raw-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl \
      --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21 \
      --repo-id local/aloha_sandwich \
      --task "Assemble a sandwich." \
      --overwrite

The converter writes three ALOHA cameras, 14-D state/action, per-frame
``is_success``, per-frame ``teleop_mask``, and ``meta/hil_segments.json`` for
auditability. Use ``type: rollout`` because the dataset contains both successes
and failures.

Convert the OpenPI JAX checkpoint to the PyTorch checkpoint expected by CFG
training:

.. code:: bash

   python /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/examples/convert_jax_model_to_pytorch.py \
      --input-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999 \
      --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all_pytorch/49999

Run the four RECAP stages:

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
      aloha_sandwich_recap_compute_returns

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
      aloha_sandwich_recap_value_model_sft

   export ALOHA_SANDWICH_VALUE_CHECKPOINT=/absolute/path/to/value/checkpoint
   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
      aloha_sandwich_recap_compute_advantages --nproc 1

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
      aloha_sandwich_cfg_rl_openpi

When ``data.hitl_aware_returns`` is enabled, successful episodes with human
intervention are split at the first teleop frame: the autonomous prefix receives
failed returns, and the intervention suffix receives successful returns. During
advantage labeling, continuous advantages remain value-model based, while
teleop frames force the boolean ``advantage`` label to positive and record the
override in ``teleop_positive``.
```

- [ ] **Step 2: Add Chinese ALOHA sandwich section**

In `docs/source-zh/rst_source/examples/embodied/recap.rst`, insert this matching section after the "Pipeline Tag" table:

```rst
ALOHA Sandwich HITL 示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 支持在 ALOHA sandwich HITL rollout 数据集上运行 RECAP。该路径保持
``robot_type: aloha``，并复用已有的 ``pi05_aloha_robotwin`` OpenPI 配置。

原始数据和 SFT 检查点位置如下：

.. code:: text

   /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl
   /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999

先将 HDF5 episode 转换为 LeRobot v2.1：

.. code:: bash

   python examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py \
      --raw-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl \
      --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21 \
      --repo-id local/aloha_sandwich \
      --task "Assemble a sandwich." \
      --overwrite

转换器会写入三个 ALOHA 相机、14 维 state/action、逐帧 ``is_success``、
逐帧 ``teleop_mask``，以及用于审计的 ``meta/hil_segments.json``。由于数据
同时包含成功和失败轨迹，应使用 ``type: rollout``。

再将 OpenPI JAX 检查点转换为 CFG 训练需要的 PyTorch 检查点：

.. code:: bash

   python /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/examples/convert_jax_model_to_pytorch.py \
      --input-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999 \
      --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all_pytorch/49999

依次运行四个 RECAP 阶段：

.. code:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
      aloha_sandwich_recap_compute_returns

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
      aloha_sandwich_recap_value_model_sft

   export ALOHA_SANDWICH_VALUE_CHECKPOINT=/absolute/path/to/value/checkpoint
   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
      aloha_sandwich_recap_compute_advantages --nproc 1

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
      aloha_sandwich_cfg_rl_openpi

启用 ``data.hitl_aware_returns`` 后，带有人类介入的成功 episode 会在第一个
teleop 帧切分：自主执行前缀使用失败回报，介入后的后缀使用成功回报。优势
标注阶段不会修改连续优势值；teleop 帧只会强制布尔 ``advantage`` 标签为正，
并在 ``teleop_positive`` 中记录该覆盖。
```

- [ ] **Step 3: Check docs syntax around the new sections**

Run:

```bash
python - <<'PY'
from pathlib import Path
for path in [
    Path("docs/source-en/rst_source/examples/embodied/recap.rst"),
    Path("docs/source-zh/rst_source/examples/embodied/recap.rst"),
]:
    text = path.read_text()
    assert "ALOHA Sandwich" in text
    assert "teleop_positive" in text
    assert "aloha_sandwich_cfg_rl_openpi" in text
    print(path, "ok")
PY
```

Expected output:

```text
docs/source-en/rst_source/examples/embodied/recap.rst ok
docs/source-zh/rst_source/examples/embodied/recap.rst ok
```

- [ ] **Step 4: Commit docs**

Run:

```bash
git add docs/source-en/rst_source/examples/embodied/recap.rst docs/source-zh/rst_source/examples/embodied/recap.rst
git commit -s -m "docs: add aloha sandwich recap guide"
```

## Task 8: Local Validation and Smoke Commands

**Files:**
- No source edits unless a previous task exposes a failure.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
python -m pytest \
  tests/unit_tests/test_aloha_recap_converter.py \
  tests/unit_tests/test_aloha_recap_returns.py \
  tests/unit_tests/test_aloha_recap_transforms.py \
  -q
```

Expected: PASS, with OpenPI transform tests skipped only if `openpi` is unavailable.

- [ ] **Step 2: Run import check for the touched RECAP modules**

Run:

```bash
python - <<'PY'
import examples.offline_rl.advantage_labeling.recap.process.compute_advantages as compute_advantages
import examples.offline_rl.advantage_labeling.recap.process.compute_returns as compute_returns
import rlinf.data.datasets.recap.value_dataset as value_dataset
import rlinf.models.embodiment.value_model.recap.checkpoint_utils as checkpoint_utils

print(compute_returns.compute_hitl_aware_returns_for_episode.__name__)
print(compute_advantages.build_obs.__name__)
print("aloha" in value_dataset._REPACK_KEYS)
print(callable(checkpoint_utils.build_input_transforms))
PY
```

Expected output:

```text
compute_hitl_aware_returns_for_episode
build_obs
True
True
```

- [ ] **Step 3: Convert the full sandwich dataset**

Run:

```bash
python examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py \
  --raw-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl \
  --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21 \
  --repo-id local/aloha_sandwich \
  --task "Assemble a sandwich." \
  --overwrite
```

Expected: conversion completes, writes 26 episodes, and writes `/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/hil_segments.json`.

- [ ] **Step 4: Validate converted dataset counts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

meta_path = Path("/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/hil_segments.json")
data = json.loads(meta_path.read_text())
assert data["total_episodes"] == 26, data["total_episodes"]
assert data["total_frames"] == 60162, data["total_frames"]
assert data["successful_episodes"] == 17, data["successful_episodes"]
assert data["failed_episodes"] == 9, data["failed_episodes"]
print("aloha sandwich metadata ok")
PY
```

Expected output:

```text
aloha sandwich metadata ok
```

- [ ] **Step 5: Run Step 1 returns smoke**

Run:

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  aloha_sandwich_recap_compute_returns
```

Expected: writes `/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/returns_sandwich_fail300.parquet` and updates `meta/stats.json`.

- [ ] **Step 6: Validate Step 1 output**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/returns_sandwich_fail300.parquet")
df = pd.read_parquet(path)
assert len(df) == 60162, len(df)
assert {"episode_index", "frame_index", "return", "reward", "prompt"} <= set(df.columns)
print("returns rows", len(df))
PY
```

Expected output starts with:

```text
returns rows 60162
```

- [ ] **Step 7: Convert the pi0.5 SFT checkpoint before CFG training**

Run:

```bash
python /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/examples/convert_jax_model_to_pytorch.py \
  --input-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999 \
  --output-dir /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all_pytorch/49999
```

Expected: output directory contains a PyTorch weight file accepted by RLinf and an `assets/` directory with sandwich normalization stats.

- [ ] **Step 8: Run a Step 3 smoke after a value checkpoint exists**

Run after Step 2 has created a checkpoint:

```bash
export ALOHA_SANDWICH_VALUE_CHECKPOINT=/absolute/path/to/value/checkpoint
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  aloha_sandwich_recap_compute_advantages \
  --nproc 1 \
  advantage.max_samples=128 \
  advantage.num_dataloader_workers_per_gpu=0
```

Expected: writes `meta/advantages_sandwich_fail300_N10_q30_teleop.parquet` with columns `advantage_continuous`, `advantage`, `teleop_mask`, and `teleop_positive`.

- [ ] **Step 9: Validate Step 3 teleop override output**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/advantages_sandwich_fail300_N10_q30_teleop.parquet")
df = pd.read_parquet(path)
assert {"advantage_continuous", "advantage", "teleop_mask", "teleop_positive"} <= set(df.columns)
assert (df.loc[df["teleop_mask"].astype(bool), "advantage"] == True).all()
print("advantages rows", len(df))
PY
```

Expected output starts with:

```text
advantages rows
```

- [ ] **Step 10: Run CFG dataloader/model-init smoke**

Run after checkpoint conversion and Step 3 smoke:

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  aloha_sandwich_cfg_rl_openpi \
  runner.max_steps=1 \
  runner.save_interval=-1 \
  actor.global_batch_size=4 \
  actor.micro_batch_size=2 \
  data.num_workers=0
```

Expected: training initializes the CFG dataset and model, runs one step, and exits without attempting a long training run.

- [ ] **Step 11: Run lint on touched Python files**

Run:

```bash
python -m ruff check \
  examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py \
  examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py \
  rlinf/data/datasets/recap/value_dataset.py \
  rlinf/models/embodiment/value_model/recap/checkpoint_utils.py \
  tests/unit_tests/test_aloha_recap_converter.py \
  tests/unit_tests/test_aloha_recap_returns.py \
  tests/unit_tests/test_aloha_recap_transforms.py
```

Expected: PASS.

- [ ] **Step 12: Run formatter check on touched Python files**

Run:

```bash
python -m ruff format --check \
  examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py \
  examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py \
  rlinf/data/datasets/recap/value_dataset.py \
  rlinf/models/embodiment/value_model/recap/checkpoint_utils.py \
  tests/unit_tests/test_aloha_recap_converter.py \
  tests/unit_tests/test_aloha_recap_returns.py \
  tests/unit_tests/test_aloha_recap_transforms.py
```

Expected: PASS.

- [ ] **Step 13: Commit final verification fixes if any source changed**

Run only if Steps 1-12 required additional source edits:

```bash
git add examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py examples/offline_rl/advantage_labeling/recap/process/compute_returns.py examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py rlinf/data/datasets/recap/value_dataset.py rlinf/models/embodiment/value_model/recap/checkpoint_utils.py tests/unit_tests/test_aloha_recap_converter.py tests/unit_tests/test_aloha_recap_returns.py tests/unit_tests/test_aloha_recap_transforms.py
git commit -s -m "fix: stabilize aloha sandwich recap validation"
```

## Self-Review

- Spec coverage: The plan covers ALOHA as a first-class RECAP robot type, HDF5-to-LeRobot conversion, HITL-aware returns, ALOHA value transforms, ALOHA advantage inference, teleop positive-label override, PyTorch checkpoint conversion requirement, sandwich configs, docs, unit tests, and smoke validation.
- Non-goals respected: The plan does not replace RECAP with OpenPI scripts, does not disguise ALOHA as another robot type, keeps continuous advantage math unchanged, and limits full training to explicit smoke or later full runs.
- Type consistency: The plan consistently uses `teleop_mask` for per-frame input metadata, `teleop_positive` for boolean-label overrides, `advantage_continuous` for the unchanged scalar advantage, `robot_type: aloha`, `model_type: pi05`, `action_dim: 14`, and `action_horizon: 10`.
- Placeholder scan: The only intentionally runtime-supplied value is `ALOHA_SANDWICH_VALUE_CHECKPOINT`, because it is produced by Step 2. All dataset, model, and config paths otherwise use concrete paths from the spec.

Plan complete and saved to `docs/superpowers/plans/2026-07-05-aloha-sandwich-recap.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
