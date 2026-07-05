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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ALOHA_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_TASK = "Assemble a sandwich."


@dataclass(frozen=True)
class EpisodePayload:
    """In-memory representation of one ALOHA HDF5 episode."""

    episode_index: int
    episode_name: str
    raw_reward: float
    teleop_segments: np.ndarray
    frames: list[dict[str, Any]]

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the episode."""
        return len(self.frames)

    @property
    def is_success(self) -> bool:
        """Return whether the episode reward indicates task success."""
        return bool(self.raw_reward > 0.0)


def _aloha_features(
    image_shape: tuple[int, int, int],
    include_velocity: bool,
    include_effort: bool,
) -> dict[str, dict[str, Any]]:
    """Build LeRobot feature metadata for ALOHA sandwich episodes."""
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

    for camera in ALOHA_CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channel"],
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

    return features


def _episode_success_array(raw_reward: float, num_frames: int) -> np.ndarray:
    """Return a per-frame success flag derived from an episode reward."""
    return np.full(num_frames, raw_reward > 0.0, dtype=bool)


def _teleop_mask(
    segments: np.ndarray,
    num_frames: int,
    episode_name: str,
) -> np.ndarray:
    """Convert half-open teleoperation segments into a per-frame mask."""
    segments = np.asarray(segments, dtype=np.int64)
    if segments.size == 0:
        segments = segments.reshape(0, 2)
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError(
            f"{episode_name}: invalid teleop segment array shape {segments.shape}"
        )

    mask = np.zeros(num_frames, dtype=np.int64)
    for start, end in segments:
        if start < 0 or end <= start or end > num_frames:
            raise ValueError(
                f"{episode_name}: invalid teleop segment [{start}, {end}) "
                f"for {num_frames} frames"
            )
        mask[start:end] = 1
    return mask


def _read_scalar_reward(ep: h5py.File, episode_name: str) -> float:
    """Read the scalar reward from an ALOHA HDF5 episode."""
    if "reward" not in ep:
        raise ValueError(f"{episode_name}: missing reward dataset")

    reward = np.asarray(ep["reward"][()])
    if reward.shape == ():
        return float(reward)
    if reward.shape == (1,):
        return float(reward[0])

    raise ValueError(
        f"{episode_name}: reward must be scalar or shape (1,), got {reward.shape}"
    )


def _read_teleop_segments(ep: h5py.File) -> np.ndarray:
    """Read teleoperation segments as an ``(N, 2)`` int64 array."""
    if "teleop_segments" not in ep:
        return np.empty((0, 2), dtype=np.int64)

    segments = np.asarray(ep["teleop_segments"], dtype=np.int64)
    if segments.size == 0:
        return segments.reshape(0, 2)
    if segments.ndim == 1 and segments.shape == (2,):
        return segments.reshape(1, 2)
    return segments


def _read_images(ep: h5py.File, episode_name: str) -> dict[str, np.ndarray]:
    """Read all required ALOHA camera streams from an episode."""
    if "observations" not in ep or "images" not in ep["observations"]:
        raise ValueError(f"{episode_name}: missing observations/images group")

    images_group = ep["observations"]["images"]
    images: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, int, int, int] | None = None
    for camera in ALOHA_CAMERAS:
        if camera not in images_group:
            raise ValueError(f"{episode_name}: missing camera {camera}")
        camera_images = np.asarray(images_group[camera])
        if camera_images.ndim != 4 or camera_images.shape[-1] != 3:
            raise ValueError(
                f"{episode_name}: camera {camera} must have shape (T, H, W, 3), "
                f"got {camera_images.shape}"
            )
        if reference_shape is None:
            reference_shape = camera_images.shape
        elif camera_images.shape != reference_shape:
            raise ValueError(
                f"{episode_name}: camera {camera} shape {camera_images.shape} "
                f"does not match {reference_shape}"
            )
        images[camera] = camera_images

    return images


def _load_episode_payload(
    episode_path: Path,
    task: str,
    episode_index: int = 0,
) -> EpisodePayload:
    """Load one ALOHA sandwich HDF5 episode into frame dictionaries."""
    episode_path = Path(episode_path)
    episode_name = episode_path.name

    with h5py.File(episode_path, "r") as ep:
        if "observations" not in ep:
            raise ValueError(f"{episode_name}: missing observations group")
        if "action" not in ep:
            raise ValueError(f"{episode_name}: missing action dataset")

        observations = ep["observations"]
        if "qpos" not in observations:
            raise ValueError(f"{episode_name}: missing observations/qpos dataset")

        state = np.asarray(observations["qpos"], dtype=np.float32)
        action = np.asarray(ep["action"], dtype=np.float32)
        velocity = (
            np.asarray(observations["qvel"], dtype=np.float32)
            if "qvel" in observations
            else None
        )
        effort = (
            np.asarray(observations["effort"], dtype=np.float32)
            if "effort" in observations
            else None
        )
        raw_reward = _read_scalar_reward(ep, episode_name)
        teleop_segments = _read_teleop_segments(ep)
        images = _read_images(ep, episode_name)

    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(
            f"{episode_name}: observations/qpos must have shape (T, 14), "
            f"got {state.shape}"
        )
    if action.shape != state.shape:
        raise ValueError(
            f"{episode_name}: action shape {action.shape} does not match "
            f"state shape {state.shape}"
        )

    num_frames = state.shape[0]
    for name, optional_array in (
        ("observations/qvel", velocity),
        ("observations/effort", effort),
    ):
        if optional_array is not None and optional_array.shape != state.shape:
            raise ValueError(
                f"{episode_name}: {name} shape {optional_array.shape} does not "
                f"match state shape {state.shape}"
            )

    for camera, camera_images in images.items():
        if camera_images.shape[0] != num_frames:
            raise ValueError(
                f"{episode_name}: camera {camera} has {camera_images.shape[0]} "
                f"frames, expected {num_frames}"
            )

    success = _episode_success_array(raw_reward, num_frames)
    teleop = _teleop_mask(teleop_segments, num_frames, episode_name)

    frames: list[dict[str, Any]] = []
    for frame_idx in range(num_frames):
        frame: dict[str, Any] = {
            "task": task,
            "is_success": np.asarray([success[frame_idx]], dtype=bool),
            "teleop_mask": np.asarray([teleop[frame_idx]], dtype=np.int64),
            "observation.state": state[frame_idx],
            "action": action[frame_idx],
        }
        if velocity is not None:
            frame["observation.velocity"] = velocity[frame_idx]
        if effort is not None:
            frame["observation.effort"] = effort[frame_idx]
        for camera, camera_images in images.items():
            frame[f"observation.images.{camera}"] = camera_images[frame_idx]
        frames.append(frame)

    return EpisodePayload(
        episode_index=episode_index,
        episode_name=episode_name,
        raw_reward=raw_reward,
        teleop_segments=teleop_segments,
        frames=frames,
    )


def _build_hil_segments_entry(payload: EpisodePayload) -> dict[str, Any]:
    """Build a JSON-serializable HIL segment metadata entry."""
    return {
        "episode_index": int(payload.episode_index),
        "episode_name": payload.episode_name,
        "num_frames": int(payload.num_frames),
        "raw_reward": float(payload.raw_reward),
        "is_success": bool(payload.is_success),
        "teleop_segments": payload.teleop_segments.astype(np.int64).tolist(),
    }
