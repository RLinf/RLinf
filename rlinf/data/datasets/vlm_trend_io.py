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

"""Shared VLM Trend episode I/O helpers used by preprocess and feature scripts.

Single source of truth for episode inspect, dual-view frame extraction, uint8
conversion, and split hashing. Preprocess and feature scripts should import the
public API from here; no private cross-imports between example modules.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any

import numpy as np
import torch


def to_uint8_rgb(image: Any) -> np.ndarray:
    """Convert a tensor/array image to ``uint8`` RGB of shape ``(H, W, 3)``."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")
    return image[..., :3]


def to_numpy_float32(value: Any) -> np.ndarray:
    """Convert tensor/array metadata to a float32 numpy array."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def extract_extra_view_image(extra_view_images: Any) -> Any | None:
    """Pick one extra-view frame from a (T, ...) or single-frame payload."""
    if extra_view_images is None:
        return None
    if torch.is_tensor(extra_view_images):
        if extra_view_images.ndim == 3:
            return extra_view_images
        if extra_view_images.ndim == 4 and extra_view_images.shape[0] > 0:
            return extra_view_images[0]
        return None

    extra_view_images = np.asarray(extra_view_images)
    if extra_view_images.ndim == 3:
        return extra_view_images
    if extra_view_images.ndim == 4 and extra_view_images.shape[0] > 0:
        return extra_view_images[0]
    return None


def extract_dual_view_frames(
    observations: list[dict[str, Any]], start_idx: int, end_idx: int
) -> tuple[list[Any], list[Any]] | None:
    """Return aligned main/extra frames over ``observations[start:end+1]``.

    Returns ``None`` when any step lacks a main or extra-view image.
    """
    main_frames: list[Any] = []
    extra_view_frames: list[Any] = []
    for idx in range(start_idx, end_idx + 1):
        obs = observations[idx]
        main_image = obs.get("main_images")
        extra_view_image = obs.get("third_view_images")
        if extra_view_image is None:
            extra_view_image = extract_extra_view_image(obs.get("extra_view_images"))
        if main_image is None or extra_view_image is None:
            return None
        main_frames.append(main_image)
        extra_view_frames.append(extra_view_image)
    return main_frames, extra_view_frames


def load_episode_pickle(path: str) -> dict[str, Any] | None:
    """Load one rollout episode pickle; return ``None`` on read errors."""
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (EOFError, pickle.UnpicklingError, OSError):
        return None


def inspect_episode(path: str, window_size: int) -> dict[str, Any] | None:
    """Read the metadata needed to sample one rollout episode.

    Returns ``None`` when the episode is unreadable or too short.
    """
    episode = load_episode_pickle(path)
    if episode is None:
        return None
    observations = episode.get("observations", [])
    actions = episode.get("actions", [])
    if len(observations) < window_size or len(actions) < window_size:
        return None
    return {
        "path": os.path.abspath(path),
        "observations": observations,
        "actions": actions,
        "terminated": episode.get("terminated", []),
        "truncated": episode.get("truncated", []),
        "success": bool(episode.get("success", False)),
        "infos": episode.get("infos", []),
        "task": (
            episode.get("task")
            or episode.get("task_description")
            or episode.get("task_name")
            or ""
        ),
        "episode_id": episode.get("episode_id"),
        "env_idx": episode.get("env_idx"),
    }


def split_for(path: str, val_split: float) -> str:
    """Assign an episode to a stable source-level train/eval split."""
    fraction = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 2**32
    return "eval" if fraction < val_split else "train"


def source_episode_hash(path: str) -> int:
    """Stable integer hash of a source-episode path for rank sharding."""
    return int(hashlib.sha256(path.encode()).hexdigest()[:16], 16)


def potential_prompt(task: str, window_size: int, num_bins: int) -> str:
    """Build the absolute potential VLM user prompt for one window."""
    return (
        "You are estimating task-conditioned success potential for a robot "
        f"manipulation state. Task: {task}. The two synchronized videos show "
        f"the same {window_size}-frame history from two camera views. Predict "
        f"the final state's potential as exactly one digit from 0 to {num_bins - 1}, "
        f"where 0 is furthest from eventual success and {num_bins - 1} is closest."
    )


def progress_prompt(task: str, window_size: int, gap_steps: int | None = None) -> str:
    """Build the relative progress VLM user prompt for a pair of windows."""
    gap_steps = window_size if gap_steps is None else gap_steps
    relation = (
        "immediately adjacent"
        if gap_steps == window_size
        else f"separated by {gap_steps} environment steps"
    )
    return (
        "You are judging local task progress in a robot manipulation trajectory. "
        f"Task: {task}. In each synchronized camera video, the first {window_size} "
        f"frames are the earlier clip and the next {window_size} frames are the "
        f"later clip; their final states are {relation}. Compare their final states. "
        "Answer with exactly one word: up, same, or down."
    )
