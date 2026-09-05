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

"""Shared episode I/O and dual-view helpers for VLM Trend preprocess scripts."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def transition_observations(
    episode: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return action-aligned observations and their source offset."""
    observations = episode.get("observations", [])
    actions = episode.get("actions", [])
    offset = int(len(observations) == len(actions) + 1)
    count = min(len(actions), len(observations) - offset)
    return observations[offset : offset + count], offset


def first_success_transition(
    episode: dict[str, Any], transition_count: int
) -> int | None:
    """Return the first action-aligned success index, if present."""
    infos = episode.get("infos", [])
    actions = episode.get("actions", [])
    offset = int(len(infos) == len(actions) + 1)
    for index, info in enumerate(infos[offset : offset + transition_count]):
        value = info.get("success") if isinstance(info, dict) else None
        if bool(value.item() if hasattr(value, "item") else value):
            return index
    if bool(episode.get("success", False)) and transition_count:
        return transition_count - 1
    return None


def load_episode(path: str | Path, *, log_errors: bool = False) -> dict | None:
    """Load one collected episode, optionally logging unreadable files."""
    try:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)
    except (EOFError, pickle.UnpicklingError, OSError) as error:
        if log_errors:
            logger.warning("Skipping unreadable episode %s: %s", path, error)
        return None


def load_dual_view_sample(
    row: dict[str, Any], idx: int = 0
) -> tuple[list[Any], list[Any]]:
    """Load main and extra-view frames from one SFT manifest row.

    Args:
        row: JSONL record with a ``pkl_path`` pointing at dual-view frames.
        idx: Sample index used in error messages.

    Returns:
        ``(main_frames, extra_view_frames)``.

    Raises:
        ValueError: If ``pkl_path`` or the dual-view frame arrays are missing.
    """
    pkl_path = row.get("pkl_path")
    if not pkl_path:
        raise ValueError(f"Sample {idx} missing pkl_path")
    with open(pkl_path, "rb") as stream:
        payload = pickle.load(stream)
    main_frames = payload.get("main_frames")
    extra_view_frames = payload.get("extra_view_frames")
    if main_frames is None or extra_view_frames is None:
        raise ValueError(f"Sample {idx} pkl missing dual-view frame arrays")
    return main_frames, extra_view_frames


def potential_prompt(task: str, window_size: int, num_bins: int = 10) -> str:
    """Build the potential prompt recorded in the SFT manifest."""
    return (
        "You are estimating task-conditioned success potential for a robot "
        f"manipulation state. Task: {task}. The two synchronized videos show "
        f"the same {window_size}-frame history from two camera views. Predict "
        f"the final state's potential as exactly one digit from 0 to {num_bins - 1}, "
        f"where 0 is furthest from eventual success and {num_bins - 1} is closest."
    )


def progress_prompt(task: str, window_size: int, gap_steps: int | None = None) -> str:
    """Build the paired-window progress prompt recorded in the SFT manifest."""
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


def sample_source_hash(row: dict[str, Any]) -> int:
    """Hash a portable source identifier for deterministic feature sharding."""
    key = f"{row.get('source_run', '')}/{Path(row['source_episode_path']).name}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def potential_bin(value: float, num_bins: int) -> int:
    """Quantize a ``[0, 1]`` teacher value into ``0 .. num_bins-1``."""
    return int(round(np.clip(value, 0.0, 1.0) * (num_bins - 1)))


def progress_label(delta: float, deadband: float) -> str:
    """Map a teacher delta to ``up``, ``same``, or ``down`` using ``deadband``."""
    if delta > deadband:
        return "up"
    if delta < -deadband:
        return "down"
    return "same"


def to_uint8_rgb(image: Any) -> np.ndarray:
    """Convert a collected frame to a contiguous RGB uint8 array."""
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


def extract_extra_view_image(extra_view_images: Any) -> Any | None:
    """Return the first extra-view frame, if the payload is a batch."""
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
    """Extract synchronized main and extra-view frames for one window."""
    main_frames = []
    extra_view_frames = []
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


def build_messages(prompt: str, label: str) -> list[dict[str, Any]]:
    """Build the chat-template messages stored in SFT manifests."""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label}],
        },
    ]


def write_manifest(
    rows: list[dict[str, Any]], output_dir: str | Path, split: str
) -> str:
    """Write the JSONL consumed by the VLM Trend SFT dataset."""
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / "segments.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(manifest)


@dataclass(frozen=True)
class Candidate:
    """One potential, progress, or terminal-success window pending export."""

    source_path: str
    source_run: str
    split: str
    sample_type: str
    task: str
    episode_success: bool
    start_idx: int
    end_idx: int
    teacher_value: float
    teacher_delta: float
    answer: str
    progress_gap_steps: int | None = None
    terminal_success: bool = False
    is_complete: bool = False


def write_labeled_sample(
    candidate: Candidate,
    output_dir: Path,
    source_cache: dict[str, dict[str, Any]],
    num_bins: int,
    window_size: int,
) -> dict[str, Any] | None:
    """Write one dual-view pickle and return the matching SFT manifest row."""
    episode = source_cache.get(candidate.source_path)
    if episode is None:
        episode = load_episode(candidate.source_path, log_errors=True)
        if episode is None:
            return None
        source_cache.clear()
        source_cache[candidate.source_path] = episode

    observations = episode.get("observations", [])
    if candidate.sample_type == "progress":
        earlier = extract_dual_view_frames(
            observations,
            candidate.start_idx,
            candidate.start_idx + window_size - 1,
        )
        later = extract_dual_view_frames(
            observations,
            candidate.end_idx - window_size + 1,
            candidate.end_idx,
        )
        frames = (
            None
            if earlier is None or later is None
            else (earlier[0] + later[0], earlier[1] + later[1])
        )
    else:
        frames = extract_dual_view_frames(
            observations, candidate.start_idx, candidate.end_idx
        )
    if frames is None:
        return None
    main_frames, extra_frames = frames
    stem = os.path.splitext(os.path.basename(candidate.source_path))[0]
    source_run = candidate.source_run
    gap_suffix = (
        f"_gap_{candidate.progress_gap_steps}"
        if candidate.progress_gap_steps is not None
        else ""
    )
    prefix = "terminal_success" if candidate.terminal_success else candidate.sample_type
    sample_id = (
        f"{prefix}_{source_run}_{stem}_{candidate.start_idx:04d}_"
        f"{candidate.end_idx:04d}{gap_suffix}"
    )
    sample_pkl = output_dir / candidate.split / "pkl" / f"{sample_id}.pkl"
    sample_pkl.parent.mkdir(parents=True, exist_ok=True)
    with sample_pkl.open("wb") as stream:
        pickle.dump(
            {
                "main_frames": [to_uint8_rgb(frame) for frame in main_frames],
                "extra_view_frames": [to_uint8_rgb(frame) for frame in extra_frames],
                "label": candidate.answer,
                "sample_type": candidate.sample_type,
                "teacher_value": candidate.teacher_value,
                "teacher_delta": candidate.teacher_delta,
                "source_episode_path": candidate.source_path,
                "start_idx": candidate.start_idx,
                "end_idx": candidate.end_idx,
                "progress_gap_steps": candidate.progress_gap_steps,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    if candidate.terminal_success:
        prompt = (
            "Estimate task-conditioned success potential for this robot "
            f"manipulation state. Task: {candidate.task}. The two synchronized "
            f"videos show the same {window_size}-frame history from two camera views."
        )
    elif candidate.sample_type == "potential":
        prompt = potential_prompt(candidate.task, window_size, num_bins)
    else:
        prompt = progress_prompt(
            candidate.task, window_size, candidate.progress_gap_steps
        )
    segment_metadata = {
        "start_step": candidate.start_idx,
        "end_step": candidate.end_idx,
        "window_size": window_size,
        "progress_gap_steps": candidate.progress_gap_steps,
        "success": candidate.episode_success,
        "sample_type": candidate.sample_type,
    }
    supervision = {
        "score_name": (
            "terminal_success"
            if candidate.terminal_success
            else "state_success_value_potential"
        ),
        "teacher_value": candidate.teacher_value,
        "teacher_delta": candidate.teacher_delta,
    }
    if candidate.terminal_success:
        segment_metadata.update(
            target_name="terminal_success",
            is_complete=candidate.is_complete,
            target_type=(
                "success_observed" if candidate.answer == "1" else "online_negative"
            ),
            source_run=source_run,
        )
    else:
        segment_metadata.update(
            source_run=source_run,
            views=["main_images", "extra_view_images[0]"],
        )
        supervision.update(
            potential_bin=potential_bin(candidate.teacher_value, num_bins),
            progress_label=(
                candidate.answer if candidate.sample_type == "progress" else None
            ),
        )
    return {
        "task": candidate.task,
        "prompt": prompt,
        "question": prompt,
        "answer": candidate.answer,
        "pkl_path": str(sample_pkl.resolve()),
        "messages": build_messages(prompt, candidate.answer),
        "source_episode_path": candidate.source_path,
        "source_run": source_run,
        "segment_metadata": segment_metadata,
        "supervision": supervision,
    }
