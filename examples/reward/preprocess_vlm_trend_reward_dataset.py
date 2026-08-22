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

"""Preprocess VLM Trend reward data into split train/eval pkl datasets.

Example:
    python examples/reward/preprocess_vlm_trend_reward_dataset.py \
        --raw-data-path logs/xxx/collected_data \
        --output-dir logs/xxx/processed_vlm_trend_reward_data

The exported JSONL points to per-sample pkl files. VLMTrendRewardSFTDataset
loads the two 5-frame video arrays directly from those pkl files, avoiding the
slow small-mp4 export path.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from rlinf.models.embodiment.modules.utils import make_mlp
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


def _load_episode(path: str | Path, *, log_errors: bool = False) -> dict | None:
    """Load one collected episode, optionally logging unreadable files."""
    try:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)
    except (EOFError, pickle.UnpicklingError, OSError) as error:
        if log_errors:
            logger.warning("Skipping unreadable episode %s: %s", path, error)
        return None


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


def _compute_sample_indices(
    n: int, num_samples_per_episode: int, keep_last_window: bool
) -> list[int]:
    """Compute sampled window indices while preserving current behavior."""
    if keep_last_window:
        if num_samples_per_episode == 1:
            return [n - 1]

        k = num_samples_per_episode - 1
        non_last_n = n - 1
        if k >= non_last_n:
            non_last_indices = list(range(non_last_n))
        elif k == 1:
            non_last_indices = [0]
        else:
            non_last_indices = [int(i * (non_last_n - 1) / (k - 1)) for i in range(k)]
        return sorted(set(non_last_indices + [n - 1]))

    if num_samples_per_episode == 1:
        return [n - 1]
    return sorted(
        {
            int(i * (n - 1) / (num_samples_per_episode - 1))
            for i in range(num_samples_per_episode)
        }
    )


def _to_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _to_uint8_rgb(image: Any) -> np.ndarray:
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


def _extract_extra_view_image(extra_view_images: Any) -> Any | None:
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


def _extract_dual_view_frames(
    observations: list[dict[str, Any]], start_idx: int, end_idx: int
) -> tuple[list[Any], list[Any]] | None:
    main_frames = []
    extra_view_frames = []
    for idx in range(start_idx, end_idx + 1):
        obs = observations[idx]
        main_image = obs.get("main_images")
        extra_view_image = obs.get("third_view_images")
        if extra_view_image is None:
            extra_view_image = _extract_extra_view_image(obs.get("extra_view_images"))
        if main_image is None or extra_view_image is None:
            return None
        main_frames.append(main_image)
        extra_view_frames.append(extra_view_image)
    return main_frames, extra_view_frames


def _build_prompt(task: str, window_size: int) -> str:
    return (
        f"You are currently performing the task: {task}. "
        f"You are given two synchronized {window_size}-frame videos from different "
        "camera views (main view and third-person view) of the same robot action "
        "window. Judge whether the action trend is positive, negative, or unclear. "
        "Answer with exactly one word: positive, negative, or unclear."
    )


def _build_messages(prompt: str, label: str) -> list[dict[str, Any]]:
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


def _build_reversed_negative_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        **sample,
        "sample_id": f"{sample['sample_id']}_reverse_negative",
        "label": "negative",
        "score": -abs(float(sample["score"])),
        "main_frames": list(reversed(sample["main_frames"])),
        "extra_view_frames": list(reversed(sample["extra_view_frames"])),
        "augmentation": "reverse_positive",
    }


def load_episodes_with_labels(
    data_path: str,
    window_size: int = 5,
    stride: int = 1,
    delta_threshold: float = 0.05,
    tail_unclear_ratio: float = 0.15,
    num_samples_per_episode: int = 0,
    keep_last_window: bool = True,
    task_description: Optional[str] = None,
    load_workers: int = 256,
) -> list[dict]:
    """Load episodes with per-window labels from collected data."""
    pkl_files = sorted(glob(os.path.join(data_path, "*.pkl")))
    logger.info(f"Found {len(pkl_files)} episode files in {data_path}")

    episodes = []
    label_counter = Counter()

    def _load_one_episode(pkl_path: str) -> Optional[dict]:
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)

            observations = episode.get("observations", [])
            score_values = episode.get("gae", None)
            score_source = "gae"
            if score_values is None or len(score_values) == 0:
                score_values = episode.get("rewards", [])
                score_source = "rewards"
            seq_len = min(len(observations), len(score_values))
            if seq_len < window_size:
                return None

            start_indices = list(range(0, seq_len - window_size + 1, stride))
            if not start_indices:
                return None

            tail_start = int(len(start_indices) * (1.0 - float(tail_unclear_ratio)))
            task = str(
                episode.get("task")
                or episode.get("task_description")
                or task_description
                or "robot manipulation progress judgment"
            )

            all_samples = []
            for sample_idx, start_idx in enumerate(start_indices):
                end_idx = start_idx + window_size - 1
                frames = _extract_dual_view_frames(observations, start_idx, end_idx)
                if frames is None:
                    continue

                start_score = _to_scalar(score_values[start_idx])
                end_score = _to_scalar(score_values[end_idx])
                score = end_score - start_score
                if abs(score) <= delta_threshold:
                    label = "unclear"
                elif score > 0:
                    label = "positive"
                else:
                    label = "negative"
                if sample_idx >= tail_start:
                    label = "unclear"

                sample_id = (
                    f"{os.path.splitext(os.path.basename(pkl_path))[0]}"
                    f"_frames_{start_idx:04d}_{end_idx:04d}"
                )
                prompt = _build_prompt(task, window_size)
                main_frames, extra_view_frames = frames
                all_samples.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "prompt": prompt,
                        "label": label,
                        "score": score,
                        "start_gae": start_score,
                        "end_gae": end_score,
                        "score_source": score_source,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "main_frames": main_frames,
                        "extra_view_frames": extra_view_frames,
                        "source_episode_path": pkl_path,
                        "episode_id": episode.get("episode_id"),
                        "env_idx": episode.get("env_idx"),
                        "success": episode.get("success"),
                        "augmentation": None,
                    }
                )

            if not all_samples:
                return None

            if (
                num_samples_per_episode > 0
                and len(all_samples) > num_samples_per_episode
            ):
                indices = _compute_sample_indices(
                    n=len(all_samples),
                    num_samples_per_episode=num_samples_per_episode,
                    keep_last_window=keep_last_window,
                )
                sampled = [all_samples[i] for i in indices]
            else:
                sampled = all_samples

            return {
                "samples": sampled,
                "source_episode_path": pkl_path,
                "episode_key": os.path.abspath(pkl_path),
            }

        except Exception as e:
            logger.warning(f"Failed to load {pkl_path}: {e}")
            return None

    if load_workers <= 1:
        for pkl_path in tqdm(pkl_files, desc="Loading episodes", unit="episode"):
            loaded = _load_one_episode(pkl_path)
            if loaded is None:
                continue
            label_counter.update(sample["label"] for sample in loaded["samples"])
            episodes.append(loaded)
    else:
        with ThreadPoolExecutor(max_workers=load_workers) as executor:
            futures = {
                executor.submit(_load_one_episode, pkl_path): pkl_path
                for pkl_path in pkl_files
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading episodes",
                unit="episode",
            ):
                loaded = future.result()
                if loaded is None:
                    continue
                label_counter.update(sample["label"] for sample in loaded["samples"])
                episodes.append(loaded)

    total_samples = sum(len(ep["samples"]) for ep in episodes)
    logger.info(
        f"Loaded {len(episodes)} episodes, {total_samples} windows: "
        f"{dict(sorted(label_counter.items()))}"
    )
    return episodes


def balance_and_split_by_episode(
    episodes: list[dict],
    val_split: float = 0.1,
    balance_labels: bool = True,
    max_samples_per_label: Optional[int] = None,
    eval_max_samples_per_label: Optional[int] = None,
    reverse_positive_as_negative: bool = True,
    random_seed: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Split by episode and optionally rebalance positive/negative/unclear."""
    if not episodes:
        logger.error("No episodes provided!")
        return [], []

    rng = random.Random(random_seed) if random_seed is not None else random

    episodes_copy = list(episodes)
    rng.shuffle(episodes_copy)
    val_ep_count = max(1, int(len(episodes_copy) * val_split))
    val_episodes = episodes_copy[:val_ep_count]
    train_episodes = episodes_copy[val_ep_count:]
    train_episode_keys = {episode["episode_key"] for episode in train_episodes}
    val_episode_keys = {episode["episode_key"] for episode in val_episodes}
    overlap_episode_keys = train_episode_keys & val_episode_keys
    if overlap_episode_keys:
        raise RuntimeError(
            "Episode leakage detected between train and eval splits: "
            f"{sorted(overlap_episode_keys)[:5]}"
        )

    logger.info(
        f"Episode split: {len(train_episodes)} train eps, {len(val_episodes)} eval eps, "
        f"overlap={len(overlap_episode_keys)}"
    )
    if eval_max_samples_per_label is None and max_samples_per_label is not None:
        eval_ratio_to_train = len(val_episodes) / max(1, len(train_episodes))
        eval_max_samples_per_label = max(
            1, int(round(max_samples_per_label * eval_ratio_to_train))
        )
    logger.info(
        "Per-label caps: "
        f"train={max_samples_per_label}, eval={eval_max_samples_per_label}"
    )

    def extract_and_sample(
        ep_list: list[dict], split_name: str, per_label_cap: Optional[int]
    ) -> list[dict]:
        grouped_samples = {"positive": [], "negative": [], "unclear": []}
        for episode in ep_list:
            for sample in episode["samples"]:
                grouped_samples[sample["label"]].append(sample)
                if reverse_positive_as_negative and sample["label"] == "positive":
                    grouped_samples["negative"].append(
                        _build_reversed_negative_sample(sample)
                    )

        raw_counts = {label: len(samples) for label, samples in grouped_samples.items()}
        logger.info(f"{split_name} raw counts: {raw_counts}")

        for samples in grouped_samples.values():
            rng.shuffle(samples)

        if balance_labels:
            non_empty_counts = [
                len(samples) for samples in grouped_samples.values() if len(samples) > 0
            ]
            if len(non_empty_counts) >= 2:
                keep_count = min(non_empty_counts)
                if per_label_cap is not None:
                    keep_count = min(keep_count, per_label_cap)
                grouped_samples = {
                    label: samples[:keep_count]
                    for label, samples in grouped_samples.items()
                    if len(samples) > 0
                }
            elif per_label_cap is not None:
                grouped_samples = {
                    label: samples[:per_label_cap]
                    for label, samples in grouped_samples.items()
                }
        elif per_label_cap is not None:
            grouped_samples = {
                label: samples[:per_label_cap]
                for label, samples in grouped_samples.items()
            }

        merged_samples = []
        for samples in grouped_samples.values():
            merged_samples.extend(samples)
        rng.shuffle(merged_samples)

        final_counts = dict(Counter(sample["label"] for sample in merged_samples))
        logger.info(f"{split_name} final counts: {final_counts}")
        return merged_samples

    train_samples = extract_and_sample(train_episodes, "train", max_samples_per_label)
    eval_samples = extract_and_sample(val_episodes, "eval", eval_max_samples_per_label)
    return train_samples, eval_samples


def preprocess_and_save_reward_datasets(
    raw_data_path: str,
    output_dir: str,
    window_size: int = 5,
    stride: int = 1,
    delta_threshold: float = 0.05,
    tail_unclear_ratio: float = 0.15,
    num_samples_per_episode: int = 0,
    keep_last_window: bool = True,
    val_split: float = 0.1,
    balance_labels: bool = True,
    max_samples_per_label: Optional[int] = None,
    eval_max_samples_per_label: Optional[int] = None,
    reverse_positive_as_negative: bool = True,
    fps: int = 2,
    task_description: Optional[str] = None,
    random_seed: Optional[int] = None,
    load_workers: int = 256,
    write_workers: int = 512,
) -> dict:
    """Build train/eval VLM Trend reward datasets from raw data."""
    episodes = load_episodes_with_labels(
        raw_data_path,
        window_size=window_size,
        stride=stride,
        delta_threshold=delta_threshold,
        tail_unclear_ratio=tail_unclear_ratio,
        num_samples_per_episode=num_samples_per_episode,
        keep_last_window=keep_last_window,
        task_description=task_description,
        load_workers=load_workers,
    )
    if len(episodes) == 0:
        raise ValueError(f"No episodes loaded from raw data path: {raw_data_path}")

    train_samples, eval_samples = balance_and_split_by_episode(
        episodes=episodes,
        val_split=val_split,
        balance_labels=balance_labels,
        max_samples_per_label=max_samples_per_label,
        eval_max_samples_per_label=eval_max_samples_per_label,
        reverse_positive_as_negative=reverse_positive_as_negative,
        random_seed=random_seed,
    )

    def _save_split(samples: list[dict], split_name: str) -> tuple[str, dict[str, int]]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        pkl_dir = os.path.join(split_dir, "pkl")
        os.makedirs(pkl_dir, exist_ok=True)

        def _build_row_and_write(sample: dict) -> dict:
            clip_stem = f"{sample['label']}_{sample['sample_id']}"
            pkl_path = os.path.abspath(os.path.join(pkl_dir, f"{clip_stem}.pkl"))
            if not (os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0):
                with open(pkl_path, "wb") as f:
                    pickle.dump(
                        {
                            "main_frames": [
                                _to_uint8_rgb(frame) for frame in sample["main_frames"]
                            ],
                            "extra_view_frames": [
                                _to_uint8_rgb(frame)
                                for frame in sample["extra_view_frames"]
                            ],
                            "label": sample["label"],
                            "score": sample["score"],
                            "source_episode_path": sample["source_episode_path"],
                            "start_idx": sample["start_idx"],
                            "end_idx": sample["end_idx"],
                            "augmentation": sample["augmentation"],
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            return {
                "task": sample["task"],
                "prompt": sample["prompt"],
                "question": sample["prompt"],
                "answer": sample["label"],
                "pkl_path": pkl_path,
                "messages": _build_messages(sample["prompt"], sample["label"]),
                "source_episode_path": sample["source_episode_path"],
                "segment_metadata": {
                    "start_step": sample["start_idx"],
                    "end_step": sample["end_idx"],
                    "window_size": window_size,
                    "episode_id": sample["episode_id"],
                    "env_idx": sample["env_idx"],
                    "success": sample["success"],
                    "augmentation": sample["augmentation"],
                    "views": ["main_images", "extra_view_images[0]"],
                },
                "supervision": {
                    "label": sample["label"],
                    "score": sample["score"],
                    "score_name": "gae_delta_window",
                    "score_source": sample["score_source"],
                    "delta_threshold": delta_threshold,
                    "start_gae": sample["start_gae"],
                    "end_gae": sample["end_gae"],
                },
            }

        rows = []
        if write_workers <= 1:
            for sample in tqdm(
                samples,
                desc=f"Saving {split_name} samples",
                unit="sample",
            ):
                rows.append(_build_row_and_write(sample))
        else:
            rows_by_index: list[dict | None] = [None] * len(samples)
            with ThreadPoolExecutor(max_workers=write_workers) as executor:
                futures = {
                    executor.submit(_build_row_and_write, sample): idx
                    for idx, sample in enumerate(samples)
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Saving {split_name} samples",
                    unit="sample",
                ):
                    rows_by_index[futures[future]] = future.result()
            rows = [row for row in rows_by_index if row is not None]

        manifest_path = os.path.join(split_dir, "segments.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        label_counts = dict(Counter(row["answer"] for row in rows))
        logger.info(
            f"Saved processed VLM Trend reward {split_name} split to "
            f"{manifest_path}: {len(rows)}"
        )
        return manifest_path, label_counts

    train_manifest, train_label_counts = _save_split(train_samples, "train")
    eval_manifest, eval_label_counts = _save_split(eval_samples, "eval")

    metadata = {
        "raw_data_path": raw_data_path,
        "output_dir": output_dir,
        "window_size": window_size,
        "stride": stride,
        "delta_threshold": delta_threshold,
        "tail_unclear_ratio": tail_unclear_ratio,
        "num_samples_per_episode": num_samples_per_episode,
        "keep_last_window": keep_last_window,
        "val_split": val_split,
        "balance_labels": balance_labels,
        "max_samples_per_label": max_samples_per_label,
        "eval_max_samples_per_label": eval_max_samples_per_label,
        "reverse_positive_as_negative": reverse_positive_as_negative,
        "fps": fps,
        "task_description": task_description,
        "random_seed": random_seed,
        "load_workers": load_workers,
        "write_workers": write_workers,
        "export_format": "pkl",
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "train_label_counts": train_label_counts,
        "eval_label_counts": eval_label_counts,
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
    }

    with open(
        os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VLM Trend reward dataset from raw episode .pkl files."
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        required=True,
        help="Path to raw collected_data directory containing .pkl episode files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/processed_vlm_trend_reward_data",
        help="Output directory for processed train/eval pkl datasets.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of frames in each exported dual-view window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive video windows.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Absolute GAE-delta threshold used to label windows as unclear.",
    )
    parser.add_argument(
        "--tail-unclear-ratio",
        type=float,
        default=0.15,
        help="Force the tail portion of windows in each episode to unclear.",
    )
    parser.add_argument(
        "--num-samples-per-episode",
        type=int,
        default=0,
        help="Number of sampled windows per episode. Use 0 for all windows.",
    )
    parser.add_argument(
        "--keep-last-window",
        dest="keep_last_window",
        action="store_true",
        default=True,
        help="Always include each episode's last valid window when sampling.",
    )
    parser.add_argument(
        "--no-keep-last-window",
        dest="keep_last_window",
        action="store_false",
        help="Allow sampling to exclude each episode's last valid window.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of episodes for evaluation.",
    )
    parser.add_argument(
        "--balance-labels",
        dest="balance_labels",
        action="store_true",
        default=True,
        help="Rebalance positive/negative/unclear windows within each split.",
    )
    parser.add_argument(
        "--no-balance-labels",
        dest="balance_labels",
        action="store_false",
        help="Keep the original label distribution in each split.",
    )
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=None,
        help="Optional train cap for each label after split and rebalancing.",
    )
    parser.add_argument(
        "--eval-max-samples-per-label",
        type=int,
        default=None,
        help=(
            "Optional eval cap for each label. If omitted, it is derived from "
            "--max-samples-per-label using the eval/train episode ratio."
        ),
    )
    parser.add_argument(
        "--reverse-positive-as-negative",
        dest="reverse_positive_as_negative",
        action="store_true",
        default=True,
        help="Reverse positive windows to synthesize additional negative samples.",
    )
    parser.add_argument(
        "--no-reverse-positive-as-negative",
        dest="reverse_positive_as_negative",
        action="store_false",
        help="Disable reversing positive windows into synthetic negative samples.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Kept for backward-compatible CLI calls; pkl export does not use FPS.",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=256,
        help="Number of parallel workers for loading and slicing episode pkl files.",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=512,
        help="Number of parallel workers for writing per-sample pkl files.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Fallback task description when raw episodes do not provide one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split and sampling.",
    )
    return parser.parse_args(argv)


def run_trend(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = preprocess_and_save_reward_datasets(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        delta_threshold=args.delta_threshold,
        tail_unclear_ratio=args.tail_unclear_ratio,
        num_samples_per_episode=args.num_samples_per_episode,
        keep_last_window=args.keep_last_window,
        val_split=args.val_split,
        balance_labels=args.balance_labels,
        max_samples_per_label=args.max_samples_per_label,
        eval_max_samples_per_label=args.eval_max_samples_per_label,
        reverse_positive_as_negative=args.reverse_positive_as_negative,
        fps=args.fps,
        task_description=args.task_description,
        random_seed=args.seed,
        load_workers=args.load_workers,
        write_workers=args.write_workers,
    )

    print("=" * 80)
    print("Dual-view trend reward dataset preprocessing complete")
    print(
        f"Train split: {metadata['train_manifest']} "
        f"({metadata['num_train_samples']} samples)"
    )
    print(
        f"Eval split:  {metadata['eval_manifest']} "
        f"({metadata['num_eval_samples']} samples)"
    )
    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    print("=" * 80)


def write_manifest(
    rows: list[dict[str, Any]], output_dir: str | Path, split: str
) -> str:
    """Write the JSONL consumed by the existing Qwen SFT dataset."""
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / "segments.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(manifest)


def build_terminal_success_rows(
    raw_data_paths: list[str],
    output_dir: str | Path,
    window_size: int,
    interval: int,
    val_split: float,
    workers: int,
    seed: int,
    task_description: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Materialize unbalanced terminal-success windows for Qwen SFT."""
    entries = sorted(
        (
            (Path(root).resolve(), path.resolve())
            for root in raw_data_paths
            for path in Path(root).glob("*.pkl")
        ),
        key=lambda entry: str(entry[1]),
    )

    def inspect(entry: tuple[Path, Path]) -> dict[str, Any] | None:
        root, path = entry
        episode = _load_episode(path)
        if episode is None:
            return None
        observations, observation_offset = transition_observations(episode)
        if len(observations) < window_size:
            return None
        end_step = len(observations) - 1
        first_success = first_success_transition(episode, len(observations))
        success = bool(episode.get("success", False) or first_success is not None)
        if first_success is not None:
            end_step = min(end_step, first_success)
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or episode.get("task_name")
            or task_description
            or ""
        ).strip()
        if not task:
            return None
        return {
            "path": path,
            "task": task,
            "end_step": end_step,
            "observation_offset": observation_offset,
            "success_steps": [end_step] if success else [],
            "success": success,
            "is_complete": (
                success
                or bool(episode.get("terminated", []) and episode["terminated"][-1])
                or bool(episode.get("truncated", []) and episode["truncated"][-1])
            ),
            "source_run": root.parent.name,
            "split_key": str(path),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        items = [item for item in executor.map(inspect, entries) if item is not None]
    rows_by_split = {"train": [], "eval": []}
    stats: dict[str, Any] = {"input_episodes": len(entries), "splits": {}}
    output_dir = Path(output_dir)
    for item in items:
        fraction = (
            int(hashlib.sha256(item["split_key"].encode()).hexdigest()[:8], 16) / 2**32
        )
        split = "eval" if fraction < val_split else "train"
        first = window_size - 1
        end_steps = list(range(first, item["end_step"] + 1, interval))
        success_steps = {step for step in item["success_steps"] if step >= first}
        end_steps.extend(success_steps - set(end_steps))
        source_cache: dict[str, dict[str, Any]] = {}
        for end_step in sorted(end_steps):
            answer = "1" if end_step in success_steps else "0"
            source_end = end_step + item["observation_offset"]
            candidate = Candidate(
                source_path=str(item["path"]),
                source_run=item["source_run"],
                split=split,
                sample_type="potential",
                task=item["task"],
                episode_success=answer == "1",
                start_idx=source_end - window_size + 1,
                end_idx=source_end,
                teacher_value=float(answer),
                teacher_delta=0.0,
                answer=answer,
                terminal_success=True,
                is_complete=item["is_complete"],
            )
            row = _write_sample(
                candidate,
                output_dir,
                source_cache,
                num_bins=10,
                window_size=window_size,
            )
            if row is not None:
                rows_by_split[split].append(row)
    for split, rows in rows_by_split.items():
        random.Random(seed + (split == "eval")).shuffle(rows)
        positives = sum(row["answer"] == "1" for row in rows)
        stats["splits"][split] = {
            "positive": positives,
            "negative": len(rows) - positives,
            "interval": interval,
        }
    stats["complete_episodes"] = sum(item["is_complete"] for item in items)
    stats["partial_episodes"] = len(items) - stats["complete_episodes"]
    return rows_by_split, stats


def load_value_model(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    """Load the lightweight state-success teacher checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    channels = [int(config["hidden_dim"])] * int(config["num_layers"]) + [1]
    model_layers = make_mlp(
        int(config["state_dim"]) * int(config["history_size"]),
        channels,
        act_builder=nn.SiLU,
        last_act=False,
        use_layer_norm=True,
    )
    dropout = float(config.get("dropout", 0.0))
    if dropout > 0:
        layers_with_dropout = []
        for layer in model_layers:
            layers_with_dropout.append(layer)
            if isinstance(layer, nn.SiLU):
                layers_with_dropout.append(nn.Dropout(dropout))
        model_layers = layers_with_dropout
    model = nn.Sequential(*model_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        config,
        np.asarray(config["mean"], dtype=np.float32),
        np.asarray(config["std"], dtype=np.float32),
    )


def score_states(
    model: nn.Module,
    config: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    states: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score all episode states with the state-success teacher."""
    history_size = int(config["history_size"])
    first = states[0]
    inputs = np.stack(
        [
            np.concatenate(
                [
                    states[index - offset] if index >= offset else first
                    for offset in range(history_size - 1, -1, -1)
                ]
            ).astype(np.float32)
            for index in range(len(states))
        ]
    )
    inputs = (inputs - mean[None]) / std[None]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch).squeeze(-1)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _run_terminal_success(args: argparse.Namespace) -> None:
    rows_by_split, stats = build_terminal_success_rows(
        args.raw_data_path,
        args.output_dir,
        args.window_size,
        args.interval,
        args.val_split,
        args.workers,
        args.seed,
        args.task_description,
    )
    output_dir = Path(args.output_dir)
    for split, rows in rows_by_split.items():
        write_manifest(rows, output_dir, split)
    (output_dir / "dataset_info.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("%s", json.dumps(stats, indent=2))


def _add_terminal_success_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-description", default=None)


@dataclass(frozen=True)
class Candidate:
    """One potential or progress window candidate pending global sampling."""

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


def smooth_values(values: np.ndarray, window_size: int) -> np.ndarray:
    """Denoise a teacher trajectory with an edge-padded moving average."""
    if window_size <= 1:
        return values
    if window_size % 2 == 0:
        raise ValueError("temporal_smoothing_window must be odd")
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float32)
    return np.clip(np.convolve(padded, kernel, mode="valid"), 0.0, 1.0)


def _write_sample(
    candidate: Candidate,
    output_dir: Path,
    source_cache: dict[str, dict[str, Any]],
    num_bins: int,
    window_size: int,
) -> dict[str, Any] | None:
    episode = source_cache.get(candidate.source_path)
    if episode is None:
        episode = _load_episode(candidate.source_path, log_errors=True)
        if episode is None:
            return None
        source_cache.clear()
        source_cache[candidate.source_path] = episode

    observations = episode.get("observations", [])
    if candidate.sample_type == "progress":
        earlier = _extract_dual_view_frames(
            observations,
            candidate.start_idx,
            candidate.start_idx + window_size - 1,
        )
        later = _extract_dual_view_frames(
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
        frames = _extract_dual_view_frames(
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
                "main_frames": [_to_uint8_rgb(frame) for frame in main_frames],
                "extra_view_frames": [_to_uint8_rgb(frame) for frame in extra_frames],
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
        "messages": _build_messages(prompt, candidate.answer),
        "source_episode_path": candidate.source_path,
        "source_run": source_run,
        "segment_metadata": segment_metadata,
        "supervision": supervision,
    }


def run_potential(args: argparse.Namespace) -> dict[str, Any]:
    """Build potential/progress labels through the existing Qwen SFT pipeline."""
    if args.progress_gap_steps is None:
        args.progress_gap_steps = [args.window_size]
    if not 2 <= args.num_bins <= 10:
        raise ValueError("num_bins must be between 2 and 10 for single digit labels")
    if args.temporal_smoothing_window < 1 or args.temporal_smoothing_window % 2 == 0:
        raise ValueError("temporal_smoothing_window must be a positive odd integer")
    if any(gap < 1 for gap in args.progress_gap_steps):
        raise ValueError("progress_gap_steps must contain only positive values")
    if not 0.0 <= args.val_split <= 1.0:
        raise ValueError(
            f"val_split must be in [0, 1], got {args.val_split}; "
            "use 0 for train-only or a fraction for the eval hold-out."
        )
    args.progress_gap_steps = sorted(set(args.progress_gap_steps))

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files_by_root = {
        str(Path(root).resolve()): sorted(glob(os.path.join(root, "*.pkl")))
        for root in args.raw_data_path
    }
    pkl_files = list(
        dict.fromkeys(path for files in files_by_root.values() for path in files)
    )
    if args.max_episodes is not None:
        pkl_files = pkl_files[: args.max_episodes]
    if not pkl_files:
        raise ValueError(f"No episode pkl files found in {args.raw_data_path}")

    source_run_by_path = {
        path: Path(root).parent.name
        for root, root_files in files_by_root.items()
        for path in root_files
    }
    split_by_path: dict[str, str] = {}
    for root_files in files_by_root.values():
        root_files = [path for path in root_files if path in pkl_files]
        rng.shuffle(root_files)
        eval_count = (
            0
            if args.val_split <= 0
            else min(
                len(root_files),
                max(1, int(round(len(root_files) * args.val_split))),
            )
        )
        split_by_path.update(
            {
                path: ("eval" if index < eval_count else "train")
                for index, path in enumerate(root_files)
            }
        )
    if args.only_split is not None:
        pkl_files = [
            path for path in pkl_files if split_by_path[path] == args.only_split
        ]
        if not pkl_files:
            raise ValueError(
                f"--only-split {args.only_split!r} selected no episodes after "
                f"applying --val-split {args.val_split}; for example "
                "--only-split eval requires val_split > 0."
            )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std = load_value_model(args.value_checkpoint, device)
    candidates: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    episode_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    first_end = args.window_size - 1
    for pkl_path in tqdm(pkl_files, desc="Scoring episodes", unit="episode"):
        episode = _load_episode(pkl_path, log_errors=True)
        if episode is None:
            skipped["unreadable_episode"] += 1
            continue
        observations, observation_offset = transition_observations(episode)
        first_success = first_success_transition(episode, len(observations))
        if first_success is not None:
            observations = observations[: first_success + 1]
        if len(observations) < args.window_size * 2:
            skipped["short_episode"] += 1
            continue
        states = [
            np.asarray(
                observation["states"].detach().cpu().numpy()
                if torch.is_tensor(observation["states"])
                else observation["states"],
                dtype=np.float32,
            ).reshape(-1)
            for observation in observations
            if "states" in observation
        ]
        if len(states) != len(observations):
            skipped["missing_states"] += 1
            continue
        values = score_states(
            model, cfg, mean, std, states, device, args.score_batch_size
        ).reshape(-1)
        values = smooth_values(values, args.temporal_smoothing_window)
        split = split_by_path[pkl_path]
        success = bool(episode.get("success", False))
        episode_counts[f"{split}_{'success' if success else 'failure'}"] += 1
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or args.task_description
            or ""
        ).strip()
        if not task:
            skipped["missing_task_description"] += 1
            continue
        for end_idx in range(first_end, len(values), args.stride):
            start_idx = end_idx - args.window_size + 1
            value = float(values[end_idx])
            candidates[(split, "potential")].append(
                Candidate(
                    pkl_path,
                    source_run_by_path[pkl_path],
                    split,
                    "potential",
                    task,
                    success,
                    start_idx + observation_offset,
                    end_idx + observation_offset,
                    value,
                    0.0,
                    str(potential_bin(value, args.num_bins)),
                )
            )
            for gap_steps in args.progress_gap_steps:
                earlier_end = end_idx - gap_steps
                if earlier_end < first_end:
                    continue
                delta = value - float(values[earlier_end])
                candidates[(split, "progress")].append(
                    Candidate(
                        pkl_path,
                        source_run_by_path[pkl_path],
                        split,
                        "progress",
                        task,
                        success,
                        earlier_end - args.window_size + 1 + observation_offset,
                        end_idx + observation_offset,
                        value,
                        delta,
                        progress_label(delta, args.progress_deadband),
                        gap_steps,
                    )
                )

    selected = []
    selection = {}
    for (split, sample_type), items in sorted(candidates.items()):
        suffix = "train" if split == "train" else "eval"
        limit = int(getattr(args, f"{sample_type}_samples_{suffix}"))
        chosen = (
            rng.sample(items, limit)
            if limit > 0 and len(items) > limit
            else list(items)
        )
        selected.extend(chosen)
        selection[f"{split}/{sample_type}"] = {
            "seen": len(items),
            "selected": len(chosen),
            "method": "uniform_without_replacement",
        }
    selected.sort(key=lambda item: (item.source_path, item.sample_type, item.start_idx))
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_cache: dict[str, dict[str, Any]] = {}
    for candidate in tqdm(selected, desc="Writing samples", unit="sample"):
        row = _write_sample(
            candidate,
            output_dir,
            source_cache,
            args.num_bins,
            args.window_size,
        )
        if row is None:
            skipped["missing_frames"] += 1
        else:
            rows_by_split[candidate.split].append(row)

    metadata: dict[str, Any] = {
        "raw_data_paths": args.raw_data_path,
        "output_dir": args.output_dir,
        "value_checkpoint": args.value_checkpoint,
        "num_bins": args.num_bins,
        "window_size": args.window_size,
        "stride": args.stride,
        "progress_deadband": args.progress_deadband,
        "progress_gap_steps": args.progress_gap_steps,
        "temporal_smoothing_window": args.temporal_smoothing_window,
        "num_episodes": len(pkl_files),
        "episode_counts": dict(episode_counts),
        "skipped": dict(skipped),
        "selection": selection,
        "splits": {},
    }
    for split in ("train", "eval"):
        rows = rows_by_split[split]
        rng.shuffle(rows)
        manifest = write_manifest(rows, output_dir, split)
        metadata["splits"][split] = {
            "manifest": manifest,
            "num_samples": len(rows),
            "sample_type_counts": dict(
                Counter(row["segment_metadata"]["sample_type"] for row in rows)
            ),
            "answer_counts": dict(Counter(row["answer"] for row in rows)),
            "outcome_counts": dict(
                Counter(
                    "success" if row["segment_metadata"]["success"] else "failure"
                    for row in rows
                )
            ),
            "source_run_counts": dict(Counter(row["source_run"] for row in rows)),
            "progress_gap_counts": dict(
                Counter(
                    str(row["segment_metadata"]["progress_gap_steps"])
                    for row in rows
                    if row["segment_metadata"]["sample_type"] == "progress"
                )
            ),
        }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    logger.info("%s", json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def _add_potential_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-data-path",
        required=True,
        action="append",
        help="Collected-data directory; repeat to merge independent collection runs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--progress-deadband", type=float, default=0.03)
    parser.add_argument(
        "--progress-gap-steps",
        type=int,
        nargs="+",
        default=None,
        help="One or more temporal gaps, for example: 10 20 40.",
    )
    parser.add_argument("--temporal-smoothing-window", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--only-split", choices=("train", "eval"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--potential-samples-train", type=int, default=16000)
    parser.add_argument("--potential-samples-eval", type=int, default=2400)
    parser.add_argument("--progress-samples-train", type=int, default=7200)
    parser.add_argument("--progress-samples-eval", type=int, default=1080)
    parser.add_argument("--task-description", type=str, default=None)
    parser.add_argument("--device", default="cuda")


@torch.no_grad()
def _encode_feature_batch(
    model: Any,
    prompts: list[str],
    videos: list[list[Any]],
    video_fps: float,
) -> torch.Tensor:
    """Pool Qwen features using the same processor path as VLM Trend SFT."""
    from rlinf.data.datasets.vlm import VLMTrendRewardSFTDataset

    _, inputs, _ = VLMTrendRewardSFTDataset.process_inputs(
        processor=model._processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[[prompt] for prompt in prompts],
        videos=videos,
        answer_text=None,
        video_fps=video_fps,
    )
    inputs = {
        key: value.to(model._model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    return model.extract_prompt_features(inputs).cpu()


def _feature_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [
        row
        for row in rows
        if row["segment_metadata"]["sample_type"] == args.sample_type
        and sample_source_hash(row) % args.world_size == args.rank
    ]
    rows.sort(
        key=lambda row: (
            row["source_episode_path"],
            row["segment_metadata"]["end_step"],
        )
    )
    return rows if args.max_samples is None else rows[: args.max_samples]


def _extract_features(
    model: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from rlinf.data.datasets.vlm import VLMTrendRewardSFTDataset

    feature_batches = []
    targets = []
    labels = []
    successes = []
    source_paths = []
    end_steps = []
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc=f"Extracting {args.sample_type} features",
    ):
        batch = rows[start : start + args.batch_size]
        prompts = []
        videos = []
        for index, row in enumerate(batch, start=start):
            _, _, source_videos, _ = VLMTrendRewardSFTDataset._parse_raw_record(
                row, index, None
            )
            window_size = int(row["segment_metadata"]["window_size"])
            if window_size != args.history_size:
                raise ValueError(
                    "--history-size must match the preprocessed manifest window size"
                )
            prompt = potential_prompt(row["task"], args.history_size, args.num_bins)
            if args.sample_type == "potential":
                prompts.append(prompt)
                videos.append(source_videos)
                targets.append(float(row["supervision"]["teacher_value"]))
                successes.append(bool(row["segment_metadata"]["success"]))
                source_paths.append(row["source_episode_path"])
                end_steps.append(int(row["segment_metadata"]["end_step"]))
            else:
                main, extra = source_videos
                if len(main) != 2 * window_size or len(extra) != 2 * window_size:
                    raise ValueError("Progress samples must contain two full windows")
                prompts.extend([prompt, prompt])
                videos.extend(
                    [
                        [main[:window_size], extra[:window_size]],
                        [main[window_size:], extra[window_size:]],
                    ]
                )
                targets.append(float(row["supervision"]["teacher_delta"]))
                labels.append(row["answer"])
        encoded = _encode_feature_batch(model, prompts, videos, args.video_fps)
        if args.sample_type == "progress":
            encoded = encoded.reshape(len(batch), 2, -1)
        feature_batches.append(encoded)

    payload: dict[str, Any] = {
        "features": torch.cat(feature_batches).to(torch.float16),
    }
    if args.sample_type == "potential":
        payload.update(
            targets=torch.tensor(targets, dtype=torch.float32),
            successes=torch.tensor(successes, dtype=torch.bool),
            source_paths=source_paths,
            end_steps=torch.tensor(end_steps, dtype=torch.int32),
        )
    else:
        payload.update(
            teacher_deltas=torch.tensor(targets, dtype=torch.float32),
            labels=labels,
        )
    return payload


def _run_features(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.reward.vlm_reward_model import VLMRewardModel

    rows = _feature_rows(args)
    if not rows:
        raise ValueError("No manifest rows selected for this feature shard")
    cfg = OmegaConf.create(
        {
            "model_path": args.model_path,
            "lora_path": args.checkpoint,
            "precision": "bf16",
            "inference_mode": "generate",
            "subprocessor_kwargs": {"video_processor": {"do_sample_frames": True}},
            "input_builder_name": "vlm_trend_reward_input_builder",
            "input_builder_params": {
                "history_buffer_names": ["history_window"],
                "prompt_template": (
                    "You are currently performing the task: {task}. "
                    "Given the current state, predict the success potential."
                ),
            },
            "reward_parser_name": "base_reward_parser",
            "reward_parser_params": {},
        }
    )
    model = VLMRewardModel(cfg)
    model._model.to(args.device).eval()
    payload = _extract_features(model, rows, args)
    payload["metadata"] = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "sample_type": args.sample_type,
        "rank": args.rank,
        "world_size": args.world_size,
        "num_samples": len(rows),
        "history_size": args.history_size,
        "num_bins": args.num_bins,
        "video_fps": args.video_fps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info("%s", json.dumps(payload["metadata"], indent=2))


def _add_features_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-type", choices=("potential", "progress"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)


_MODE_ADDERS = {
    "terminal_success": _add_terminal_success_args,
    "potential": _add_potential_args,
    "features": _add_features_args,
}
_MODE_RUNNERS = {
    "terminal_success": _run_terminal_success,
    "potential": run_potential,
    "features": _run_features,
}


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(
        description="Preprocess Trend, Success, or Potential reward data.",
        add_help=False,
    )
    pre.add_argument(
        "--mode",
        choices=("trend", *_MODE_ADDERS),
        default="trend",
        help="Label pipeline (default: trend).",
    )
    pre.add_argument("-h", "--help", action="store_true")
    known, remaining = pre.parse_known_args(argv)
    if known.help:
        remaining = ["--help"]
    if known.mode == "trend":
        run_trend(remaining)
        return
    parser = argparse.ArgumentParser(description=f"preprocess --mode {known.mode}")
    _MODE_ADDERS[known.mode](parser)
    _MODE_RUNNERS[known.mode](parser.parse_args(remaining))


if __name__ == "__main__":
    main()
