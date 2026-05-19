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

"""Preprocess Qwen trend reward data into split train/eval pkl datasets.

Example:
    python examples/reward/preprocess_qwentrend_reward_dataset.py \
        --raw-data-path logs/xxx/collected_data \
        --output-dir logs/xxx/processed_qwentrend_reward_data

The exported JSONL points to per-sample pkl files. QwenTrendProgressSFTDataset
loads the two 5-frame video arrays directly from those pkl files, avoiding the
slow small-mp4 export path.
"""

import argparse
import json
import os
import pickle
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from rlinf.utils.logging import get_logger

logger = get_logger()


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
        f"Please judge whether the operation shown in these two {window_size}-frame "
        "videos, which capture the same time window from two different views, makes "
        "the task better, worse, or unclear. Answer with exactly one word: "
        "positive, negative, or unclear."
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
    """Build train/eval Qwen trend reward datasets from raw data."""
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
            f"Saved processed Qwen trend reward {split_name} split to "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Qwen trend reward dataset from raw episode .pkl files."
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
        default="logs/processed_qwentrend_reward_data",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    print("Qwen trend reward dataset preprocessing complete")
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


if __name__ == "__main__":
    main()
