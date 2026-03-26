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

"""Preprocess raw reward data into split train/val .pt files.

Example:
    python examples/reward/preprocess_reward_dataset.py \
        --raw-data-path logs/xxx/collected_data \
        --output-dir logs/xxx/processed_reward_data
"""

import argparse
import json
import os
import pickle
import random
from glob import glob
from typing import Optional

import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.utils.logging import get_logger

logger = get_logger()


def _compute_sample_indices(
    n: int, num_samples_per_episode: int, keep_last_frame: bool
) -> list[int]:
    """Compute sampled frame indices while preserving current behavior."""
    if keep_last_frame:
        if num_samples_per_episode == 1:
            return [n - 1]

        # Reserve one slot for the final frame, and evenly sample
        # the remaining slots from earlier frames.
        k = num_samples_per_episode - 1
        non_last_n = n - 1
        if k >= non_last_n:
            non_last_indices = list(range(non_last_n))
        elif k == 1:
            non_last_indices = [0]
        else:
            non_last_indices = [int(i * (non_last_n - 1) / (k - 1)) for i in range(k)]
        return sorted(set(non_last_indices + [n - 1]))

    # Evenly spaced sampling.
    if num_samples_per_episode == 1:
        return [n - 1]
    return sorted(
        {
            int(i * (n - 1) / (num_samples_per_episode - 1))
            for i in range(num_samples_per_episode)
        }
    )


def load_episodes_with_labels(
    data_path: str, num_samples_per_episode: int = 5, keep_last_frame: bool = True
) -> list[dict]:
    """Load episodes with per-frame labels from collected data.

    Args:
        data_path: Path to directory containing .pkl episode files.
        num_samples_per_episode: Number of frames to sample per episode.
            Samples are evenly spaced (start, middle, end, etc).
            Set to 0 or negative to use all frames.
        keep_last_frame: Whether to always keep the last valid frame of each
            episode in sampled results.

    Returns:
        List of episode dicts, each containing 'images' and 'labels' lists.
    """
    pkl_files = sorted(glob(os.path.join(data_path, "*.pkl")))
    logger.info(f"Found {len(pkl_files)} episode files in {data_path}")

    episodes = []

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)

            observations = episode.get("observations", [])
            infos = episode.get("infos", [])

            if not observations or not infos:
                continue

            # First collect all valid frames (exclude last frame - it's from next episode after reset)
            all_frames = []
            num_frames = len(infos)
            for idx in range(num_frames):
                obs = observations[idx]
                info = infos[idx]

                success_flag = info.get(
                    "success", info.get("episode", {}).get("success_once", False)
                )

                img = obs.get("main_images")
                if img is None:
                    continue

                all_frames.append((img, 1 if success_flag else 0))

            if not all_frames:
                continue

            # Sample frames based on num_samples_per_episode
            n = len(all_frames)
            if num_samples_per_episode > 0 and n > num_samples_per_episode:
                indices = _compute_sample_indices(
                    n=n,
                    num_samples_per_episode=num_samples_per_episode,
                    keep_last_frame=keep_last_frame,
                )
                sampled = [all_frames[i] for i in indices]
            else:
                # Use all frames
                sampled = all_frames

            ep_images = [f[0] for f in sampled]
            ep_labels = [f[1] for f in sampled]

            if ep_images:
                episodes.append({"images": ep_images, "labels": ep_labels})

        except Exception as e:
            logger.warning(f"Failed to load {pkl_path}: {e}")

    total_frames = sum(len(ep["images"]) for ep in episodes)
    total_success = sum(sum(ep["labels"]) for ep in episodes)
    sample_info = (
        f"{num_samples_per_episode} per ep" if num_samples_per_episode > 0 else "all"
    )
    logger.info(
        f"Loaded {len(episodes)} episodes, {total_frames} frames ({sample_info}): {total_success} success, {total_frames - total_success} fail"
    )
    return episodes


def balance_and_split_by_episode(
    episodes: list[dict],
    val_split: float = 0.2,
    fail_success_ratio: float = 2.0,
    random_seed: Optional[int] = None,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor], list[int]]:
    """Split by EPISODE and sample with configurable fail:success ratio.

    Strategy:
    1. Split episodes into train/val sets (entire episodes)
    2. Use ALL frames from each episode (no sparse sampling)
    3. Sample fail frames to achieve fail:success ratio (e.g., 2:1)

    This prevents data leakage because frames from the same episode
    won't appear in both train and val sets.

    Args:
        episodes: List of episode dicts with 'images' and 'labels' keys.
        val_split: Fraction of episodes for validation.
        fail_success_ratio: Ratio of fail:success frames (e.g., 2.0 means 2:1).
        random_seed: Optional random seed for deterministic split/sampling.

        Returns:
        Tuple of (train_images, train_labels, val_images, val_labels).
    """
    if not episodes:
        logger.error("No episodes provided!")
        return [], [], [], []

    rng = random.Random(random_seed) if random_seed is not None else random

    # Shuffle and split EPISODES
    episodes_copy = list(episodes)
    rng.shuffle(episodes_copy)
    val_ep_count = max(1, int(len(episodes_copy) * val_split))
    val_episodes = episodes_copy[:val_ep_count]
    train_episodes = episodes_copy[val_ep_count:]

    logger.info(
        f"Episode split: {len(train_episodes)} train eps, {len(val_episodes)} val eps"
    )

    def extract_and_sample(ep_list: list[dict], ratio: float) -> tuple[list, list]:
        """Extract frames and sample to achieve fail:success ratio."""
        success_imgs = []
        fail_imgs = []
        for ep in ep_list:
            for img, lbl in zip(ep["images"], ep["labels"]):
                if lbl == 1:
                    success_imgs.append(img)
                else:
                    fail_imgs.append(img)

        logger.info(f"  Raw: {len(success_imgs)} success, {len(fail_imgs)} fail")

        if len(success_imgs) == 0:
            logger.warning("  No success frames!")
            return [], []

        # Sample fail frames to achieve ratio
        target_fail = int(len(success_imgs) * ratio)
        rng.shuffle(fail_imgs)
        fail_imgs = fail_imgs[:target_fail]

        logger.info(
            f"  After {ratio}:1 ratio: {len(success_imgs)} success, {len(fail_imgs)} fail"
        )

        # Combine and shuffle
        images = success_imgs + fail_imgs
        labels = [1] * len(success_imgs) + [0] * len(fail_imgs)

        pairs = list(zip(images, labels))
        rng.shuffle(pairs)
        if pairs:
            images, labels = zip(*pairs)
            return list(images), list(labels)
        return [], []

    logger.info("Processing train set:")
    train_images, train_labels = extract_and_sample(train_episodes, fail_success_ratio)
    logger.info("Processing val set:")
    val_images, val_labels = extract_and_sample(val_episodes, fail_success_ratio)

    logger.info(
        f"Episode-based split complete - Train: {len(train_images)} frames "
        f"({sum(train_labels) if train_labels else 0} success), "
        f"Val: {len(val_images)} frames ({sum(val_labels) if val_labels else 0} success)"
    )

    return train_images, train_labels, val_images, val_labels


def preprocess_and_save_reward_datasets(
    raw_data_path: str,
    train_output_path: str,
    val_output_path: str,
    num_samples_per_episode: int = 5,
    keep_last_frame: bool = True,
    val_split: float = 0.2,
    fail_success_ratio: float = 2.0,
    random_seed: Optional[int] = None,
) -> dict:
    """Build train/val datasets from raw data and save split files."""
    episodes = load_episodes_with_labels(
        raw_data_path,
        num_samples_per_episode=num_samples_per_episode,
        keep_last_frame=keep_last_frame,
    )
    if len(episodes) == 0:
        raise ValueError(f"No episodes loaded from raw data path: {raw_data_path}")

    train_images, train_labels, val_images, val_labels = balance_and_split_by_episode(
        episodes=episodes,
        val_split=val_split,
        fail_success_ratio=fail_success_ratio,
        random_seed=random_seed,
    )
    metadata = {
        "raw_data_path": raw_data_path,
        "num_samples_per_episode": num_samples_per_episode,
        "keep_last_frame": keep_last_frame,
        "val_split": val_split,
        "fail_success_ratio": fail_success_ratio,
        "random_seed": random_seed,
        "num_train_samples": len(train_images),
        "num_val_samples": len(val_images),
    }

    def _save_split(
        images: list, labels: list[int], output_path: str, split_name: str
    ) -> None:
        RewardDatasetPayload(images=images, labels=labels, metadata=metadata).save(
            output_path
        )
        logger.info(
            f"Saved processed reward {split_name} split to {output_path}: {len(images)}"
        )

    _save_split(train_images, train_labels, train_output_path, "train")
    _save_split(val_images, val_labels, val_output_path, "val")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess reward dataset from raw episode .pkl files."
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
        default="logs/processed_reward_data",
        help="Output directory for processed split files (train.pt / val.pt).",
    )
    parser.add_argument(
        "--train-output-path",
        type=str,
        default=None,
        help="Optional explicit output path for train split file.",
    )
    parser.add_argument(
        "--val-output-path",
        type=str,
        default=None,
        help="Optional explicit output path for val split file.",
    )
    parser.add_argument(
        "--num-samples-per-episode",
        type=int,
        default=0,
        help="Number of sampled frames per episode. Use 0 for all frames.",
    )
    parser.add_argument(
        "--keep-last-frame",
        dest="keep_last_frame",
        action="store_true",
        default=True,
        help="Always include each episode's last valid frame when sampling.",
    )
    parser.add_argument(
        "--no-keep-last-frame",
        dest="keep_last_frame",
        action="store_false",
        help="Allow sampling to exclude each episode's last valid frame.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of episodes for validation.",
    )
    parser.add_argument(
        "--fail-success-ratio",
        type=float,
        default=2.0,
        help="Sampling ratio of fail:success frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic episode split and sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_output_path = args.train_output_path or os.path.join(
        args.output_dir, "train.pt"
    )
    val_output_path = args.val_output_path or os.path.join(args.output_dir, "val.pt")

    metadata = preprocess_and_save_reward_datasets(
        raw_data_path=args.raw_data_path,
        train_output_path=train_output_path,
        val_output_path=val_output_path,
        num_samples_per_episode=args.num_samples_per_episode,
        keep_last_frame=args.keep_last_frame,
        val_split=args.val_split,
        fail_success_ratio=args.fail_success_ratio,
        random_seed=args.seed,
    )

    print("=" * 80)
    print("Reward dataset preprocessing complete")
    print(f"Train split: {train_output_path} ({metadata['num_train_samples']} samples)")
    print(f"Val split:   {val_output_path} ({metadata['num_val_samples']} samples)")
    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
