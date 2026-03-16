import os
import torch
import pickle
import random
from glob import glob
from typing import Optional
import numpy as np
from torch.utils.data import Dataset

from rlinf.utils.logging import get_logger

logger = get_logger()

class RewardBinaryDataset(Dataset):
    """Dataset for binary classification reward model training.

    Uses per-frame 'is_obj_placed' field from infos to determine success/fail labels.
    This is more accurate than using episode-level labels from filenames.
    """

    def __init__(
        self,
        images: list[torch.Tensor],
        labels: list[int],
    ):
        """Initialize dataset with pre-loaded images and labels.

        Args:
            images: List of image tensors (C, H, W).
            labels: List of binary labels (0=fail, 1=success).
        """
        assert len(images) == len(labels), "Images and labels must have same length"
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get (image, label) pair.

        Returns:
            Tuple of (image tensor (C, H, W), label (0 or 1))
        """
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def load_episodes_with_labels(
    data_path: str, num_samples_per_episode: int = 5
) -> list[dict]:
    """Load episodes with per-frame labels from collected data.

    Uses 'is_obj_placed' field from infos to determine success/fail per frame.
    Returns data organized by episode to enable episode-level train/val splitting.

    Args:
        data_path: Path to directory containing .pkl episode files.
        num_samples_per_episode: Number of frames to sample per episode.
            Samples are evenly spaced (start, middle, end, etc).
            Set to 0 or negative to use all frames.

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
            num_frames = min(len(observations), len(infos))
            # Skip the last frame as it's the reset state
            for idx in range(num_frames - 1):
                obs = observations[idx]
                info = infos[idx]

                success_flag = info.get("is_obj_placed", None)
                if success_flag is None:
                    continue

                try:
                    if hasattr(success_flag, "item"):
                        is_success = bool(success_flag.item())
                    else:
                        is_success = bool(success_flag)
                except Exception:
                    continue

                img = _extract_image(obs)
                if img is None:
                    continue

                all_frames.append((img, 1 if is_success else 0))

            if not all_frames:
                continue

            # Sample frames based on num_samples_per_episode
            n = len(all_frames)
            if num_samples_per_episode > 0 and n > num_samples_per_episode:
                # Evenly spaced sampling
                indices = [
                    int(i * (n - 1) / (num_samples_per_episode - 1))
                    for i in range(num_samples_per_episode)
                ]
                indices = sorted(set(indices))  # Remove duplicates
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
            continue

    total_frames = sum(len(ep["images"]) for ep in episodes)
    total_success = sum(sum(ep["labels"]) for ep in episodes)
    sample_info = (
        f"{num_samples_per_episode} per ep" if num_samples_per_episode > 0 else "all"
    )
    logger.info(
        f"Loaded {len(episodes)} episodes, {total_frames} frames ({sample_info}): {total_success} success, {total_frames - total_success} fail"
    )
    return episodes


def _extract_image(obs: dict) -> Optional[torch.Tensor]:
    """Extract and preprocess image from observation dict.

    Args:
        obs: Observation dictionary with 'main_images' or 'images' key.

    Returns:
        Image tensor (C, H, W) in float32 [0, 1], or None if extraction fails.
    """
    img = obs.get("main_images")
    if img is None:
        img = obs.get("images")

    if img is None:
        return None

    # Convert to tensor if needed
    if isinstance(img, torch.Tensor):
        if img.numel() == 0:
            return None
    elif isinstance(img, np.ndarray):
        if img.size == 0:
            return None
        img = torch.from_numpy(img.copy())
    else:
        return None

    # Ensure tensor is on CPU
    if img.is_cuda:
        img = img.cpu()

    # Handle different formats
    if img.dim() == 4:
        # (1, H, W, C) or (1, C, H, W) -> (C, H, W)
        img = img.squeeze(0)

    if img.dim() == 3:
        # (H, W, C) -> (C, H, W)
        last_dim = img.shape[-1]
        if isinstance(last_dim, int) and last_dim in [1, 3, 4]:
            img = img.permute(2, 0, 1)

    # Ensure float32 in [0, 1]
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    elif img.dtype != torch.float32:
        img = img.float()

    return img


def balance_and_split_by_episode(
    episodes: list[dict],
    val_split: float = 0.2,
    fail_success_ratio: float = 2.0,
) -> tuple[RewardBinaryDataset, RewardBinaryDataset]:
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

        Returns:
        Tuple of (train_dataset, val_dataset).
    """
    if not episodes:
        logger.error("No episodes provided!")
        return RewardBinaryDataset([], []), RewardBinaryDataset([], [])

    # Shuffle and split EPISODES
    random.shuffle(episodes)
    val_ep_count = max(1, int(len(episodes) * val_split))
    val_episodes = episodes[:val_ep_count]
    train_episodes = episodes[val_ep_count:]

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
        random.shuffle(fail_imgs)
        fail_imgs = fail_imgs[:target_fail]

        logger.info(
            f"  After {ratio}:1 ratio: {len(success_imgs)} success, {len(fail_imgs)} fail"
        )

        # Combine and shuffle
        images = success_imgs + fail_imgs
        labels = [1] * len(success_imgs) + [0] * len(fail_imgs)

        pairs = list(zip(images, labels))
        random.shuffle(pairs)
        if pairs:
            images, labels = zip(*pairs)
            return list(images), list(labels)
        return [], []

    logger.info("Processing train set:")
    train_images, train_labels = extract_and_sample(train_episodes, fail_success_ratio)
    logger.info("Processing val set:")
    val_images, val_labels = extract_and_sample(val_episodes, fail_success_ratio)

    train_dataset = RewardBinaryDataset(train_images, train_labels)
    val_dataset = RewardBinaryDataset(val_images, val_labels)

    logger.info(
        f"Episode-based split complete - Train: {len(train_dataset)} frames "
        f"({sum(train_labels) if train_labels else 0} success), "
        f"Val: {len(val_dataset)} frames ({sum(val_labels) if val_labels else 0} success)"
    )

    return train_dataset, val_dataset