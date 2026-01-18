# Copyright 2025 The RLinf Authors.
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

"""Reward Workers for RLinf.

This module provides:
- RewardWorker: For inference/computing rewards during RL training
- FSDPRewardWorker: For training reward models with FSDP (like FSDPSftWorker)
"""

import logging
import os
import pickle
import random
from glob import glob
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models.embodiment.reward import ResNetRewardModel
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import clear_memory

logger = logging.getLogger(__name__)


class RewardWorker(Worker):
    """Reward Worker for inference during RL training."""

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            raise NotImplementedError("Reward model is not implemented yet.")
        else:
            self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            with self.worker_timer():
                if rollout_result.rewards is None:
                    if self.cfg.reward.use_reward_model:
                        with self.device_lock:
                            batch = rollout_result.to_actor_batch(
                                self.cfg.data.max_prompt_length,
                                self.cfg.actor.model.encoder_seq_length,
                                self.tokenizer.eos_token_id,
                            )
                            rollout_result.rewards = (
                                self.compute_batch_rewards_with_model(batch)
                            )
                    else:
                        rollout_result.rewards = self._compute_rule_based_rewards(
                            rollout_result
                        )

            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        # Decode only the generated tokens; response_ids are already the post-prompt tokens
        texts = rollout_result.response_texts
        if texts is None:
            texts = self.tokenizer.batch_decode(
                rollout_result.response_ids, skip_special_tokens=True
            )

        kwargs = {}
        if getattr(self.cfg.reward, "use_prompt", False):
            prompts = rollout_result.prompt_texts
            if prompts is None:
                prompts = self.tokenizer.batch_decode(
                    rollout_result.prompt_ids, skip_special_tokens=True
                )
            kwargs["prompts"] = prompts
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def compute_batch_rewards_with_model(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("Reward model is not implemented yet.")


# =============================================================================
# Dataset for Binary Classification Reward Model Training
# =============================================================================


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


# =============================================================================
# FSDP Reward Worker for Training (like FSDPSftWorker)
# =============================================================================


class FSDPRewardWorker(FSDPModelManager, Worker):
    """FSDP-based worker for reward model training.

    This follows the same pattern as FSDPSftWorker:
    - Inherits FSDPModelManager for optimizer, scheduler, FSDP setup
    - Implements model_provider_func() to return ResNetRewardModel
    - Implements build_dataloader() for data loading
    - Implements run_training() for training loop
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.val_loader = self.build_dataloader()
        self.data_iter = iter(self.data_loader) if self.data_loader else None

        # Training step counter for validation interval
        self._training_step = 0
        self._val_interval = cfg.runner.get("val_check_interval", 50)

    def init_worker(self):
        """Initialize model and optimizer using FSDP wrapping.

        We manually wrap with FSDP since base class's wrap_model needs SupportedModel enum.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
        from torch.distributed.fsdp import (
            MixedPrecision,
            ShardingStrategy,
        )

        from rlinf.config import torch_dtype_from_precision

        # Create base model
        module = self.model_provider_func()

        # Log gradient checkpointing status
        if self._cfg.fsdp_config.get("gradient_checkpointing", False):
            logger.info("[FSDP] Gradient checkpointing enabled")
        else:
            logger.info("[FSDP] Gradient checkpointing is disabled")

        # Setup mixed precision
        mixed_precision_config = self._cfg.fsdp_config.mixed_precision
        param_dtype = torch_dtype_from_precision(mixed_precision_config.param_dtype)
        reduce_dtype = torch_dtype_from_precision(mixed_precision_config.reduce_dtype)
        buffer_dtype = torch_dtype_from_precision(mixed_precision_config.buffer_dtype)
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        # Get sharding strategy
        sharding_strategy_name = self._cfg.fsdp_config.sharding_strategy.upper()
        sharding_strategy = getattr(
            ShardingStrategy, sharding_strategy_name, ShardingStrategy.NO_SHARD
        )

        # Wrap model with FSDP
        self.model = FSDP(
            module=module,
            device_id=self.device,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self._device_mesh,
            use_orig_params=self._cfg.fsdp_config.use_orig_params,
        )

        # Mark as not using LoRA
        self.is_lora = False

        # Use base class methods for optimizer and scheduler
        self.optimizer = self.build_optimizer(
            model=self.model, enable_critic_warmup=False
        )
        self.lr_scheduler = self.build_lr_scheduler(optimizer=self.optimizer)
        self.grad_scaler = self.build_grad_scaler(
            self._cfg.fsdp_config.amp.get("use_grad_scaler", False)
        )

        logger.info(
            f"Initialized FSDPRewardWorker with "
            f"{sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def _save_dataset_images(
        self, dataset: RewardBinaryDataset, save_dir: str, split: str
    ) -> None:
        """Save all images from dataset for debugging.

        Args:
            dataset: The dataset to save images from.
            save_dir: Base directory to save images.
            split: 'train' or 'val' to indicate which split.
        """
        from PIL import Image
        from tqdm import tqdm

        success_dir = os.path.join(save_dir, split, "success")
        fail_dir = os.path.join(save_dir, split, "fail")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)

        success_count = 0
        fail_count = 0

        logger.info(f"Saving {len(dataset)} {split} images to {save_dir}...")

        for idx in tqdm(range(len(dataset)), desc=f"Saving {split} images"):
            img_tensor, label = dataset[idx]

            # Convert tensor to numpy image
            # img_tensor is (C, H, W) float [0, 1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img_np)

            is_success = label.item() > 0.5
            if is_success:
                img_pil.save(os.path.join(success_dir, f"{split}_{idx:05d}.png"))
                success_count += 1
            else:
                img_pil.save(os.path.join(fail_dir, f"{split}_{idx:05d}.png"))
                fail_count += 1

        logger.info(f"Saved {split} images: {success_count} success, {fail_count} fail")

    def model_provider_func(self) -> torch.nn.Module:
        """Provide the ResNet reward model."""
        model_cfg = self.cfg.actor.model
        model = ResNetRewardModel(model_cfg)
        return model

    def build_dataloader(self) -> tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Build training and validation dataloaders.

        Uses per-frame 'is_obj_placed' labels from infos instead of episode-level labels.
        Splits by EPISODE to prevent data leakage (adjacent frames are too similar).
        Uses sparse sampling (start/middle/end) to reduce redundancy.
        """
        data_cfg = self.cfg.get("data", {})
        data_path = data_cfg.get("data_path")

        if not data_path or not os.path.exists(data_path):
            logger.warning(f"Data path not found: {data_path}")
            return None, None

        # Load episodes with configurable sampling
        num_samples = data_cfg.get("num_samples_per_episode", 5)
        logger.info(
            f"Loading episodes from {data_path} with {num_samples} samples per episode..."
        )
        episodes = load_episodes_with_labels(data_path, num_samples)

        if len(episodes) == 0:
            logger.warning("No episodes loaded from dataset")
            return None, None

        # Split by EPISODE (prevents data leakage), sample with fail:success ratio
        val_split = data_cfg.get("val_split", 0.2)
        fail_success_ratio = data_cfg.get("fail_success_ratio", 2.0)
        train_dataset, val_dataset = balance_and_split_by_episode(
            episodes, val_split, fail_success_ratio
        )

        if len(train_dataset) == 0:
            logger.warning("Training dataset is empty after balancing")
            return None, None

        # Debug: save training images to verify data pipeline
        debug_save_dir = data_cfg.get("debug_save_dir", None)
        if debug_save_dir and self._rank == 0:
            self._save_dataset_images(train_dataset, debug_save_dir, "train")
            self._save_dataset_images(val_dataset, debug_save_dir, "val")

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=False,
        )

        batch_size = self.cfg.actor.micro_batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        logger.info(
            f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val"
        )

        return train_loader, val_loader

    def run_training(self) -> dict[str, float]:
        """Run one training iteration with gradient accumulation.

        Follows the same pattern as FSDPSftWorker.run_training().
        """
        # Check if data loader is available
        if self.data_iter is None or self.data_loader is None:
            raise RuntimeError(
                "Data loader is not available. Please check that:\n"
                f"  1. Data path exists: {self.cfg.get('data', {}).get('data_path')}\n"
                "  2. Data path is an absolute path (not relative)\n"
                "  3. Data files (*_success.pkl and *_fail.pkl) are present"
            )

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                # Get batch (image, label)
                try:
                    images, labels = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    images, labels = next(self.data_iter)

                # Move to device: images shape is (B, C, H, W), labels shape is (B,)
                images = images.to(self.device)
                labels = labels.to(self.device)

                with self.amp_context:
                    # Forward pass - loss computed inside model
                    outputs = self.model(images, labels)
                    loss = outputs["loss"]

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

                # Accumulate metrics
                append_to_dict(
                    metrics,
                    {
                        "loss": outputs["loss"].item(),
                        "accuracy": outputs["accuracy"].item(),
                        "probabilities_mean": outputs["probabilities"].mean().item(),
                    },
                )

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            # Increment step counter and run validation at interval
            self._training_step += 1
            if self._training_step % self._val_interval == 0:
                val_metrics = self.run_validation()
                train_metrics.update(val_metrics)

            return train_metrics

    def run_validation(self) -> dict[str, float]:
        """Run validation over the entire validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with self.amp_context:
                    outputs = self.model(images, labels)

                append_to_dict(
                    metrics,
                    {
                        "val_loss": outputs["loss"].item(),
                        "val_accuracy": outputs["accuracy"].item(),
                        "val_probabilities_mean": outputs["probabilities"]
                        .mean()
                        .item(),
                    },
                )

        val_metrics = {key: np.mean(value) for key, value in metrics.items()}
        val_metrics = all_reduce_dict(val_metrics, op=torch.distributed.ReduceOp.AVG)

        return val_metrics

    def set_global_step(self, global_step: int) -> None:
        """Set global step (for compatibility with SFTRunner)."""
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)


# =============================================================================
# Image Reward Worker for Inference (with Channel Communication)
# =============================================================================


class ImageRewardWorker(Worker):
    """Image-based Reward Worker for inference during RL training.

    This worker loads a trained ResNetRewardModel checkpoint and computes
    rewards for images received via channel communication.

    Usage:
        - Configure with checkpoint_path pointing to trained model
        - Connect input_channel (receives images) and output_channel (sends rewards)
        - Call compute_rewards() in the training loop
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        # Device setup
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        # Placement
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # Model will be loaded in init_worker
        self.model = None

        # Debug image saving (based on ResNet classification)
        self.debug_save_dir = os.environ.get("DEBUG_IMAGE_SAVE_DIR", None)
        self.debug_success_count = 0
        self.debug_fail_count = 0
        # Use higher threshold to reduce false positives
        # Model: success mean=0.61, fail mean=0.52 (too close!)
        # Threshold 0.6 reduces FP rate significantly
        self.reward_threshold = 0.6

    def init_worker(self):
        """Initialize the reward model from checkpoint."""
        reward_cfg = self.cfg.get("reward", {})
        model_cfg = reward_cfg.get("model", {})

        # Get debug save dir from config or environment variable
        self.debug_save_dir = model_cfg.get("debug_save_dir") or os.environ.get(
            "DEBUG_IMAGE_SAVE_DIR", None
        )

        # Setup debug image saving directories
        if self.debug_save_dir:
            os.makedirs(
                os.path.join(self.debug_save_dir, "resnet_success"), exist_ok=True
            )
            os.makedirs(os.path.join(self.debug_save_dir, "resnet_fail"), exist_ok=True)
        checkpoint_path = model_cfg.get("checkpoint_path")

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified for ImageRewardWorker")

        # Create model (will auto-load checkpoint if checkpoint_path is in model_cfg)
        self.model = ResNetRewardModel(model_cfg)

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards for images received from input channel.

        Expected input format via channel:
            dict with keys:
                - 'images': torch.Tensor of shape (B, C, H, W) or (B, H, W, C)
                - 'episode_ids': optional, for tracking

        Output format via channel:
            dict with keys:
                - 'rewards': torch.Tensor of shape (B,)
                - 'episode_ids': passed through if provided

        Args:
            input_channel: Channel to receive image batches from
            output_channel: Channel to send computed rewards to
        """
        with self.worker_timer():
            # Receive data from channel
            data = input_channel.get()

            # Try multiple image keys: reward_images (from StateWithRGBWrapper),
            # images, or main_images
            images = data.get("reward_images")
            if images is None:
                images = data.get("images")
            if images is None:
                images = data.get("main_images")
            if images is None:
                logger.warning(
                    "No images in input data (tried: reward_images, images, main_images)"
                )
                output_channel.put({"rewards": None}, async_op=True)
                return

            # Convert to tensor if needed (don't normalize here, model will do it)
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)

            images = images.to(self.device)

            # Ensure model is in eval mode (BatchNorm layers update running stats in train mode!)
            self.model.eval()

            # Let model.preprocess_images() handle all preprocessing (permute, normalize, resize)
            # This ensures consistency with training validation
            with torch.no_grad():
                outputs = self.model(images)
                probs = outputs["probabilities"]

                # Apply threshold - use 0.5 for best accuracy (87% validated)
                # probs > 0.5 means success, otherwise fail
                rewards = torch.where(
                    probs > self.reward_threshold,
                    probs,  # High confidence success: use probability as reward
                    torch.zeros_like(probs),  # Below threshold: zero reward
                )

            # Save debug images based on ResNet classification
            # Note: images are already preprocessed by model.forward() -> preprocess_images()
            # But we need to save the original images (before preprocessing) for visualization
            if self.debug_save_dir:
                # Get original images before preprocessing (from channel data)
                # Use explicit None check instead of 'or' (Tensor doesn't support bool conversion)
                original_images = data.get("main_images")
                if original_images is None:
                    original_images = data.get("images")
                if original_images is None:
                    original_images = data.get("reward_images")
                if original_images is not None:
                    if isinstance(original_images, np.ndarray):
                        original_images = torch.from_numpy(original_images)
                    original_images = original_images.to(self.device)
                    # Convert to CHW format if needed, and to [0, 1] float
                    if original_images.dim() == 4 and original_images.shape[-1] in [
                        1,
                        3,
                        4,
                    ]:
                        original_images = original_images.permute(0, 3, 1, 2)
                    if original_images.dtype == torch.uint8:
                        original_images = original_images.float() / 255.0
                    self._save_debug_images(original_images, rewards, probs)

            # Prepare output
            output_data = {
                "rewards": rewards.cpu(),
            }

            # Pass through episode_ids if provided
            if "episode_ids" in data:
                output_data["episode_ids"] = data["episode_ids"]

            output_channel.put(output_data, async_op=True)

    def compute_reward_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Compute rewards for a batch of images directly (without channel).

        Args:
            images: Tensor of shape (B, C, H, W) or (B, H, W, C)

        Returns:
            rewards: Tensor of shape (B,)
        """
        images = self._preprocess_images(images)
        images = images.to(self.device)

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():
            rewards = self.model.compute_reward({"images": images})

        return rewards

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images to expected format (B, C, H, W) in [0, 1]."""
        # Convert numpy to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # Handle (B, H, W, C) -> (B, C, H, W)
        if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)

        # Ensure float in [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()

        return images

    def _save_debug_images(
        self, images: torch.Tensor, rewards: torch.Tensor, probs: torch.Tensor = None
    ):
        """Save debug images based on ResNet classification results.

        Args:
            images: Preprocessed images (B, C, H, W) in [0, 1]
            rewards: Reward values from ResNet model (B,)
            probs: Raw probability outputs from ResNet (B,), used for filename
        """
        from PIL import Image

        # Convert to numpy for saving
        images_np = images.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        probs_np = probs.cpu().numpy() if probs is not None else rewards_np

        for idx in range(len(rewards_np)):
            reward = rewards_np[idx]
            prob = probs_np[idx]
            img = images_np[idx]

            # Convert from (C, H, W) to (H, W, C)
            if img.shape[0] in [1, 3, 4]:
                img = img.transpose(1, 2, 0)

            # Convert from [0, 1] to [0, 255]
            img = (img * 255).clip(0, 255).astype(np.uint8)

            # Handle grayscale
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            # Classify based on reward threshold
            is_success = reward > self.reward_threshold

            # Save both success and fail images
            # Filename format: {id}_{prob:.4f}.png for easy comparison
            if is_success:
                self.debug_success_count += 1
                save_dir = os.path.join(self.debug_save_dir, "resnet_success")
                filename = f"{self.debug_success_count:06d}_prob{prob:.4f}.png"
            else:
                self.debug_fail_count += 1
                save_dir = os.path.join(self.debug_save_dir, "resnet_fail")
                filename = f"{self.debug_fail_count:06d}_prob{prob:.4f}.png"

            os.makedirs(save_dir, exist_ok=True)

            # Save image
            try:
                pil_img = Image.fromarray(img)
                pil_img.save(os.path.join(save_dir, filename))
            except Exception:
                pass

    def run_inference_loop(self, input_channel: Channel, output_channel: Channel):
        """Run continuous inference loop (for use in RL training).

        This method runs until input_channel signals completion (returns None).

        Args:
            input_channel: Channel to receive image batches from
            output_channel: Channel to send computed rewards to
        """
        logger.info("Starting ImageRewardWorker inference loop")

        while True:
            try:
                data = input_channel.get(timeout=1.0)
                if data is None:
                    logger.info("Received stop signal, ending inference loop")
                    break

                images = data.get("images")
                if images is None:
                    continue

                images = self._preprocess_images(images)
                images = images.to(self.device)

                # Ensure model is in eval mode
                self.model.eval()

                with torch.no_grad():
                    rewards = self.model.compute_reward({"images": images})

                output_data = {"rewards": rewards.cpu()}
                if "episode_ids" in data:
                    output_data["episode_ids"] = data["episode_ids"]

                output_channel.put(output_data, async_op=True)

            except Exception as e:
                logger.warning(f"Error in inference loop: {e}")
                continue

        logger.info("ImageRewardWorker inference loop ended")

    def set_global_step(self, global_step: int) -> None:
        """Set global step (for compatibility with EmbodiedRunner)."""
        self.global_step = global_step

    def save_checkpoint(self, save_path: str, global_step: int) -> None:
        """Save reward model checkpoint.

        Args:
            save_path: Directory to save the checkpoint
            global_step: Current global step for naming
        """
        if self.model is None:
            return

        os.makedirs(save_path, exist_ok=True)
        checkpoint_file = os.path.join(save_path, f"reward_model_step_{global_step}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "global_step": global_step,
            },
            checkpoint_file,
        )
