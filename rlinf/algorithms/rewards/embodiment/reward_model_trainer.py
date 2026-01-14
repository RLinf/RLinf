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

"""Reward model trainer for training ResNet-based reward models.

This module provides training utilities for reward models, integrated with
the RLinf framework. It can be used standalone or as part of the RL pipeline.
"""

import logging
import os
import pickle
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RewardDataset(Dataset):
    """Dataset for reward model training.

    Supports data formats:
    1. PNG images in success/ and failure/ subdirectories
    2. Episode pkl files in success/ and failure/ subdirectories
       Each pkl contains frames with obs["main_images"]
    """

    def __init__(
        self,
        data_path: str,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        max_success: int = 5000,
        max_failure: int = 5000,
        use_last_frame: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to data directory containing success/ and failure/ subdirs.
            image_size: Target image size (H, W).
            augment: Whether to apply data augmentation.
            max_success: Maximum number of success samples.
            max_failure: Maximum number of failure samples.
            use_last_frame: If True, only use last frame of each episode.
                           If False, use all frames with per-frame labels.
        """
        self.image_size = image_size
        self.augment = augment
        self.max_success = max_success
        self.max_failure = max_failure
        self.use_last_frame = use_last_frame
        self.samples: list[tuple[Any, int]] = []  # (image_data or path, label)
        self._is_png_mode = False

        self._load_data(data_path)

    def _load_data(self, data_path: str) -> None:
        """Load data from path."""
        if not os.path.isdir(data_path):
            raise ValueError(f"Invalid data path: {data_path}")

        success_dir = os.path.join(data_path, "success")
        failure_dir = os.path.join(data_path, "failure")

        success_samples = []
        failure_samples = []

        # Load success samples
        if os.path.isdir(success_dir):
            success_samples = self._load_from_dir(success_dir, label=1)
            logger.info(f"Found {len(success_samples)} success samples")

        # Load failure samples
        if os.path.isdir(failure_dir):
            failure_samples = self._load_from_dir(failure_dir, label=0)
            logger.info(f"Found {len(failure_samples)} failure samples")

        # Limit to max samples
        import random
        if len(success_samples) > self.max_success:
            success_samples = random.sample(success_samples, self.max_success)
            logger.info(f"Limited success samples to {self.max_success}")
        if len(failure_samples) > self.max_failure:
            failure_samples = random.sample(failure_samples, self.max_failure)
            logger.info(f"Limited failure samples to {self.max_failure}")

        self.samples = success_samples + failure_samples
        random.shuffle(self.samples)

        logger.info(
            f"Loaded {len(self.samples)} samples "
            f"({len(success_samples)} success, {len(failure_samples)} failure)"
        )

    def _load_from_dir(self, dir_path: str, label: int) -> list[tuple[Any, int]]:
        """Load samples from a directory (PNG or pkl files)."""
        from glob import glob
        import numpy as np

        samples = []

        # Check for PNG files
        png_files = glob(os.path.join(dir_path, "*.png"))
        if png_files:
            self._is_png_mode = True
            samples.extend([(f, label) for f in png_files])
            return samples

        # Load pkl files (episode format)
        pkl_files = sorted(glob(os.path.join(dir_path, "*.pkl")))
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)

            frames = episode.get("frames", [])
            if not frames:
                continue

            if self.use_last_frame:
                # Only use last frame with episode-level label
                last_frame = frames[-1]
                if "main_images" in last_frame.get("obs", {}):
                    img = last_frame["obs"]["main_images"]
                    if isinstance(img, np.ndarray):
                        samples.append((img.copy(), label))
            else:
                # Use all frames with per-frame labels
                for frame in frames:
                    if "main_images" in frame.get("obs", {}):
                        img = frame["obs"]["main_images"]
                        frame_label = 1 if frame.get("success", False) else 0
                        if isinstance(img, np.ndarray):
                            samples.append((img.copy(), frame_label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        item, label = self.samples[idx]

        if self._is_png_mode:
            # Load PNG image
            from PIL import Image

            image = Image.open(item).convert("RGB")
            image = image.resize(self.image_size)
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        elif isinstance(item, np.ndarray):
            # Numpy array from pkl (main_images format)
            image = torch.from_numpy(item)

            # Handle channel ordering: ensure [C, H, W]
            if image.dim() == 3:
                if image.shape[-1] in [1, 3, 4]:  # [H, W, C]
                    image = image.permute(2, 0, 1)

            # Normalize to [0, 1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            # Resize if needed
            if image.shape[1:] != self.image_size:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        else:
            # Legacy pickle mode - item is the sample dict
            sample = item
            image = torch.from_numpy(sample["image"])
            label = sample["label"]

            # Handle channel ordering: ensure [C, H, W]
            if image.dim() == 3:
                if image.shape[-1] in [1, 3, 4]:  # [H, W, C]
                    image = image.permute(2, 0, 1)

            # Normalize to [0, 1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            # Resize if needed
            if image.shape[1:] != self.image_size:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        # Apply augmentation
        if self.augment and self.training:
            image = self._augment(image)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image, torch.tensor(label, dtype=torch.float32)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-1])

        # Random brightness/contrast (simple version)
        if torch.rand(1) > 0.5:
            factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
            image = image * factor
            image = torch.clamp(image, 0, 1)

        return image

    @property
    def training(self) -> bool:
        return self.augment


class RewardModelTrainer:
    """Trainer for reward models.

    Provides training, validation, and checkpoint management for reward models.
    Designed to integrate with the RLinf framework.

    Example config:
        ```yaml
        reward_model_training:
          enabled: True
          data_path: "/path/to/reward_data"
          epochs: 100
          batch_size: 64
          lr: 1e-4
          save_dir: "/path/to/checkpoints"
        ```
    """

    def __init__(self, cfg: DictConfig, model: nn.Module):
        """Initialize the trainer.

        Args:
            cfg: Training configuration.
            model: Reward model to train.
        """
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.epochs = cfg.get("epochs", 100)
        self.batch_size = cfg.get("batch_size", 64)
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 1e-5)
        self.val_split = cfg.get("val_split", 0.1)
        self.save_dir = cfg.get("save_dir", "./reward_checkpoints")
        self.save_best = cfg.get("save_best", True)
        self.early_stopping_patience = cfg.get("early_stopping_patience", 10)

        # Initialize
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        self.criterion = nn.BCELoss()

        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def prepare_data(self, data_path: str) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders.

        Args:
            data_path: Path to collected reward data.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        image_size = self.cfg.get("image_size", [3, 224, 224])
        max_success = self.cfg.get("max_success", 5000)
        max_failure = self.cfg.get("max_failure", 5000)
        use_last_frame = self.cfg.get("use_last_frame", True)

        dataset = RewardDataset(
            data_path,
            image_size=(image_size[1], image_size[2]),
            augment=True,
            max_success=max_success,
            max_failure=max_failure,
            use_last_frame=use_last_frame,
        )

        # Split into train/val
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(f"Train size: {train_size}, Val size: {val_size}")
        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        return {
            "train_loss": total_loss / total,
            "train_acc": correct / total,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "val_loss": total_loss / total,
            "val_acc": correct / total,
        }

    def train(self, data_path: str) -> dict[str, Any]:
        """Run full training loop.

        Args:
            data_path: Path to collected reward data.

        Returns:
            Dictionary of final training results.
        """
        train_loader, val_loader = self.prepare_data(data_path)

        logger.info(f"Starting training for {self.epochs} epochs")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])

            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])

            # Update scheduler
            self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.4f}"
            )

            # Save best model
            if self.save_best:
                if val_metrics["val_acc"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val_acc"]
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        return {
            "history": history,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        save_path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )
        logger.info(f"Saved checkpoint to {save_path}")
        return save_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
