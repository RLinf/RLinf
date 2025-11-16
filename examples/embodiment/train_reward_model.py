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

"""Training script for frame-based reward classifier model."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rlinf.models.embodiment.reward_classifier import BinaryRewardClassifier


class RewardDataset(Dataset):
    """Dataset for training reward classifier with trajectory-level data.

    Expected data format:
    - Each .npy file contains a dictionary with:
      - 'images': Dict of numpy arrays, keyed by image_key, shape [T, C, H, W]
      - 'labels': numpy array of success_frame labels, shape [T]
    - positive_dir: Directory containing positive trajectory files (success_once=1)
    - negative_dir: Directory containing negative trajectory files (success_once=0)
    """

    def __init__(
        self,
        positive_dir: str,
        negative_dir: str,
        image_key: str,
        image_size: list,
        device: str = "cpu",
    ):
        """
        Args:
            positive_dir: Path to directory containing positive trajectory files
            negative_dir: Path to directory containing negative trajectory files
            image_key: Image key to use (e.g., "base_camera")
            image_size: Image size [C, H, W]
            device: Device to load data on
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.image_key = image_key
        self.image_size = image_size
        self.device = device

        if not self.positive_dir.exists():
            raise ValueError(f"Positive directory not found: {self.positive_dir}")
        if not self.negative_dir.exists():
            raise ValueError(f"Negative directory not found: {self.negative_dir}")

        # Collect all trajectory files from both directories
        self.samples = []
        
        # Load positive trajectories
        positive_files = sorted(list(self.positive_dir.glob("*.npy")))
        for traj_path in positive_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if isinstance(traj_data, dict) and 'images' in traj_data and 'labels' in traj_data:
                if self.image_key in traj_data['images']:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data['images'][self.image_key].shape[0]
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame (even in positive trajectories)
                        label = float(traj_data['labels'][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))
        
        # Load negative trajectories
        negative_files = sorted(list(self.negative_dir.glob("*.npy")))
        for traj_path in negative_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if isinstance(traj_data, dict) and 'images' in traj_data and 'labels' in traj_data:
                if self.image_key in traj_data['images']:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data['images'][self.image_key].shape[0]
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame
                        label = float(traj_data['labels'][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))

        self.num_samples = len(self.samples)
        print(f"Loaded {len(positive_files)} positive and {len(negative_files)} negative trajectories")
        print(f"Total {self.num_samples} frames from all trajectories")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a single sample (frame).

        Returns:
            images: Dict of image tensors, keyed by image_key, shape [C, H, W]
            label: Binary label tensor (0 or 1) from success_frame
        """
        traj_path, frame_idx, label = self.samples[idx]

        # Load trajectory data
        traj_data = np.load(traj_path, allow_pickle=True).item()
        
        if not isinstance(traj_data, dict) or 'images' not in traj_data:
            raise ValueError(f"Invalid trajectory file format: {traj_path}")
        
        if self.image_key not in traj_data['images']:
            raise ValueError(f"Image key {self.image_key} not found in trajectory file: {traj_path}")
        
        # Extract frame image: [T, C, H, W] -> [C, H, W]
        frame_image = traj_data['images'][self.image_key][frame_idx]
        
        # Convert to torch tensor
        if isinstance(frame_image, np.ndarray):
            img = torch.from_numpy(frame_image).float()
        else:
            img = torch.tensor(frame_image, dtype=torch.float32)

        # Ensure correct shape [C, H, W]
        if img.dim() == 3:
            if img.shape[0] == self.image_size[0]:
                pass  # Already [C, H, W]
            else:
                img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        elif img.dim() == 4:
            img = img.squeeze(0)  # [1, C, H, W] -> [C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Ensure correct size
        if img.shape != tuple(self.image_size):
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=self.image_size[1:], mode="bilinear"
            ).squeeze(0)

        images = {self.image_key: img}
        
        # Use label from success_frame (which may differ from trajectory label)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return images, label_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images_dict, labels in pbar:
        # Move to device
        images_dict = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]

        # Forward pass
        logits = model(images_dict, train=True)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "acc": correct / total if total > 0 else 0.0,
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy




def main():
    parser = argparse.ArgumentParser(description="Train reward classifier model")
    parser.add_argument(
        "--positive-dir",
        type=str,
        required=True,
        help="Path to directory containing positive samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        required=True,
        help="Path to directory containing negative samples",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Path to save the trained model checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet10",
        help="Backbone architecture (resnet10, resnet18, etc.)",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="base_camera",
        help="Image key to use",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=3,
        default=[3, 64, 64],
        help="Image size [C, H, W]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for classifier",
    )
    parser.add_argument(
        "--num-spatial-blocks",
        type=int,
        default=8,
        help="Number of spatial blocks for pooling",
    )
    parser.add_argument(
        "--pretrained-encoder-path",
        type=str,
        default=None,
        help="Path to pretrained encoder weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    # Create output directory (argparse converts --arg-name to arg_name)
    output_dir = os.path.dirname(args.output_checkpoint)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create model (currently only supports ResNet10, but can be extended)
    if args.backbone != "resnet10":
        print(f"Warning: Only resnet10 is currently supported. Using resnet10 instead of {args.backbone}")

    model = BinaryRewardClassifier(
        image_keys=[args.image_key],
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        num_spatial_blocks=args.num_spatial_blocks,
        pretrained_encoder_path=args.pretrained_encoder_path,
        use_pretrain=args.pretrained_encoder_path is not None,
        freeze_encoder=True,
    )

    model = model.to(args.device)

    # Create dataset
    train_dataset = RewardDataset(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        image_key=args.image_key,
        image_size=args.image_size,
        device=args.device,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False,
    )

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_train_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save best model
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            torch.save(checkpoint, args.output_checkpoint)
            print(f"Saved best model with train_acc: {train_acc:.4f} to {args.output_checkpoint}")

    print("\nTraining completed!")
    print(f"Best model saved to {args.output_checkpoint}")


if __name__ == "__main__":
    main()

