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
from PIL import Image

from rlinf.models.reward_model.reward_classifier import BinaryRewardClassifier


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
        
        # Statistics
        positive_traj_count = 0
        negative_traj_count = 0
        positive_dir_frame_count = 0  # Frames from positive_dir trajectories
        negative_dir_frame_count = 0  # Frames from negative_dir trajectories
        # Separate statistics for each directory
        positive_dir_label_1_count = 0  # Frames with label == 1 in positive_dir
        positive_dir_label_0_count = 0  # Frames with label == 0 in positive_dir
        negative_dir_label_1_count = 0  # Frames with label == 1 in negative_dir
        negative_dir_label_0_count = 0  # Frames with label == 0 in negative_dir
        
        # Load positive trajectories
        positive_files = sorted(list(self.positive_dir.glob("*.npy")))
        for traj_path in positive_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if isinstance(traj_data, dict) and 'images' in traj_data and 'labels' in traj_data:
                if self.image_key in traj_data['images']:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data['images'][self.image_key].shape[0]
                    positive_dir_frame_count += num_frames
                    positive_traj_count += 1
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame (even in positive trajectories)
                        label = float(traj_data['labels'][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))
                        if label >= 0.5:  # Treat >= 0.5 as positive
                            positive_dir_label_1_count += 1
                        else:
                            positive_dir_label_0_count += 1
        
        # Load negative trajectories
        negative_files = sorted(list(self.negative_dir.glob("*.npy")))
        for traj_path in negative_files:
            traj_data = np.load(traj_path, allow_pickle=True).item()
            if isinstance(traj_data, dict) and 'images' in traj_data and 'labels' in traj_data:
                if self.image_key in traj_data['images']:
                    # Each trajectory file contains multiple frames
                    num_frames = traj_data['images'][self.image_key].shape[0]
                    negative_dir_frame_count += num_frames
                    negative_traj_count += 1
                    for frame_idx in range(num_frames):
                        # Use frame's own label from success_frame
                        label = float(traj_data['labels'][frame_idx])
                        self.samples.append((traj_path, frame_idx, label))
                        if label >= 0.5:  # Treat >= 0.5 as positive
                            negative_dir_label_1_count += 1
                        else:
                            negative_dir_label_0_count += 1

        self.num_samples = len(self.samples)
        
        # Calculate totals
        total_label_1_count = positive_dir_label_1_count + negative_dir_label_1_count
        total_label_0_count = positive_dir_label_0_count + negative_dir_label_0_count
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Dataset Statistics:")
        print(f"{'='*60}")
        print(f"Trajectories: {positive_traj_count} positive, {negative_traj_count} negative")
        print(f"\nFrom positive_dir ({self.positive_dir}):")
        print(f"  Total frames: {positive_dir_frame_count}")
        print(f"  Frames with label=1 (success): {positive_dir_label_1_count}")
        print(f"  Frames with label=0 (failure): {positive_dir_label_0_count}")
        if positive_dir_frame_count > 0:
            print(f"  Label ratio (1/0): {positive_dir_label_1_count}/{positive_dir_label_0_count} = {positive_dir_label_1_count/max(positive_dir_label_0_count, 1):.3f}")
        print(f"\nFrom negative_dir ({self.negative_dir}):")
        print(f"  Total frames: {negative_dir_frame_count}")
        print(f"  Frames with label=1 (success): {negative_dir_label_1_count}")
        print(f"  Frames with label=0 (failure): {negative_dir_label_0_count}")
        if negative_dir_frame_count > 0:
            print(f"  Label ratio (1/0): {negative_dir_label_1_count}/{negative_dir_label_0_count} = {negative_dir_label_1_count/max(negative_dir_label_0_count, 1):.3f}")
        print(f"\nOverall:")
        print(f"  Total frames: {self.num_samples}")
        print(f"  Total frames with label=1 (success): {total_label_1_count}")
        print(f"  Total frames with label=0 (failure): {total_label_0_count}")
        print(f"  Overall label ratio (1/0): {total_label_1_count}/{total_label_0_count} = {total_label_1_count/max(total_label_0_count, 1):.3f}")
        print(f"{'='*60}\n")

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


def visualize_positive_samples(dataset, output_dir="positive_samples_vis"):
    """Visualize all samples with label=1 (positive samples).
    
    Args:
        dataset: RewardDataset instance
        output_dir: Directory to save visualization images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Visualizing positive samples (label=1)...")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")
    
    positive_samples = []
    for idx in range(len(dataset.samples)):
        traj_path, frame_idx, label = dataset.samples[idx]
        if label >= 0.5:  # Positive label
            positive_samples.append((traj_path, frame_idx, label))
    
    print(f"Found {len(positive_samples)} positive samples to visualize")
    
    if len(positive_samples) == 0:
        print("No positive samples found!")
        return
    
    # Visualize each positive sample
    for sample_idx, (traj_path, frame_idx, label) in enumerate(tqdm(positive_samples, desc="Saving images")):
        # Determine which directory this trajectory comes from
        traj_path_obj = Path(traj_path)
        if str(dataset.positive_dir) in str(traj_path_obj.parent):
            dir_name = "positive"
        elif str(dataset.negative_dir) in str(traj_path_obj.parent):
            dir_name = "negative"
        else:
            # Extract directory name from path
            dir_name = traj_path_obj.parent.name
        
        # Get trajectory filename (without extension)
        traj_filename = traj_path_obj.stem
        
        # Construct image filename: dir_name_traj_filename_frame_idx.png
        image_filename = f"{dir_name}_{traj_filename}_frame{frame_idx:03d}.png"
        image_path = output_path / image_filename
        
        # Load and save the image
        traj_data = np.load(traj_path, allow_pickle=True).item()
        frame_image = traj_data['images'][dataset.image_key][frame_idx]
        
        # Convert to numpy array if needed
        if isinstance(frame_image, torch.Tensor):
            frame_image = frame_image.cpu().numpy()
        
        # Ensure shape is [H, W, C] for PIL
        if len(frame_image.shape) == 3:
            if frame_image.shape[0] == dataset.image_size[0]:  # [C, H, W]
                frame_image = np.transpose(frame_image, (1, 2, 0))  # [H, W, C]
            # else already [H, W, C]
        
        # Normalize to [0, 255] if needed
        if frame_image.dtype == np.float32 or frame_image.dtype == np.float64:
            if frame_image.max() <= 1.0:
                frame_image = (frame_image * 255).astype(np.uint8)
            else:
                frame_image = np.clip(frame_image, 0, 255).astype(np.uint8)
        else:
            frame_image = np.clip(frame_image, 0, 255).astype(np.uint8)
        
        # Handle grayscale (single channel) or RGB
        if len(frame_image.shape) == 2:
            # Grayscale
            pass
        elif len(frame_image.shape) == 3:
            if frame_image.shape[2] == 1:
                frame_image = frame_image.squeeze(2)
            elif frame_image.shape[2] == 3:
                # RGB - ensure it's uint8
                pass
        
        # Save image
        Image.fromarray(frame_image).save(image_path)
    
    print(f"Saved {len(positive_samples)} positive sample images to {output_path}")
    print(f"{'='*60}\n")


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
    parser.add_argument(
        "--visualize-positive",
        action="store_true",
        help="Visualize all positive samples (label=1) before training",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=str,
        default="positive_samples_vis",
        help="Output directory for positive sample visualizations",
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
    
    # Visualize positive samples if requested
    if args.visualize_positive:
        visualize_positive_samples(train_dataset, args.vis_output_dir)
    
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

