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

"""Train a visual reward classifier from success/failure images.

Usage
-----
.. code-block:: bash

    python examples/embodiment/train_reward_classifier.py \\
        --log_dir logs/<timestamp>-reward-classifier-<env> \\
        --pretrained_ckpt RLinf-ResNet10-pretrained/resnet10_pretrained.pt \\
        --image_keys wrist_1 wrist_2 \\
        --num_epochs 200

The ``log_dir`` should be the directory created by
``collect_classifier_data.sh``.  It must contain pickle files whose names
include ``success`` or ``failure``.  The trained checkpoint
(``reward_classifier.pt``) is saved into the same directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

# Allow standalone execution without PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from rlinf.envs.realworld.common.reward_classifier.classifier import (
    RewardClassifier,
)
from rlinf.envs.realworld.common.reward_classifier.data_augmentation import (
    augment_batch,
)


# ── Dataset ──────────────────────────────────────────────────────────


class ClassifierDataset(Dataset):
    """Loads success/failure observation pickles into a flat dataset.

    Each pickle file is a list of transition dicts.  We extract the
    camera frames for the configured ``image_keys`` and assign label
    ``1`` (success) or ``0`` (failure) based on the file name.
    """

    def __init__(
        self,
        data_dir: str,
        image_keys: list[str],
        obs_key: str = "frames",
    ) -> None:
        super().__init__()
        self.image_keys = image_keys
        self.obs_key = obs_key
        self.samples: list[tuple[dict[str, np.ndarray], int]] = []

        for path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
            fname = os.path.basename(path).lower()
            if "success" in fname:
                label = 1
            elif "failure" in fname or "fail" in fname:
                label = 0
            else:
                continue
            with open(path, "rb") as f:
                transitions = pickle.load(f)
            for t in transitions:
                obs = t.get("observations", t)
                frames = obs.get(self.obs_key, obs)
                sample_frames = {}
                for key in image_keys:
                    img = frames.get(key)
                    if img is None:
                        for alt in [key, f"image_{key}", key.replace("wrist_", "")]:
                            if alt in frames:
                                img = frames[alt]
                                break
                    if img is not None:
                        sample_frames[key] = np.asarray(img)
                if len(sample_frames) == len(image_keys):
                    self.samples.append((sample_frames, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frames, label = self.samples[idx]
        tensors = {}
        for key, img in frames.items():
            t = torch.from_numpy(img.copy())
            if t.ndim == 2:
                t = t.unsqueeze(-1).expand(-1, -1, 3)
            if t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            tensors[key] = t.float() / 255.0
        return tensors, label


def collate_fn(batch):
    """Custom collate that groups per-camera tensors into batched dicts."""
    frames_list, labels = zip(*batch)
    keys = frames_list[0].keys()
    batched_frames = {
        k: torch.stack([f[k] for f in frames_list]) for k in keys
    }
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_frames, labels


# ── Training ─────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    log_dir = args.log_dir

    print(f"Loading data from {log_dir} ...")
    dataset = ClassifierDataset(
        data_dir=log_dir,
        image_keys=args.image_keys,
    )
    n_pos = sum(1 for _, l in dataset.samples if l == 1)
    n_neg = len(dataset) - n_pos
    print(f"  success: {n_pos}   failure: {n_neg}   total: {len(dataset)}")
    assert len(dataset) > 0, (
        "No data found. Check --log_dir and --image_keys."
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print("Building classifier ...")
    model = RewardClassifier(
        image_keys=args.image_keys,
        pretrained_ckpt=args.pretrained_ckpt,
        image_size=(args.image_size, args.image_size),
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    print(f"Training for {args.num_epochs} epochs ...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_frames, labels in loader:
            labels = labels.to(device)

            aug_frames = {}
            for key in args.image_keys:
                imgs = batch_frames[key].to(device)
                imgs = augment_batch(imgs, crop_padding=4, color_jitter=True)
                aug_frames[key] = imgs

            logits = model(aug_frames)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        print(
            f"  Epoch {epoch:3d}/{args.num_epochs}  "
            f"loss={avg_loss:.4f}  acc={accuracy:.4f}"
        )

        # Save best checkpoint
        if accuracy > best_acc:
            best_acc = accuracy
            ckpt_path = os.path.join(log_dir, "reward_classifier.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_keys": args.image_keys,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "accuracy": accuracy,
                },
                ckpt_path,
            )

    ckpt_path = os.path.join(log_dir, "reward_classifier.pt")
    print(f"\nBest accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a visual reward classifier."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Log directory (created by collect_classifier_data.sh), "
        "containing success/failure pickle files; weights are also saved here.",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="RLinf-ResNet10-pretrained/resnet10_pretrained.pt",
        help="Path to pretrained ResNet-10 weights.",
    )
    parser.add_argument(
        "--image_keys", nargs="+", default=["wrist_1"],
        help="Camera observation keys (e.g., wrist_1 wrist_2 for dual cameras).",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
