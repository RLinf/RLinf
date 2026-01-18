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

# !/usr/bin/env python3
"""Test script to evaluate the reward model on debug images and training data.

Usage:
    python examples/reward/test_reward_model.py

    # Or with custom config
    python examples/reward/test_reward_model.py checkpoint_path=/path/to/checkpoint
"""

import glob
import os
import pickle
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rlinf.models.embodiment.reward import ResNetRewardModel


def load_debug_images(debug_dir: str, max_images: int = 50):
    """Load images from debug directories."""
    success_dir = os.path.join(debug_dir, "resnet_success")
    fail_dir = os.path.join(debug_dir, "resnet_fail")

    success_images = []
    fail_images = []

    # Load success images
    if os.path.exists(success_dir):
        for img_path in sorted(glob.glob(os.path.join(success_dir, "*.png")))[
            :max_images
        ]:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            success_images.append(img)

    # Load fail images
    if os.path.exists(fail_dir):
        for img_path in sorted(glob.glob(os.path.join(fail_dir, "*.png")))[:max_images]:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            fail_images.append(img)

    return success_images, fail_images


def load_training_data(data_path: str, max_each: int = 50):
    """Load a subset of training data."""
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    success_files = [f for f in pkl_files if "_success.pkl" in f]
    fail_files = [f for f in pkl_files if "_fail.pkl" in f]

    success_images = []
    fail_images = []

    # Load success images
    for pkl_path in success_files:
        if len(success_images) >= max_each:
            break
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
            observations = episode.get("observations", [])
            if observations:
                # Take last frame
                obs = observations[-1]
                img = obs.get("main_images") or obs.get("images")
                if img is not None:
                    if isinstance(img, np.ndarray):
                        if img.ndim == 4:
                            img = img.squeeze(0)
                        success_images.append(img)
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            continue

    # Load fail images
    for pkl_path in fail_files:
        if len(fail_images) >= max_each:
            break
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
            observations = episode.get("observations", [])
            if observations:
                obs = observations[-1]
                img = obs.get("main_images") or obs.get("images")
                if img is not None:
                    if isinstance(img, np.ndarray):
                        if img.ndim == 4:
                            img = img.squeeze(0)
                        fail_images.append(img)
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            continue

    return success_images, fail_images


@hydra.main(version_base="1.1", config_path="config", config_name="test_reward_model")
def main(cfg: DictConfig) -> None:
    """Main test function."""
    # Get paths from config
    checkpoint_path = cfg.checkpoint_path
    debug_dir = cfg.debug_dir
    training_data_path = cfg.training_data_path

    # Create model config from yaml
    model_cfg = DictConfig(
        {
            "arch": cfg.model.arch,
            "pretrained": cfg.model.pretrained,
            "checkpoint_path": checkpoint_path,
            "normalize": cfg.model.normalize,
        }
    )

    # Load model
    print(f"Loading model from {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetRewardModel(model_cfg)
    model = model.to(device)
    model.eval()

    # Test on debug images
    print("\n=== Testing on DEBUG images ===")
    success_imgs, fail_imgs = load_debug_images(
        debug_dir, max_images=cfg.test.max_debug_images
    )
    print(f"Loaded {len(success_imgs)} success images, {len(fail_imgs)} fail images")

    if success_imgs and fail_imgs:
        # Process success images
        success_tensors = []
        for img in success_imgs:
            t = torch.from_numpy(img).float()
            if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
                t = t.permute(2, 0, 1)
            success_tensors.append(t)
        success_batch = torch.stack(success_tensors).to(device)

        # Process fail images
        fail_tensors = []
        for img in fail_imgs:
            t = torch.from_numpy(img).float()
            if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
                t = t.permute(2, 0, 1)
            fail_tensors.append(t)
        fail_batch = torch.stack(fail_tensors).to(device)

        with torch.no_grad():
            success_rewards = model.compute_reward({"images": success_batch})
            fail_rewards = model.compute_reward({"images": fail_batch})

        print("\n'Success' images (classified as success by model):")
        print(
            f"  Probabilities: mean={success_rewards.mean():.4f}, min={success_rewards.min():.4f}, max={success_rewards.max():.4f}"
        )
        correct = (success_rewards > 0.5).sum().item()
        print(f"  Correctly classified (>0.5): {correct}/{len(success_rewards)}")

        print("\n'Fail' images (classified as fail by model):")
        print(
            f"  Probabilities: mean={fail_rewards.mean():.4f}, min={fail_rewards.min():.4f}, max={fail_rewards.max():.4f}"
        )
        correct = (fail_rewards <= 0.5).sum().item()
        print(f"  Correctly classified (<=0.5): {correct}/{len(fail_rewards)}")

    # Test on training data
    print("\n\n=== Testing on TRAINING data ===")
    success_imgs, fail_imgs = load_training_data(
        training_data_path, max_each=cfg.test.max_training_images
    )
    print(f"Loaded {len(success_imgs)} success images, {len(fail_imgs)} fail images")

    if success_imgs and fail_imgs:
        # Process success images
        success_tensors = []
        for img in success_imgs:
            t = torch.from_numpy(img).float()
            if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
                t = t.permute(2, 0, 1)
            success_tensors.append(t)
        success_batch = torch.stack(success_tensors).to(device)

        # Process fail images
        fail_tensors = []
        for img in fail_imgs:
            t = torch.from_numpy(img).float()
            if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
                t = t.permute(2, 0, 1)
            fail_tensors.append(t)
        fail_batch = torch.stack(fail_tensors).to(device)

        with torch.no_grad():
            success_rewards = model.compute_reward({"images": success_batch})
            fail_rewards = model.compute_reward({"images": fail_batch})

        print("\nTraining SUCCESS images:")
        print(
            f"  Probabilities: mean={success_rewards.mean():.4f}, min={success_rewards.min():.4f}, max={success_rewards.max():.4f}"
        )
        correct = (success_rewards > 0.5).sum().item()
        print(f"  Correctly classified (>0.5): {correct}/{len(success_rewards)}")

        print("\nTraining FAIL images:")
        print(
            f"  Probabilities: mean={fail_rewards.mean():.4f}, min={fail_rewards.min():.4f}, max={fail_rewards.max():.4f}"
        )
        correct = (fail_rewards <= 0.5).sum().item()
        print(f"  Correctly classified (<=0.5): {correct}/{len(fail_rewards)}")


if __name__ == "__main__":
    main()
