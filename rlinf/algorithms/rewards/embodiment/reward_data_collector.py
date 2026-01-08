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

"""Reward data collector for training reward models.

This module provides utilities to collect labeled image data during RL training
for subsequent reward model training. Uses a fixed-size buffer with balanced
sampling strategy.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class RewardDataCollector:
    """Collects labeled image data for reward model training.

    Uses a fixed-size buffer strategy:
    - Maintains separate buffers for success and failure samples
    - Samples failure frames throughout episodes (since success is rare early)
    - Samples success frames when they occur (termination=True)
    - When buffer is full, randomly replaces old samples to maintain distribution
    - Saves images as PNG files for easy inspection

    Attributes:
        success_buffer: List of success sample indices (paths)
        failure_buffer: List of failure sample indices (paths)
        target_success: Target number of success samples
        target_failure: Target number of failure samples
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the reward data collector.

        Args:
            cfg: Configuration containing:
                - enabled: Whether collection is enabled
                - save_dir: Directory to save images
                - target_success: Target buffer size for success samples
                - target_failure: Target buffer size for failure samples
                - sample_rate_fail: Probability to sample a fail frame (0-1)
                - sample_rate_success: Probability to sample a success frame (0-1)
        """
        self.cfg = cfg
        self.enabled = cfg.get("enabled", False)
        self.save_dir = cfg.get("save_dir", "./reward_data")
        
        # Buffer sizes
        self.target_success = cfg.get("target_success", 5000)
        self.target_failure = cfg.get("target_failure", 5000)
        
        # Sampling rates
        self.sample_rate_fail = cfg.get("sample_rate_fail", 0.1)  # Sample 10% of fail frames
        self.sample_rate_success = cfg.get("sample_rate_success", 1.0)  # Sample all success frames
        
        # Buffers store file paths
        self.success_buffer: List[str] = []
        self.failure_buffer: List[str] = []
        
        # Counters
        self.total_success_seen = 0
        self.total_failure_seen = 0
        self._sample_counter = 0
        
        if self.enabled:
            # Create directories
            self.success_dir = os.path.join(self.save_dir, "success")
            self.failure_dir = os.path.join(self.save_dir, "failure")
            os.makedirs(self.success_dir, exist_ok=True)
            os.makedirs(self.failure_dir, exist_ok=True)
            
            logger.info(
                f"RewardDataCollector initialized. Save dir: {self.save_dir}, "
                f"Buffer: {self.target_success} success + {self.target_failure} failure, "
                f"Sample rates: fail={self.sample_rate_fail}, success={self.sample_rate_success}"
            )

    @property
    def success_count(self) -> int:
        """Current number of success samples in buffer."""
        return len(self.success_buffer)

    @property
    def failure_count(self) -> int:
        """Current number of failure samples in buffer."""
        return len(self.failure_buffer)

    @property
    def is_full(self) -> bool:
        """Check if both buffers have reached target size."""
        return (self.success_count >= self.target_success and 
                self.failure_count >= self.target_failure)

    def _save_image(
        self, 
        image: torch.Tensor, 
        label: int,
        goal_pos_2d: Optional[tuple] = None,
    ) -> Optional[str]:
        """Save image to disk with optional goal marker.

        Args:
            image: Image tensor [C, H, W] or [H, W, C], values in [0, 255] or [0, 1]
            label: 1 for success, 0 for failure
            goal_pos_2d: Optional (x, y) pixel coordinates to draw green marker

        Returns:
            Path to saved image, or None if save failed
        """
        try:
            # Convert to numpy
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
            else:
                img_np = image

            # Handle channel ordering [C, H, W] -> [H, W, C]
            if img_np.ndim == 3:
                if img_np.shape[0] in [1, 3, 4]:  # Likely [C, H, W]
                    img_np = np.transpose(img_np, (1, 2, 0))

            # Normalize to [0, 255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # Handle grayscale
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 1:
                img_np = np.concatenate([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 4:  # RGBA -> RGB
                img_np = img_np[:, :, :3]

            # Draw goal marker if provided
            if goal_pos_2d is not None:
                img_np = self._draw_goal_marker(img_np, goal_pos_2d)

            # Create PIL image
            pil_img = Image.fromarray(img_np)

            # Generate filename
            self._sample_counter += 1
            save_dir = self.success_dir if label == 1 else self.failure_dir
            filename = f"{self._sample_counter:08d}.png"
            filepath = os.path.join(save_dir, filename)

            # Save
            pil_img.save(filepath)
            return filepath

        except Exception as e:
            logger.warning(f"Failed to save image: {e}")
            return None

    def _draw_goal_marker(
        self, 
        img_np: np.ndarray, 
        goal_pos_2d: tuple,
        color: tuple = (0, 255, 0),  # Green
        radius: int = 5,
    ) -> np.ndarray:
        """Draw a circular marker at the goal position.

        Args:
            img_np: Image array [H, W, C]
            goal_pos_2d: (x, y) pixel coordinates
            color: RGB color tuple
            radius: Circle radius in pixels

        Returns:
            Image with marker drawn
        """
        import cv2
        
        x, y = int(goal_pos_2d[0]), int(goal_pos_2d[1])
        h, w = img_np.shape[:2]
        
        # Clamp to image bounds
        x = max(radius, min(w - radius - 1, x))
        y = max(radius, min(h - radius - 1, y))
        
        # Draw filled circle
        img_np = img_np.copy()
        cv2.circle(img_np, (x, y), radius, color, -1)
        # Draw border for visibility
        cv2.circle(img_np, (x, y), radius, (255, 255, 255), 1)
        
        return img_np

    def _add_to_buffer(self, filepath: str, label: int) -> None:
        """Add sample to buffer, replacing old sample if full.

        Args:
            filepath: Path to saved image
            label: 1 for success, 0 for failure
        """
        if label == 1:
            buffer = self.success_buffer
            target = self.target_success
        else:
            buffer = self.failure_buffer
            target = self.target_failure

        if len(buffer) < target:
            # Buffer not full, just append
            buffer.append(filepath)
        else:
            # Buffer full, randomly replace
            idx = random.randint(0, target - 1)
            old_path = buffer[idx]
            # Delete old file
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except Exception as e:
                logger.warning(f"Failed to delete old image: {e}")
            buffer[idx] = filepath

    def decide_samples(
        self,
        batch_size: int,
        successes: Optional[torch.Tensor] = None,
    ) -> List[tuple]:
        """Decide which samples to collect WITHOUT processing images.
        
        Call this FIRST to determine which indices need images, then only
        extract/process images for those indices.

        Args:
            batch_size: Number of samples in the batch
            successes: Boolean tensor indicating success [B]

        Returns:
            List of (index, label) tuples for samples to collect
        """
        if not self.enabled:
            return []

        # Determine labels
        if successes is None:
            labels = [False] * batch_size
        else:
            labels = successes.bool().cpu().tolist() if isinstance(successes, torch.Tensor) else successes

        samples_to_collect = []

        for i in range(batch_size):
            is_success = labels[i] if i < len(labels) else False
            
            # Decide whether to sample this frame
            if is_success:
                # Success frame - sample with success rate
                self.total_success_seen += 1
                if random.random() > self.sample_rate_success:
                    continue
                # Skip if success buffer is way ahead of failure buffer (keep balance)
                if self.success_count > self.failure_count * 2 and self.failure_count > 0:
                    continue
            else:
                # Failure frame - sample with lower rate
                self.total_failure_seen += 1
                if random.random() > self.sample_rate_fail:
                    continue
                # Skip if failure buffer is way ahead
                if self.failure_count > self.success_count * 2 and self.success_count > 0:
                    continue

            label = 1 if is_success else 0
            samples_to_collect.append((i, label))

        return samples_to_collect

    def save_samples(
        self,
        images: torch.Tensor,
        samples_to_collect: List[tuple],
        goal_positions_2d: Optional[List[tuple]] = None,
    ) -> int:
        """Save only the pre-selected samples.
        
        Call this AFTER decide_samples() with only the needed images.

        Args:
            images: Image tensor [B, H, W, C] or [B, C, H, W]
            samples_to_collect: List of (index, label) from decide_samples()
            goal_positions_2d: Optional list of (x, y) pixel coords

        Returns:
            Number of samples saved
        """
        if not self.enabled or not samples_to_collect:
            return 0

        collected = 0
        for idx, label in samples_to_collect:
            # Get goal position for this sample if available
            goal_pos = None
            if goal_positions_2d is not None and idx < len(goal_positions_2d):
                goal_pos = goal_positions_2d[idx]

            # Save image with optional goal marker
            filepath = self._save_image(images[idx], label, goal_pos_2d=goal_pos)
            
            if filepath:
                self._add_to_buffer(filepath, label)
                collected += 1

        return collected

    def collect(
        self,
        observations: Dict[str, Any],
        terminations: Optional[torch.Tensor] = None,
        successes: Optional[torch.Tensor] = None,
        task_descriptions: Optional[List[str]] = None,
        step_id: int = 0,
        episode_ids: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        goal_positions_2d: Optional[List[tuple]] = None,
    ) -> int:
        """Collect data samples from current batch.

        With lazy rendering, images may not always be present.
        Only collects when images are available.

        Args:
            observations: Dictionary containing 'images' or 'main_images' (may be None with lazy rendering)
            terminations: Not used directly
            successes: Boolean tensor indicating success [B] (from env infos["success"])
            task_descriptions: Optional task descriptions
            step_id: Current step number
            episode_ids: Optional episode IDs
            dones: Episode end signals (for logging)
            goal_positions_2d: Optional list of (x, y) pixel coords for goal markers

        Returns:
            Number of samples collected in this call
        """
        if not self.enabled:
            return 0

        # Get images - with lazy rendering, may be None
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is None:
            # No images this step (lazy rendering skipped) - just update counters
            if successes is not None:
                labels = successes.bool().cpu().tolist() if isinstance(successes, torch.Tensor) else successes
                for is_success in labels:
                    if is_success:
                        self.total_success_seen += 1
                    else:
                        self.total_failure_seen += 1
            return 0

        batch_size = images.shape[0]

        # Step 1: Decide which samples to collect (fast, no image processing)
        samples_to_collect = self.decide_samples(batch_size, successes)

        if not samples_to_collect:
            return 0

        # Step 2: Only process and save the selected samples
        return self.save_samples(images, samples_to_collect, goal_positions_2d)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_success_seen": self.total_success_seen,
            "total_failure_seen": self.total_failure_seen,
            "is_full": self.is_full,
        }

    def save_metadata(self) -> str:
        """Save metadata about collected data.

        Returns:
            Path to metadata file
        """
        import json
        
        metadata = {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_success_seen": self.total_success_seen,
            "total_failure_seen": self.total_failure_seen,
            "success_files": self.success_buffer,
            "failure_files": self.failure_buffer,
            "config": {
                "target_success": self.target_success,
                "target_failure": self.target_failure,
                "sample_rate_success": self.sample_rate_success,
                "sample_rate_fail": self.sample_rate_fail,
            }
        }
        
        filepath = os.path.join(self.save_dir, "metadata.json")
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {filepath}")
        return filepath

    def save(self, filename: Optional[str] = None) -> str:
        """Save metadata (images are already saved individually).

        Args:
            filename: Ignored, kept for API compatibility

        Returns:
            Path to metadata file
        """
        return self.save_metadata()
