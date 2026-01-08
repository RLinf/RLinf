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

"""Base class for multi-frame video reward models.

This module provides the abstract base class for video-based reward models
that process sequences of frames. Suitable for Vision-Language Models (VLMs)
that can understand temporal context for reward prediction.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Literal

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards.embodiment.base_reward_model import BaseRewardModel


# Type alias for frame sampling strategies
SampleStrategy = Literal["uniform_k", "last_k", "first_last_k", "random_k"]


class BaseVideoRewardModel(BaseRewardModel):
    """Base class for multi-frame video reward models.

    This class is designed for reward models that process video sequences
    with temporal context. It is suitable for Vision-Language Models (VLMs)
    such as Qwen3-VL, InternVL, or LLaVA that can understand task progress
    from video frames.

    The input images are expected to be video sequences of shape [B, T, C, H, W],
    where:
        - B: Batch size
        - T: Number of frames (temporal dimension)
        - C: Number of channels (typically 3 for RGB)
        - H: Image height
        - W: Image width

    The class provides frame sampling strategies for efficient inference,
    as VLMs may not need to process all frames.

    Attributes:
        sample_k: Number of frames to sample from the sequence.
        sample_strategy: Strategy for frame sampling (uniform_k, last_k, etc.).
        task_prompt_template: Template string for generating task prompts.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the video reward model.

        Args:
            cfg: Configuration dictionary containing:
                - sample_k: Number of frames to sample (default: 6).
                - sample_strategy: Frame sampling strategy (default: "uniform_k").
                - task_prompt_template: Template for task prompts.
                - model_path: Path to the VLM model.
                - checkpoint_path: Optional path to fine-tuned checkpoint.
        """
        super().__init__(cfg)
        self.sample_k = cfg.get("sample_k", 6)
        self.sample_strategy: SampleStrategy = cfg.get("sample_strategy", "uniform_k")
        self.task_prompt_template = cfg.get(
            "task_prompt_template",
            "Is the task '{task}' completed successfully in this video?"
        )

    @abstractmethod
    def compute_reward(
        self,
        observations: Dict[str, Any],
        task_descriptions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from video observations.

        Args:
            observations: Dictionary containing:
                - 'images': Video tensor of shape [B, T, C, H, W] or [B, T, H, W, C].
                - 'states': Optional state tensor of shape [B, T, state_dim].
            task_descriptions: List of task description strings for each sample.
                These are used to construct prompts for the VLM.

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        pass

    def sample_frames(
        self,
        images: torch.Tensor,
        strategy: Optional[SampleStrategy] = None,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample frames from video sequence using specified strategy.

        Args:
            images: Video tensor of shape [B, T, C, H, W].
            strategy: Sampling strategy. If None, uses self.sample_strategy.
            k: Number of frames to sample. If None, uses self.sample_k.

        Returns:
            torch.Tensor: Sampled frames of shape [B, K, C, H, W], where K
                is the number of sampled frames.
        """
        if strategy is None:
            strategy = self.sample_strategy
        if k is None:
            k = self.sample_k

        B, T, C, H, W = images.shape

        # If we have fewer frames than k, return all frames
        if T <= k:
            return images

        if strategy == "uniform_k":
            # Uniformly sample k frames across the sequence
            indices = torch.linspace(0, T - 1, k).long()
        elif strategy == "last_k":
            # Take the last k frames
            indices = torch.arange(T - k, T)
        elif strategy == "first_last_k":
            # Take first frame, last frame, and uniformly sample the rest
            if k <= 2:
                indices = torch.tensor([0, T - 1])[:k]
            else:
                middle_indices = torch.linspace(1, T - 2, k - 2).long()
                indices = torch.cat([
                    torch.tensor([0]),
                    middle_indices,
                    torch.tensor([T - 1])
                ])
        elif strategy == "random_k":
            # Randomly sample k frames
            indices = torch.randperm(T)[:k].sort().values
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return images[:, indices]

    def preprocess_video(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess video for model input.

        Handles channel ordering and normalization for video sequences.

        Args:
            images: Input video tensor, either [B, T, H, W, C] or [B, T, C, H, W].

        Returns:
            torch.Tensor: Preprocessed video tensor of shape [B, T, C, H, W],
                normalized to [0, 1] range.
        """
        # Handle channel-last format (BTHWC -> BTCHW)
        if images.dim() == 5 and images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 1, 4, 2, 3)

        # Normalize to [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        return images.to(self.device)

    def format_prompt(self, task_description: str) -> str:
        """Format task description using the prompt template.

        Args:
            task_description: The task description string.

        Returns:
            str: Formatted prompt string for the VLM.
        """
        return self.task_prompt_template.format(task=task_description)

    @property
    def model_type(self) -> str:
        """Return the type identifier of this reward model."""
        return "video"

