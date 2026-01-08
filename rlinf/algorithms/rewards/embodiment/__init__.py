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

"""Embodied reward models for reinforcement learning.

This package provides reward model implementations for embodied RL tasks,
including both image-based (single-frame) and video-based (multi-frame) models.

Available Models:
    - ResNetRewardModel: ResNet-based binary classifier for success detection.
    - Qwen3VLRewardModel: Qwen3-VL based video reward model (placeholder).

Usage:
    The recommended way to use reward models is through the RewardManager:

    ```python
    from rlinf.algorithms.rewards.embodiment import RewardManager

    # Initialize from config
    reward_manager = RewardManager(cfg.reward)

    # Compute rewards
    rewards = reward_manager.compute_rewards(observations, task_descriptions)
    ```

    Alternatively, you can instantiate models directly:

    ```python
    from rlinf.algorithms.rewards.embodiment import ResNetRewardModel

    model = ResNetRewardModel(cfg)
    rewards = model.compute_reward(observations)
    ```
"""

from rlinf.algorithms.rewards.embodiment.base_reward_model import BaseRewardModel
from rlinf.algorithms.rewards.embodiment.base_image_reward_model import (
    BaseImageRewardModel,
)
from rlinf.algorithms.rewards.embodiment.base_video_reward_model import (
    BaseVideoRewardModel,
)
from rlinf.algorithms.rewards.embodiment.resnet_reward_model import ResNetRewardModel
from rlinf.algorithms.rewards.embodiment.qwen3_vl_reward_model import (
    Qwen3VLRewardModel,
)
from rlinf.algorithms.rewards.embodiment.reward_manager import RewardManager
from rlinf.algorithms.rewards.embodiment.reward_data_collector import (
    RewardDataCollector,
)
from rlinf.algorithms.rewards.embodiment.reward_model_trainer import (
    RewardModelTrainer,
    RewardDataset,
)

__all__ = [
    # Base classes
    "BaseRewardModel",
    "BaseImageRewardModel",
    "BaseVideoRewardModel",
    # Concrete implementations
    "ResNetRewardModel",
    "Qwen3VLRewardModel",
    # Manager
    "RewardManager",
    # Data collection and training
    "RewardDataCollector",
    "RewardModelTrainer",
    "RewardDataset",
]

