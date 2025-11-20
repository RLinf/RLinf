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

"""Reward model package for embodied RL tasks.

This package contains:
- RewardDataCollector: Collects trajectory data during RL training
- BinaryRewardClassifier: Frame-based binary classifier for reward prediction
- train_reward_model.py: Training script for reward classifier
"""

from rlinf.models.reward_model.reward_classifier import BinaryRewardClassifier, ResNetEncoderWrapper
from rlinf.models.reward_model.reward_data_collector import RewardDataCollector, create_reward_data_collector

__all__ = [
    "BinaryRewardClassifier",
    "ResNetEncoderWrapper",
    "RewardDataCollector",
    "create_reward_data_collector",
]

