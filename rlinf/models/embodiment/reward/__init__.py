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

"""Reward models for embodied RL."""

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel

__all__ = [
    "BaseRewardModel",
    "BaseImageRewardModel",
    "ResNetRewardModel",
]
