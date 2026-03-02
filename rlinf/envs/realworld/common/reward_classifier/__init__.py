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

"""Visual reward classifier for real-world embodied RL.

Provides a binary image classifier built on a frozen pretrained ResNet-10
backbone.  The classifier is trained on success/failure images collected
via teleoperation and used at runtime to compute dense visual rewards.
"""

from .classifier import RewardClassifier, load_reward_classifier

__all__ = [
    "RewardClassifier",
    "load_reward_classifier",
]
