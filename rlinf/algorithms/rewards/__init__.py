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

from rlinf.algorithms.rewards.code import CodeRewardOffline
from rlinf.algorithms.rewards.math import MathReward
from rlinf.algorithms.rewards.searchr1 import SearchR1Reward
from rlinf.algorithms.rewards.vqa import VQAReward


def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class


def get_reward_class(name: str):
    assert name in reward_registry, f"Reward {name} not found"
    return reward_registry[name]


reward_registry = {}

register_reward("math", MathReward)
register_reward("vqa", VQAReward)
register_reward("code_offline", CodeRewardOffline)
register_reward("searchr1", SearchR1Reward)
