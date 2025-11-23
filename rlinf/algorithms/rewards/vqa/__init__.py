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

import torch
from omegaconf import DictConfig

from .format_rewards import answer_format_reward, think_format_reward
from .qa_rewards import qa_accuracy_reward


class VQAReward:
    NEEDED_REWARD_FUNCTIONS = {
        "qa_accuracy": qa_accuracy_reward,
        "think_format": think_format_reward,
        "answer_format": answer_format_reward,
    }

    def __init__(self, config: DictConfig):
        assert "reward_weights" in config, "VQAReward requires reward_weights in config"

        self.reward_weights_config = config.reward_weights
        assert set(self.reward_weights_config.keys()) == set(
            self.NEEDED_REWARD_FUNCTIONS.keys()
        ), (
            f"Reward weights must contains all of: {self.NEEDED_REWARD_FUNCTIONS.keys()} but got {list(self.reward_weights_config.keys())}"
        )
        assert all(
            reward_weight >= 0 for reward_weight in self.reward_weights_config.values()
        ), (
            f"All reward weights must be non-negative but got {list(self.reward_weights_config.values())}"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_reward(self, completions: list[str], answers: list[dict]) -> list[float]:
        rewards = []
        reward_weights = []
        for reward_name, reward_function in self.NEEDED_REWARD_FUNCTIONS.items():
            if self.reward_weights_config[reward_name] > 0:
                rewards.append(reward_function(completions, answers))
            else:
                rewards.append([0.0] * len(completions))
            reward_weights.append(self.reward_weights_config[reward_name])

        rewards_tensor = torch.tensor(rewards, device=self.device)
        weights_tensor = torch.tensor(reward_weights, device=self.device)

        final_rewards = (rewards_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)

        return final_rewards.tolist()
