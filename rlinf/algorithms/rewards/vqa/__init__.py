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

from typing import List

import torch
from omegaconf import DictConfig

from .format_rewards import answer_format_reward, think_format_reward
from .qa_rewards import qa_accuracy_reward


class VQAReward:
    def __init__(self, config: DictConfig):
        reward_weights_config = config.get(
            "reward_weights",
            {
                "qa_accuracy": 1.0,
                "think_format": 0.0,
                "answer_format": 0.0,
            },
        )
        for reward_name, reward_weight in reward_weights_config.items():
            assert reward_name in ["qa_accuracy", "think_format", "answer_format"], (
                f"Reward {reward_name} not supported"
            )
            assert reward_weight >= 0, (
                f"Reward weight {reward_weight} must be non-negative"
            )
        self.reward_weights = [
            reward_weights_config["qa_accuracy"],
            reward_weights_config["think_format"],
            reward_weights_config["answer_format"],
        ]

        self.reward_functions = [
            qa_accuracy_reward,
            think_format_reward,
            answer_format_reward,
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_reward(self, completions: List[str], answers: List[dict]) -> List[float]:
        rewards = []
        for i, reward_function in enumerate(self.reward_functions):
            if self.reward_weights[i] > 0:
                rewards.append(reward_function(completions, answers))
            else:
                rewards.append([0.0] * len(completions))

        # Apply weights to each reward function's output and sum

        # rewards [num_reward_functions, len(completions)]
        rewards_tensor = torch.tensor(rewards, device=self.device)
        weights_tensor = torch.tensor(self.reward_weights, device=self.device)

        # [num_reward_functions, num_completions] * [num_reward_functions, 1] -> [num_completions]
        final_rewards = (rewards_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)

        return final_rewards.tolist()
