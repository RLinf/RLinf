import torch
from typing import List
from omegaconf import DictConfig
from .qa_rewards import qa_accuracy_reward
from .format_rewards import think_format_reward, answer_format_reward


class VQAReward:
    def __init__(self, config: DictConfig):
        self.reward_weights = config.get("reward_weights", {
            "qa_accuracy": 1.0,
            "think_format": 0.0,
            "answer_format": 0.0,
        })
        for reward_name, reward_weight in self.reward_weights.items():
            assert reward_name in ["qa_accuracy", "think_format", "answer_format"], f"Reward {reward_name} not supported"
            assert reward_weight >= 0, f"Reward weight {reward_weight} must be non-negative"
        self.reward_weights = [reward_weight["qa_accuracy"], reward_weight["think_format"], reward_weight["answer_format"]]

        self.reward_functions = [qa_accuracy_reward, think_format_reward, answer_format_reward]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_reward(self, completions: List[str], answers: List[str]) -> List[float]:
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
        