from .math import MathReward
from .vqa import VQAReward

def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class

def get_reward_class(name: str):
    assert name in reward_registry, f"Reward {name} not found"
    return reward_registry[name]

reward_registry = {}

register_reward("math", MathReward)
register_reward("vqa", VQAReward)