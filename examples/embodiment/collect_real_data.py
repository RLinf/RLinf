import rlinf.envs.physical.franka.tasks
from rlinf.envs.physical.franka.wrappers.spacemouse_intervention import SpaceMouseExpert
import gymnasium as gym
import numpy as np

def main():
    success_needed = 1
    success_cnt = 0
    env = gym.make("PegInsertionEnv-v1")
    env = SpaceMouseExpert(env)
    env.reset()

    while success_cnt < success_needed:
        action = np.zeros((6,))
        env.step(action)
    

if __name__ == "__main__":
    pass