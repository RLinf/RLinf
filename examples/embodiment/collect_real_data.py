import rlinf.envs.physical.franka.tasks
from rlinf.envs.physical.franka.wrappers import (
    GripperCloseEnv, 
    SpacemouseIntervention,
    RelativeFrame, 
    Quat2EulerWrapper,  
)
import os
import pickle as pkl
import gymnasium as gym
import numpy as np
import hydra
from rlinf.scheduler import Cluster
import copy

@hydra.main(
    version_base="1.1", config_path="config", config_name="real_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)

    success_needed = 20
    success_cnt = 0
    total_cnt = 0
    env = gym.make(
        "PegInsertionEnv-v1", 
        override_cfg={
            "robot_ip": "192.168.1.2", 
            "camera_serials": ["141722078696", ], 
        }
    
    )
    
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)

    transitions = []

    obs, _ = env.reset()
    print("Start collecting data...")
    while success_cnt < success_needed:
        action = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transitions.append(transition)

        obs = next_obs

        if done:
            success_cnt += rew
            total_cnt += 1
            print(
                f"{rew}\tGot {success_cnt} successes of {total_cnt} trials. {success_needed} successes needed."
            )
            obs, _ = env.reset()

    save_file_path =os.path.join(cfg.runner.logger.log_path, "data.pkl")
    with open(save_file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {save_file_path}")

    env.close()


if __name__ == "__main__":
    main()