import rlinf.envs.physical.franka.tasks
from rlinf.envs.physical.franka.wrappers import (
    GripperCloseEnv, 
    SpacemouseIntervention,
    RelativeFrame, 
    Quat2EulerWrapper,  
)
import gymnasium as gym
import numpy as np
import hydra
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
import copy

@hydra.main(
    version_base="1.1", config_path="config", config_name="real_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    success_needed = 10
    success_cnt = 0
    env = gym.make(
        "PegInsertionEnv-v1", 
        override_cfg={
            "robot_ip": "192.168.1.2", 
            "camera_serials": ["141722075170", ]
        }
    
    )
    
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)

    transitions = []

    env.reset()
    print("Start collecting data...")
    while success_cnt < success_needed:
        action = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
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
            total_count += 1
            print(
                f"{rew}\tGot {success_cnt} successes of {total_count} trials. {success_needed} successes needed."
            )
            # pbar.update(rew)
            obs, _ = env.reset()

    # with open(file_path, "wb") as f:
    #     pkl.dump(transitions, f)
    #     print(f"saved {success_needed} demos to {file_path}")

    env.close()
    # pbar.close()


if __name__ == "__main__":
    main()