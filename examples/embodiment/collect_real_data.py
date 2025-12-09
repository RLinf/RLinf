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

@hydra.main(
    version_base="1.1", config_path="config", config_name="real_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    success_needed = 1
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

    env.reset()
    print("Start collecting data...")
    while success_cnt < success_needed:
        action = np.zeros((6,))
        env.step(action)
    

if __name__ == "__main__":
    main()