import hydra
import numpy as np
from tqdm import tqdm
import torch
from rlinf.envs.roboverse.roboverse_env import RoboVerseEnv


@hydra.main(
    version_base="1.1", config_path="examples/embodiment/config", config_name="roboverse_grpo_openvla"
)
def main(cfg) -> None:
    # print(cfg.env.train.num_envs)
    cfg.env.train.num_group = 2
    cfg.env.train.group_size = 4
    cfg.env.train.num_envs = cfg.env.train.num_group * cfg.env.train.group_size
    cfg.env.train.max_episode_steps = 32
    cfg.env.train.use_fixed_reset_state_ids = True
    env = RoboVerseEnv(cfg.env.train, rank=0, world_size=1)
    
    print("now finish flush video wait")
    a = torch.zeros((cfg.env.train.num_envs, 7))
    for i in tqdm(range(1, 30)):
        env.step(a)

        if i % 10 == 0:
            # 保存前十步的Video
            env.flush_video("test-roboverse")
            # reset
            env.seed = list(range(0, cfg.env.train.num_envs))
            env.seed = 0
            env.is_start = True
            print(i)
    env.flush_video("test-roboverse")
    

if __name__ == "__main__":
    main()
