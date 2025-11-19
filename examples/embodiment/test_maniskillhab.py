import hydra
import numpy as np
from tqdm import tqdm

from rlinf.envs.maniskillhab.maniskillhab_env import ManiskillHABEnv


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # print(cfg.env.train.num_envs)
    cfg.env.train.num_group = 2
    cfg.env.train.group_size = 4
    cfg.env.train.num_envs = cfg.env.train.num_group * cfg.env.train.group_size
    cfg.env.train.max_episode_steps = 32
    cfg.env.train.use_fixed_reset_state_ids = True
    env = ManiskillHABEnv(cfg.env.train, seed_offset=0, total_num_processes=1)
    breakpoint()
    for i in range(10):
        env.update_reset_state_ids()
        print(env.reset_state_ids)
        print(f"{env.start_idx=}")
    
    # 开始
    env.seed = list(range(0, cfg.env.train.num_envs))
    env.seed = 0
    env.is_start = True
    env.step()
    env.flush_video_wait("test-mshab-wait")  # 保存wait 10步
    print("now finish flush video wait")
    a = np.random.random((cfg.env.train.num_envs, 7))
    for i in tqdm(range(1, 30)):
        # a = np.zeros((10, 7))
        env.step(a)
        breakpoint()

        if i % 10 == 0:
            # 保存前十步的Video
            env.flush_video("test-mshab")
            # reset
            env.seed = list(range(0, cfg.env.train.num_envs))
            env.seed = 0
            env.is_start = True
            env.step()
            env.flush_video_wait("test-mshab-wait")  # 保存wait 10步
            # print(i)
    env.flush_video("test-mshab")
    

if __name__ == "__main__":
    main()
