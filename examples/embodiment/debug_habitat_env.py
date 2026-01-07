import os
from omegaconf.omegaconf import OmegaConf
import numpy as np
from rlinf.envs.habitat.habitat_env import HabitatEnv


def create_minimal_habitat_cfg(config_path: str, **kwargs):
    # Default minimal configuration based on libero_10_grpo_openvlaoft.yaml
    default_cfg = {
        "seed": 1,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "use_ordered_reset_state_ids": True,
        "ignore_terminations": False,
        "auto_reset": True,
        "use_rel_reward": True,
        "reward_coef": 5.0,
        "max_episode_steps": 512,
        "is_eval": True,
        "specific_reset_id": None,
        "num_gpus": 1,
        "video_cfg": {
            "save_video": True,
            "info_on_video": False,
            "video_base_dir": "private/test_videos",
            "fps": 2,
        },
        "include_depth": True,
        "include_semantic": True,
        "init_params": {
            "config_path": config_path,
        },
    }

    # Override with provided kwargs
    default_cfg.update(kwargs)

    return OmegaConf.create(default_cfg)


def test_habitat_env():
    test_config_path = os.environ.get(
        "HABITAT_CONFIG_PATH",
        "/data/RLinf/VLN-CE/config/vln_r2r.yaml",
    )
    cfg = create_minimal_habitat_cfg(
        config_path=test_config_path,
    )

    num_envs = 3
    seed_offset = 0
    total_num_processes = 1

    env = HabitatEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
    )
    env.reset()

    action_space = ["turn_left", "turn_right", "move_forward"]
    for i in range(10):
        dummy_actions = np.random.choice(action_space, size=num_envs)
        env.step(dummy_actions, auto_reset=False)
        print(f"step {i} done")

    for video_name, video_frames in env.render_images.items():
        env.flush_video(video_name, video_frames)


if __name__ == "__main__":
    test_habitat_env()
