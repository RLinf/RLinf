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

import os

import numpy as np
from omegaconf.omegaconf import OmegaConf

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
        "max_episode_steps": 30,
        "num_action_chunks": 3,
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
        "examples/embodiment/config/habitat_vlnce_r2r.yaml",
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

    n_chunk_steps = cfg.max_episode_steps // cfg.num_action_chunks
    action_space = ["turn_left", "turn_right", "move_forward"]

    for i in range(n_chunk_steps):
        dummy_actions = np.random.choice(
            action_space, size=(num_envs, cfg.num_action_chunks)
        )
        env.chunk_step(dummy_actions)
        print(f"step {i} done")

    for video_name, video_frames in env.render_images.items():
        env.flush_video(video_name, video_frames)


if __name__ == "__main__":
    test_habitat_env()
