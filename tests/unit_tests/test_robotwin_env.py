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

#!/usr/bin/env python3
"""
Test the functionality of RoboTwinEnv
"""

import multiprocessing as mp
import os

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def create_test_config():
    """Create test configuration"""
    config = OmegaConf.create(
        {
            "seed": 42,
            "assets_path": "/mnt/public/guozhen/test_robotwin/RLinf_RoboTwin",
            "auto_reset": True,
            "use_rel_reward": True,
            "ignore_terminations": False,
            "num_group": 1,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "num_envs": 2,
            "max_step": 30,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "./videos",
            },
            "task_config": {
                "root_path": "/mnt/public/guozhen/test_robotwin/RLinf_RoboTwin",
                "task_name": "place_shoe",
                "render_freq": 0,
                "episode_num": 500,
                "use_seed": False,
                "save_freq": 15,
                "embodiment": ["aloha-agilex"],
                "save_path": "./data",
                "language_num": 100,
                "domain_randomization": {
                    "random_background": False,
                    "cluttered_table": False,
                    "clean_background_rate": 0,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0,
                    "random_light": False,
                    "crazy_random_light_rate": 0,
                },
                "camera": {
                    "head_camera_type": "D435",
                    "wrist_camera_type": "D435",
                    "collect_head_camera": True,
                    "collect_wrist_camera": True,
                },
                "data_type": {
                    "rgb": True,
                    "third_view": False,
                    "depth": False,
                    "pointcloud": False,
                    "endpose": True,
                    "qpos": True,
                    "mesh_segmentation": False,
                    "actor_segmentation": False,
                },
                "pcd_down_sample_num": 1024,
                "pcd_crop": True,
                "clear_cache_freq": 1,
                "collect_data": True,
                "eval_video_log": True,
            },
        }
    )
    return config


def test_robotwin_env():
    """Test basic functionality of RoboTwinEnv"""
    print("Starting RoboTwinEnv tests...")

    # Create configuration
    cfg = create_test_config()

    # Create environment
    env = RoboTwinEnv(cfg, seed_offset=0, total_num_processes=1, record_metrics=True)
    print(f"✓ Environment created successfully, num_envs: {env.num_envs}")

    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful, observation keys: {obs.keys()}")
    print(f"  Image shape: {obs['images'].shape}")
    print(f"  Wrist image shape: {obs['wrist_images'].shape}")
    print(f"  State shape: {obs['states'].shape}")

    # Test step
    print("Testing step...")
    for step in range(5):
        # Generate random actions
        actions = np.random.randn(env.num_envs, 1, 14)
        actions = np.clip(actions, 0, 1)

        obs, reward, terminated, truncated, info = env.step(actions)
        print(
            f"  Step {step}: reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, info={info}"
        )

    # Test chunk_step
    print("Testing chunk_step...")
    chunk_actions = np.random.randn(env.num_envs, 3, 14)  # 3-step chunk
    chunk_actions = np.clip(chunk_actions, 0, 1)

    obs, chunk_rewards, chunk_terminations, chunk_truncations, info = env.chunk_step(
        chunk_actions
    )
    print(f"✓ Chunk step successful, reward shape: {chunk_rewards.shape}")
    print(f"✓ Chunk step successful, info: {info}")

    # Test partial reset
    print("Testing partial reset...")
    obs, info = env.reset(env_idx=[0])  # Only reset environment 0
    print(f"✓ Partial reset successful, observation keys: {obs.keys()}")
    print(f"✓ Partial reset successful, info: {info}")

    # Cleanup
    env.clear()
    print("✓ Environment cleaned up successfully")

    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    test_robotwin_env()
