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

"""
Run profile script for embodied profiling: runs a few steps and collects profile data.

"""

import logging

import hydra
import torch.multiprocessing as mp

from examples.embodiment.train_embodied_agent import train
from rlinf.config import validate_cfg
from toolkits.auto_placement.util import (
    extract_from_tensorboard,
    has_profile_data,
    has_target_env_num,
    log_env_num_instavg_to_tensorboard,
    modify_cfg_for_profiling_with_group,
    update_yaml_with_profile_data,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_ppo_openvlaoft",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)

    logger = logging.getLogger("CollectProfile")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s %(asctime)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not has_target_env_num(cfg):
        logger.info("Target env num not found in config. Please provide it .")
        return None

    elif not has_profile_data(cfg):
        logger.info("No profile data found in config. Starting profiling...")
        env_num_test_list = [
            cfg.data.env_num,
            cfg.data.env_num,
            cfg.data.env_num,
        ]

        for run_idx, env_num in enumerate(env_num_test_list):
            modify_cfg_for_profiling_with_group(cfg, env_num, run_idx)

            runner, actor_group, rollout_group, env_group, component_placement = train(
                cfg
            )

            env_num_per_instance = env_num / component_placement.get_world_size("env")

            log_env_num_instavg_to_tensorboard(cfg, env_num_per_instance)

            # actor_group._close()
            # rollout_group._close()
            # env_group._close()
            # runner.env_channel._channel_worker_group._close()
            # runner.rollout_channel._channel_worker_group._close()
            # runner.actor_channel._channel_worker_group._close()

        profile_data = extract_from_tensorboard()
        update_yaml_with_profile_data(profile_data)

    else:
        logger.info("Profile data ready.")


if __name__ == "__main__":
    main()
