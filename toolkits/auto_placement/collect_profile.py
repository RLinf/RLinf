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

import hydra
import torch.multiprocessing as mp

import logging
import yaml

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from toolkits.auto_placement.util import (
    update_yaml_with_profile_data,
    has_target_env_num,
    has_profile_data,
    modify_cfg_for_profiling,
)

from toolkits.auto_placement.collector import EmbodiedCostCollector

mp.set_start_method("spawn", force=True)

@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_ppo_openvlaoft",
)
def main(cfg, step_per_env: int = 1) -> None:
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

        env_cost_dict: dict[int, float] = {}
        rollout_cost_dict: dict[int, float] = {}
        actor_cost: float = 0.0
        env_num_test_list = [ cfg.data.env_num//4, cfg.data.env_num//2, cfg.data.env_num ]

        for env_num in env_num_test_list:

            modify_cfg_for_profiling(cfg, env_num, step_per_env)

            cluster = Cluster(cluster_cfg=cfg.cluster)
            component_placement = HybridComponentPlacement(cfg, cluster)

            # Create actor worker group
            actor_placement = component_placement.get_strategy("actor")

            if cfg.algorithm.loss_type == "embodied_sac":
                from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

                actor_worker_cls = EmbodiedSACFSDPPolicy
            else:
                from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

                actor_worker_cls = EmbodiedFSDPActor

            actor_group = actor_worker_cls.create_group(cfg).launch(
                cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
            )
            # Create rollout worker group
            rollout_placement = component_placement.get_strategy("rollout")
            rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
                cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
            )
            # Create env worker group
            env_placement = component_placement.get_strategy("env")
            env_group = EnvWorker.create_group(cfg).launch(
                cluster, name=cfg.env.group_name, placement_strategy=env_placement
            )


            collector = EmbodiedCostCollector(
                cfg=cfg,
                actor=actor_group,
                rollout=rollout_group,
                env=env_group,
            )

            collector.init_workers()
            collector.collect()

            # Save the profile data for this round
            env_num_per_instance = env_num // component_placement.get_world_size("env")
            env_cost_dict[env_num_per_instance] = collector.env_cost / component_placement.get_world_size("env")
            rollout_cost_dict[env_num_per_instance] = collector.rollout_cost / component_placement.get_world_size("rollout")
            if env_num == cfg.data.env_num:
                actor_cost = collector.actor_cost
            logger.info(f"Profiled with env_num={env_num}, avg_env_cost={collector.env_cost}, avg_rollout_cost={collector.rollout_cost}, avg_actor_cost={collector.actor_cost}")

            # Close workers and channels to release resources for next round of profiling
            actor_group._close()
            rollout_group._close()
            env_group._close()
            collector.env_channel._channel_worker_group._close()
            collector.rollout_channel._channel_worker_group._close()
            collector.actor_channel._channel_worker_group._close()

        # Parse profile_results to extract the data we want to save in config        
        profile_data = {
                    "actor_cost": actor_cost,                     
                    "env_profile_data": env_cost_dict,          
                    "rollout_profile_data": rollout_cost_dict    
                }
        
        logger.info(f"Profile completed.\n{yaml.dump({'profile_data':profile_data}, default_flow_style=False, sort_keys=False)}")
        update_yaml_with_profile_data(profile_data)

    else:
        logger.info("Profile data ready.")

    logger.info("Next Step: use profile data to run the auto placement script to find the best placement strategy for your task")
    logger.info("[Usage] bash examples/embodiment/run_placement_autotune.sh <your_config_name>")

if __name__ == "__main__":
    main()
