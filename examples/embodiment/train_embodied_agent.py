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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward import EmbodiedAPIRewardWorker, EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

_REWARD_SERVER_COMPONENT_NAME = "reward_server"


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "rlt_ac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for rlt_ac."
            )
        from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy

        actor_worker_cls = RLTACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        if use_training_pipeline:
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
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

    # Create reward worker group
    server_group = None
    router_group = None
    reward_group = None
    reward_cfg = cfg.get("reward", {})
    api_base = str(reward_cfg.get("api", {}).get("api_base") or "").strip()
    if (
        reward_cfg.get("use_reward_model", False)
        and str(reward_cfg.get("worker_type", "model")).lower() == "api"
        and not api_base
    ):
        from rlinf.workers.rollout.sglang_server import launch_sglang_api

        api_base, server_group, router_group = launch_sglang_api(
            config=cfg,
            cluster=cluster,
            rollout_hardware_ranks=None,
            router_server_args=cfg.router_server_args,
            placement_strategy=component_placement.get_strategy(
                _REWARD_SERVER_COMPONENT_NAME
            ),
        )
        with open_dict(cfg.reward):
            if "api" not in cfg.reward:
                cfg.reward.api = {}
            cfg.reward.api.api_base = api_base

    if reward_cfg.get("use_reward_model", False) and not reward_cfg.get(
        "standalone_realworld", False
    ):
        reward_placement = component_placement.get_strategy("reward")
        reward_worker_cls = (
            EmbodiedAPIRewardWorker
            if str(cfg.reward.get("worker_type", "model")).lower() == "api"
            else EmbodiedRewardWorker
        )
        reward_group = reward_worker_cls.create_group(cfg).launch(
            cluster,
            name=cfg.reward.group_name,
            placement_strategy=reward_placement,
        )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    runner.init_workers()
    runner.run()

    if reward_group is not None:
        reward_group.stop().wait()
    if router_group is not None:
        router_group.shutdown().wait()
    if server_group is not None:
        server_group.shutdown().wait()


if __name__ == "__main__":
    main()
