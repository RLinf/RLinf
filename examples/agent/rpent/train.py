# Copyright 2026 The RLinf Authors.
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

"""Train a text-only RPent planner with synchronous PPO."""

import json

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.agent_ppo_runner import AgentPPORunner
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic
from rlinf.workers.rollout.utils import get_rollout_backend_worker


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_strategy
        )
    )
    agent_loop = RPentAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
    )
    actor = MAMegatronActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    critic = MAMegatronCritic.create_group(cfg, placement).launch(
        cluster,
        name=cfg.critic.group_name,
        placement_strategy=placement.get_strategy("critic"),
    )

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)
    runner = AgentPPORunner(
        cfg=cfg,
        placement=placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent_loop,
        critic=critic,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
