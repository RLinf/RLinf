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
from omegaconf.omegaconf import OmegaConf

from rlinf.agents.wideseek_r1.tools import WideSeekR1ToolWorker
from rlinf.agents.wideseek_r1.wideseek_r1 import WideSeekR1AgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start GRPO training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)
    assert component_placement.placement_mode == PlacementMode.COLLOCATED, (
        "multi-agent only supports collocated mode"
    )

    # Generator group
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    assert not cfg.rollout.get("use_fixed_worker", False), (
        "Currently we only support two engine in evluation"
    )

    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    # AgentLoop group.
    agentloop_placement_strategy = NodePlacementStrategy(
        [
            placement.cluster_node_rank
            for placement in rollout_placement_strategy.get_placement(cluster)
        ]
    )
    assert (
        len(agentloop_placement_strategy._node_ranks)
        == component_placement.rollout_dp_size
    ), (
        f"agentloop worker num {len(agentloop_placement_strategy._node_ranks)} now should be equal to rollout dp size {component_placement.rollout_dp_size}"
    )

    agentloop_group = WideSeekR1AgentLoopWorker.create_group(
        cfg, component_placement
    ).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=agentloop_placement_strategy,
    )

    # GRPO Actor group
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_group = MAMegatronActor.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    # Dataset
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    # Tool workers group
    num_tool_worker_per_node = 32
    tool_placement = [
        node_id
        for node_id in range(cfg.cluster.num_nodes)
        for _ in range(num_tool_worker_per_node)
    ]
    singleton_tool_placement = NodePlacementStrategy(tool_placement)
    tool_workers = {
        WideSeekR1ToolWorker.create_group(cfg).launch(
            cluster, name="search", placement_strategy=singleton_tool_placement
        ): ToolWorkerInfo(tool_names=["search"], has_session=False),
        WideSeekR1ToolWorker.create_group(cfg).launch(
            cluster, name="access", placement_strategy=singleton_tool_placement
        ): ToolWorkerInfo(tool_names=["access"], has_session=False),
    }

    runner = AgentRunner(
        cfg=cfg,
        placement=component_placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout_group,
        actor=actor_group,
        agent_loop=agentloop_group,
        tool_workers=tool_workers,
        solid_rollouts={},
        inference=None,
        reward=None,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
