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

from rlinf.agents.multiturn_demo.mcp_agent_loop import MCPAgentLoopWorker
from rlinf.agents.multiturn_demo.mcp_filesystem_worker import MCPFilesystemClientWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor import get_actor_worker
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.reward.reward_worker import RewardWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start GRPO training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    # Generator group
    rollout_worker_cls = get_rollout_backend_worker(cfg, component_placement)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
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
    ), "agentloop worker num now should be equal to rollout dp size"
    agentloop_group = MCPAgentLoopWorker.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=agentloop_placement_strategy,
    )

    # Inference group
    inference_group = None
    if (
        component_placement.placement_mode == PlacementMode.DISAGGREGATED
        and cfg.algorithm.recompute_logprobs
    ):
        inference_placement_strategy = component_placement.get_strategy("inference")
        inference_group = MegatronInference.create_group(
            cfg, component_placement
        ).launch(
            cluster,
            name=cfg.inference.group_name,
            placement_strategy=inference_placement_strategy,
        )

    # Reward group
    reward_placement_strategy = component_placement.get_strategy("reward")
    reward_group = RewardWorker.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.reward.group_name,
        placement_strategy=reward_placement_strategy,
    )

    # GRPO Actor group
    actor_worker_cls = get_actor_worker(cfg)
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_group = actor_worker_cls.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    # Dataset
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    # Tool workers group
    singleton_tool_placement = NodePlacementStrategy([0])
    tool_workers = {
        # FakeToolWorker.create_group(cfg).launch(
        #     cluster, name="Tool1", placement_strategy=singleton_tool_placement
        # ): ToolWorkerInfo(tool_names=["tool1"], has_session=False),
        MCPFilesystemClientWorker.create_group(cfg).launch(
            cluster,
            name="MCPFilesystemClient",
            placement_strategy=singleton_tool_placement,
        ): ToolWorkerInfo(
            tool_names=["write_file", "list_directory"], has_session=True
        ),
    }

    runner = AgentRunner(
        cfg=cfg,
        placement=component_placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout_group,
        inference=inference_group,
        actor=actor_group,
        reward=reward_group,
        agent_loop=agentloop_group,
        tool_workers=tool_workers,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
