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

"""Entry script for disaggregated async embodied VLM-as-Policy training.

Launches the unified RLinf agentic stack for multimodal embodied policies:
- ``SGLangWorker`` (server backend) + ``SGLangRouterWorker`` (``policy: manual``)
- ``SessionServerWorker`` (engine-side streaming sessions + incremental payload)
- ``EmbodiedVLMAgentLoopWorker`` + stateful ``EmbodiedEnvToolWorker``
- ``RolloutBufferWorker`` (admission, DAPO equal-reward rejection, staleness)
- ``MAFSDPActor`` (FSDP2 + chunked FP32 LM head + ``pear_reinforce``)
- ``AsyncAgentRunner``
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.agents.embodied import (
    EmbodiedEnvToolWorker,
    EmbodiedTaskDataset,
    EmbodiedVLMAgentLoopWorker,
)
from rlinf.config import validate_cfg
from rlinf.runners.async_agent_runner import AsyncAgentRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.rollout.session_server.server_worker import SessionServerWorker
from rlinf.workers.rollout.sglang.router_worker import SGLangRouterWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker
from rlinf.workers.rollout_buffer.rollout_buffer_worker import RolloutBufferWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    assert cfg.runner.get("async_mode", False), (
        "train_vlm_agent.py requires runner.async_mode=True"
    )
    assert OmegaConf.select(cfg, "session_server.enable", default=False), (
        "train_vlm_agent.py requires session_server.enable=True"
    )

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    # Rollout engines (SGLang server backend).
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )
    rollout_placements = rollout_placement_strategy.get_placement(cluster)
    rollout_node_ranks = list(
        dict.fromkeys(placement.cluster_node_rank for placement in rollout_placements)
    )

    # Embodied VLM AgentLoop workers on rollout nodes.
    agentloop_placement_strategy = NodePlacementStrategy(
        [placement.cluster_node_rank for placement in rollout_placements]
    )
    agentloop_group = EmbodiedVLMAgentLoopWorker.create_group(
        cfg, component_placement
    ).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=agentloop_placement_strategy,
    )

    # Session servers (round-robin across rollout nodes) + SGLang router.
    num_servers = int(cfg.session_server.num_servers)
    session_servers = [
        SessionServerWorker.create_group(cfg, server_index).launch(
            cluster,
            name=(
                cfg.session_server.group_name
                if num_servers == 1
                else f"{cfg.session_server.group_name}_{server_index}"
            ),
            placement_strategy=NodePlacementStrategy(
                [rollout_node_ranks[server_index % len(rollout_node_ranks)]]
            ),
        )
        for server_index in range(num_servers)
    ]
    session_router = SGLangRouterWorker.create_group(cfg, cfg.rollout.router).launch(
        cluster,
        name="SessionServerRouter",
        placement_strategy=NodePlacementStrategy([rollout_node_ranks[0]]),
    )

    # Disaggregated FSDP2 multi-agent actor.
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_group = MAFSDPActor.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement_strategy,
    )

    # Embodied episode task dataset for RolloutBuffer.
    min_dataset_size = int(cfg.data.rollout_batch_size) * int(
        cfg.algorithm.get("max_num_gen_batches", 1)
    )
    train_ds = EmbodiedTaskDataset(
        task_suite_name=str(cfg.data.get("task_suite_name", "libero_gso")),
        embodiment=str(cfg.rollout.model.get("embodiment", "libero")),
        task_id_filter=cfg.data.get("task_id_filter", None),
        specific_reset_id=cfg.data.get("specific_reset_id", None),
        min_size=min_dataset_size,
    )

    # Stateful simulator tool workers placed across rollout nodes.
    env_tool_placement = NodePlacementStrategy(rollout_node_ranks)
    tool_workers = {
        EmbodiedEnvToolWorker.create_group(cfg).launch(
            cluster,
            name="embodied_env",
            placement_strategy=env_tool_placement,
        ): ToolWorkerInfo(
            tool_names=[EmbodiedEnvToolWorker.TOOL_NAME],
            has_session=True,
        )
    }

    # RolloutBuffer on the head node owns dataloader, admission, DAPO filtering,
    # and staleness checks.
    rollout_buffer = RolloutBufferWorker.create_group(cfg, train_ds).launch(
        cluster,
        name="RolloutBuffer",
        placement_strategy=NodePlacementStrategy([0]),
    )

    runner = AsyncAgentRunner(
        cfg=cfg,
        placement=component_placement,
        train_dataset=train_ds,
        val_dataset=None,
        rollout=rollout_group,
        inference=None,
        actor=actor_group,
        reward=None,
        agent_loop=agentloop_group,
        rollout_buffer=rollout_buffer,
        tool_workers=tool_workers,
        session_servers=session_servers,
        session_router=session_router,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
