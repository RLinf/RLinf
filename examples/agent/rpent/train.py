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

"""Train an RPent planner with synchronous or asynchronous GRPO."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.agents.rpent.rollout_buffer import (
    RPentCohortBuffer,
    RPentRolloutBufferWorker,
)
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.agents.rpent.session_server import RPentSessionServerWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    from rlinf.agents.rpent.train_runner import RPentRunner
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    cfg = validate_cfg(cfg)
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_strategy,
        )
    )
    agent_loop = RPentAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
        isolate_gpu=False,
    )
    actor = MAMegatronActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    # The session captures images beside the one rollout instance. Node-wide
    # placement could initialize a CUDA context on an unrelated default GPU.
    servers = [
        RPentSessionServerWorker.create_group(cfg, server_index).launch(
            cluster,
            name=f"{cfg.session_server.group_name}_{server_index}",
            placement_strategy=rollout_strategy,
        )
        for server_index in range(cfg.session_server.num_servers)
    ]
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)
    buffer_cls = (
        RPentCohortBuffer if cfg.runner.synchronous else RPentRolloutBufferWorker
    )
    rollout_buffer = buffer_cls.create_group(cfg, train_ds).launch(
        cluster, name="RolloutBuffer", placement_strategy=rollout_strategy
    )
    runner = RPentRunner(
        rollout_buffer=rollout_buffer,
        session_servers=servers,
        session_router=None,
        cfg=cfg,
        placement=placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent_loop,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
