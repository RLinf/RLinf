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

"""Evaluate a served RPent planner on fixed LIBERO tasks."""

import json

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent.eval_runner import RPentEvalRunner
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelEvalComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelEvalComponentPlacement(cfg, cluster)

    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement, weight_reload=None)
        .launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_strategy
        )
    )
    agent_loop = RPentAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
    )

    tokenizer = hf_tokenizer(cfg.rollout.model.model_path)
    _, val_ds = create_reasoning_datasets(cfg, tokenizer)
    runner = RPentEvalRunner(
        cfg=cfg,
        placement=placement,
        val_dataset=val_ds,
        rollout=rollout,
        reward=None,
        agent_loop=agent_loop,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
