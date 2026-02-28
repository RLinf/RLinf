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

from rlinf.agents.wideseek_r1.eval_runner import WideSeekR1AgentEvalRunner as EvalRunner
from rlinf.agents.wideseek_r1.tools import WideSeekR1ToolWorker
from rlinf.agents.wideseek_r1.wideseek_r1 import WideSeekR1AgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Cluster, NodePlacementStrategy, PackedPlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start evaluation"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    # Generator group
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    if cfg.rollout.get("use_fixed_worker", False):
        # main agent and sub agent use different rollout engine
        rollout_accel_num = (
            rollout_placement_strategy._end_hw_rank
            - rollout_placement_strategy._start_hw_rank
            + 1
        )
        assert rollout_accel_num % 2 == 0
        assert (rollout_accel_num // 2) % cfg.rollout.tensor_parallel_size == 0
        assert (
            rollout_accel_num // 2
        ) % cfg.rollout_fixed_worker.tensor_parallel_size == 0
        planner_rollout_placement_strategy = PackedPlacementStrategy(
            rollout_placement_strategy._start_hw_rank,
            rollout_placement_strategy._start_hw_rank + rollout_accel_num // 2 - 1,
            num_hardware_per_process=rollout_placement_strategy._num_hardware_per_process,
            stride=rollout_placement_strategy._stride,
        )
        subwoker_rollout_placement_strategy = PackedPlacementStrategy(
            rollout_placement_strategy._start_hw_rank + rollout_accel_num // 2,
            rollout_placement_strategy._end_hw_rank,
            num_hardware_per_process=rollout_placement_strategy._num_hardware_per_process,
            stride=rollout_placement_strategy._stride,
        )
        component_placement._placements["rollout"] = planner_rollout_placement_strategy
        component_placement._rollout_num_gpus = rollout_accel_num // 2
        rollout_placement_strategy = planner_rollout_placement_strategy
        rollout_group = rollout_worker_cls.create_group(
            cfg, component_placement, weight_reload=None
        ).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=planner_rollout_placement_strategy,
        )
        subworker_rollout_group = rollout_worker_cls.create_group(
            cfg,
            component_placement,
            weight_reload=None,
            config_rollout=cfg.rollout_fixed_worker,
        ).launch(
            cluster,
            name=cfg.rollout_fixed_worker.group_name,
            placement_strategy=subwoker_rollout_placement_strategy,
        )
    else:
        # only one rollout engine
        rollout_group = rollout_worker_cls.create_group(
            cfg, component_placement, weight_reload=None
        ).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement_strategy,
        )
        subworker_rollout_group = None

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

    # Dataset
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    # Tool workers group
    if cfg.get("tools", {}).get("online") is True:
        num_tool_worker_per_node = 1
    else:
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

    runner = EvalRunner(
        cfg=cfg,
        placement=component_placement,
        val_dataset=val_ds,
        rollout=rollout_group,
        reward=None,
        agent_loop=agentloop_group,
        tool_workers=tool_workers,
        solid_rollouts={"subworker": subworker_rollout_group}
        if subworker_rollout_group is not None
        else {},
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
