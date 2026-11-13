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

from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from rlinf.algorithms.distillation.config import build_distillation_plan
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.scheduler.worker.worker_group import WorkerGroup
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
    PlacementMode,
    TeacherComponentPlacement,
)
from rlinf.workers.actor import get_ma_actor_worker
from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.inference.utils import get_inference_backend_worker
from rlinf.workers.rollout.utils import get_rollout_backend_worker
from rlinf.workers.teacher.runtime import launch_teacher_workers


def run_agent_training(
    cfg: DictConfig,
    agent_loop_cls: type[MultiAgentLoopWorker],
    launch_tools: Callable[[DictConfig, Cluster], dict[WorkerGroup, ToolWorkerInfo]]
    | None = None,
    *,
    create_datasets: Callable[
        [DictConfig, PreTrainedTokenizerBase], tuple[Dataset | None, Dataset | None]
    ] = create_reasoning_datasets,
) -> None:
    """Launch a multi-turn policy, its rollout backend and optional tool workers."""
    cfg = validate_cfg(cfg)
    get_logger().info(
        "Training configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True)
    )
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    # Generator group
    rollout_worker_cls = get_rollout_backend_worker(cfg)
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
        == component_placement.rollout_num_worker_processes
    ), "Expected one agent loop worker per rollout process"
    agentloop_group = agent_loop_cls.create_group(cfg, component_placement).launch(
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
        inference_worker_cls = get_inference_backend_worker(cfg, "actor")
        inference_group = inference_worker_cls.create_group(
            cfg, component_placement
        ).launch(
            cluster,
            name=cfg.inference.group_name,
            placement_strategy=inference_placement_strategy,
        )

    # GRPO Actor group
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_worker_cls = get_ma_actor_worker(cfg)
    actor_group = actor_worker_cls.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    # Dataset
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_datasets(cfg, tokenizer)

    tool_workers = launch_tools(cfg, cluster) if launch_tools else {}

    teacher_runtime = None
    if build_distillation_plan(cfg).enabled:
        teacher_placement = TeacherComponentPlacement(cfg, cluster)
        teacher_placement.validate_disjoint(component_placement)
        teacher_runtime = launch_teacher_workers(cfg, cluster, teacher_placement)

    runner = None
    try:
        if cfg.runner.get("async_mode", False):
            from rlinf.runners.async_agent_runner import AsyncAgentRunner
            from rlinf.workers.rollout_buffer.rollout_buffer_worker import (
                RolloutBufferWorker,
            )

            runner_cls = AsyncAgentRunner
            runner_kwargs = {
                "rollout_buffer": RolloutBufferWorker.create_group(
                    cfg, train_ds
                ).launch(
                    cluster,
                    name="RolloutBuffer",
                    placement_strategy=NodePlacementStrategy([0]),
                )
            }
        else:
            runner_cls = AgentRunner
            runner_kwargs = {"teacher_runtime": teacher_runtime}

        runner = runner_cls(
            cfg=cfg,
            placement=component_placement,
            train_dataset=train_ds,
            val_dataset=val_ds,
            rollout=rollout_group,
            inference=inference_group,
            actor=actor_group,
            reward=None,
            agent_loop=agentloop_group,
            tool_workers=tool_workers,
            **runner_kwargs,
        )
        runner.init_workers()
        runner.run()
    finally:
        if teacher_runtime is not None:
            try:
                if runner is not None:
                    runner.close_teacher_clients()
            finally:
                teacher_runtime.close()
