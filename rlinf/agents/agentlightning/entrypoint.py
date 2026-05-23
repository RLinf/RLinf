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

from __future__ import annotations

import typing
from typing import Any

if typing.TYPE_CHECKING:
    from agentlightning.adapter import TraceAdapter
    from agentlightning.store.base import LightningStore
    from agentlightning.types import Dataset

from omegaconf import DictConfig

from rlinf.config import validate_cfg


def run_rlinf_training(
    config: dict[str, Any] | DictConfig,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    adapter: TraceAdapter[Any] | None,
    eval: bool = False,
) -> None:
    """Run rlinf training and evaluation for agentlightning task."""
    from rlinf.runners.agentlightning_runner import (
        AgentLightningEvalRunner,
        AgentLightningRLinfRunner,
    )
    from rlinf.scheduler import Cluster
    from rlinf.scheduler.placement import PackedPlacementStrategy
    from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.agent.agentlightning_rollout_worker import (
        AgentLightningRolloutWorker,
    )
    from rlinf.workers.inference.utils import get_inference_backend_worker
    from rlinf.workers.rollout.utils import get_rollout_backend_worker

    cfg = config
    cfg = validate_cfg(cfg)

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    singleton_placement_strategy = PackedPlacementStrategy(
        start_hardware_rank=0, end_hardware_rank=0
    )

    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_create_kwargs = {"weight_reload": None} if eval else {}
    rollout_group = rollout_worker_cls.create_group(
        cfg, component_placement, **rollout_create_kwargs
    ).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    agentlightning_rollout_group = AgentLightningRolloutWorker.create_group(
        cfg, component_placement
    ).launch(
        cluster,
        name="AgentLightningRolloutWorker",
        placement_strategy=singleton_placement_strategy,
    )

    inference_group = None
    if (
        component_placement.placement_mode == PlacementMode.DISAGGREGATED
        and cfg.algorithm.recompute_logprobs
    ):
        inference_worker_cls = get_inference_backend_worker(cfg)
        inference_placement_strategy = component_placement.get_strategy("inference")
        inference_group = inference_worker_cls.create_group(
            cfg, component_placement
        ).launch(
            cluster,
            name=cfg.inference.group_name,
            placement_strategy=inference_placement_strategy,
        )
    actor_worker_cls = MAMegatronActor
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_group = actor_worker_cls.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    if eval:
        runner = AgentLightningEvalRunner(
            cfg=cfg,
            placement=component_placement,
            val_dataset=val_dataset,
            rollout=rollout_group,
            actor=actor_group,
            store=store,
            adapter=adapter,
            agentlightning_rollout_worker=agentlightning_rollout_group,
        )
        runner.eval()
    else:
        runner = AgentLightningRLinfRunner(
            cfg=cfg,
            placement=component_placement,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            rollout=rollout_group,
            inference=inference_group,
            actor=actor_group,
            store=store,
            adapter=adapter,
            agentlightning_rollout_worker=agentlightning_rollout_group,
        )
        runner.init_workers()
        runner.run()
