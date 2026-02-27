from __future__ import annotations

import json
import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from agentlightning.adapter import TraceAdapter
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.scheduler.placement import PackedPlacementStrategy
from rlinf.runners.agentlightning_runner import AgentLightningRLinfRunner
from rlinf.runners.agentlightning_eval_runner import AgentLightningEvalRunner
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import get_rollout_backend_worker
from rlinf.workers.inference.utils import get_inference_backend_worker
from rlinf.workers.actor import get_actor_worker
from rlinf.workers.agent.agentlightning_rollout_worker import AgentLightningRolloutWorker

logger = logging.getLogger(__name__)


def run_rlinf_training(
    config: dict[str, Any] | DictConfig,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    adapter: TraceAdapter[Any] | None,
    eval: bool = False,
    eval_checkpoint_dir: str | None = None,
) -> None:
    cfg = config
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    singleton_placement_strategy = PackedPlacementStrategy(
        start_hardware_rank=0, end_hardware_rank=0
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

    advantage_mode = cfg.algorithm.get("advantage_mode", "trajectory")
    if advantage_mode == "turn":
        from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

        actor_worker_cls = MAMegatronActor
    else:
        actor_worker_cls = get_actor_worker(cfg)

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
        runner.eval(checkpoint_dir=eval_checkpoint_dir)
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

