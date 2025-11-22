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

from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor import get_actor_worker
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

    rollout_worker_cls = get_rollout_backend_worker(cfg, component_placement)

    # Rollout group
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    # Inference group
    inference_group = None
    if (
        component_placement.placement_mode
        in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
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

    # Dynamic Scheduler group
    if component_placement._placement_mode == PlacementMode.AUTO:
        scheduler_placement_strategy = NodePlacementStrategy(node_ranks=[0])
        scheduler = SchedulerWorker.create_group(cfg, component_placement).launch(
            cluster=cluster,
            name="DynamicScheduler",
            placement_strategy=scheduler_placement_strategy,
        )
    else:
        scheduler = None

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    runner = ReasoningRunner(
        cfg=cfg,
        placement=component_placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout_group,
        inference=inference_group,
        actor=actor_group,
        reward=reward_group,
        scheduler=scheduler,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
