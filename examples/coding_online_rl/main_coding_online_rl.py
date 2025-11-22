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
from rlinf.runners.coding_online_rl_runner import CodingOnlineRLRunner
from rlinf.scheduler import Cluster
from rlinf.scheduler.placement import PackedPlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.rollout.server.online_router_worker import OnlineRouterWorker
from rlinf.workers.rollout.server.server_rollout_worker import ServerRolloutWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start PPO training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    singleton_placement_strategy = PackedPlacementStrategy(
        start_hardware_rank=0, end_hardware_rank=0
    )
    online_router = OnlineRouterWorker.create_group(cfg, component_placement).launch(
        cluster=cluster,
        name="OnlineRouterWorker",
        placement_strategy=singleton_placement_strategy,
    )
    server_rollout = ServerRolloutWorker.create_group(cfg).launch(
        cluster=cluster,
        name="ServerRolloutWorker",
        placement_strategy=singleton_placement_strategy,
    )

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

    # PPO Actor group
    actor_placement_strategy = component_placement.get_strategy("actor")
    actor_group = MegatronActor.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    runner = CodingOnlineRLRunner(
        cfg=cfg,
        placement=component_placement,
        rollout=rollout_group,
        inference=inference_group,
        actor=actor_group,
        online_router=online_router,
        server_rollout=server_rollout,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
