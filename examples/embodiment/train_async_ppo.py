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
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor
from rlinf.workers.env.async_ppo_env_worker import AsyncPPOEnvWorker
from rlinf.workers.rollout.hf.async_ppo_huggingface_worker import (
    AsyncPPOMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_async_ppo_openvlaoft",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor_group = AsyncPPOEmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = AsyncPPOMultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
    )

    env_placement = component_placement.get_strategy("env")
    env_group = AsyncPPOEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
