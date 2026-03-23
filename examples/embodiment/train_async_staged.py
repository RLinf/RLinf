# Copyright 2026 Shirui Chen
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

"""Async embodied entry point for staged YAM training with a VLM planner."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from staged_utils import launch_vlm_planner
from rlinf.config import validate_cfg
from rlinf.envs.remote.simulated_desktop import (
    launch_simulated_desktop_server,
    stop_process,
)
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="yam_async_ppo_openpi_topreward",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    try:
        cluster = Cluster(
            cluster_cfg=cfg.cluster,
            distributed_log_dir=cfg.runner.per_worker_log_path,
        )
        component_placement = HybridComponentPlacement(cfg, cluster)

        actor_placement = component_placement.get_strategy("actor")

        if cfg.algorithm.loss_type != "decoupled_actor_critic":
            raise ValueError(
                "Async staged embodied runner requires "
                "algorithm.loss_type=decoupled_actor_critic."
            )

        from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
        from rlinf.workers.actor.async_ppo_fsdp_worker import (
            AsyncPPOEmbodiedFSDPActor,
        )

        actor_group = AsyncPPOEmbodiedFSDPActor.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=actor_placement,
        )

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )

        env_placement = component_placement.get_strategy("env")
        env_group = AsyncEnvWorker.create_group(cfg).launch(
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

        vlm_actor = launch_vlm_planner(cfg, cluster)
        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        runner.run()
    finally:
        stop_process(simulated_desktop_server)


if __name__ == "__main__":
    main()
