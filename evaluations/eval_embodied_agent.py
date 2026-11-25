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
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import EmbodiedComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.utils import (
    get_embodied_rollout_worker,
    launch_embodied_sglang_router,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="libero",
    config_name="libero_spatial_starvla_eval",
)
def main(cfg) -> None:
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = EmbodiedComponentPlacement(cfg, cluster)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_worker_cls = get_embodied_rollout_worker(cfg)
    rollout_group = rollout_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # RTC rollout and environment workers share a separate request protocol.
    env_worker_cls = EnvWorker
    if cfg.runner.get("rtc", {}).get("enabled", False):
        from rlinf.workers.env.rtc_env_worker import RTCEnvWorker

        env_worker_cls = RTCEnvWorker
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    router_group = None
    try:
        runner.init_workers()
        router_group = launch_embodied_sglang_router(cfg, cluster, rollout_group)
        runner.run()
    finally:
        if router_group is not None:
            router_group.stop().wait()
        if cfg.rollout.get("rollout_backend") == "sglang":
            rollout_group.stop().wait()


if __name__ == "__main__":
    main()
