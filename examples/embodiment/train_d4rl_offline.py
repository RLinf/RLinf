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
from omegaconf.omegaconf import OmegaConf

from rlinf.runners.d4rl_offline_runner import D4RLOfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_iql_policy_worker import EmbodiedIQLFSDPPolicy


@hydra.main(
    version_base="1.1", config_path="config", config_name="d4rl_offline_mujoco"
)
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedIQLFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    runner = D4RLOfflineRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()

