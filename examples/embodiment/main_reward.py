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
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.reward.reward_worker import RewardWorker

"""Script to start reward model training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="train_reward_model")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Reward group
    reward_placement_strategy = component_placement.get_strategy("reward")
    reward_group = RewardWorker.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.reward.group_name,
        placement_strategy=reward_placement_strategy,
    )

    # Initialize workers
    print("Initializing workers...", flush=True)
    reward_group.init_worker().wait()
    print("Workers initialized. Starting training...", flush=True)

    # Start training
    reward_group.fit().wait()


if __name__ == "__main__":
    main()
