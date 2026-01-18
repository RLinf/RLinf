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

"""Training script for ResNet Reward Model (same pattern as SFT).

Usage:
    python examples/reward/train_reward_model.py

    # Or with custom config
    python examples/reward/train_reward_model.py data.data_path=/path/to/data
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.reward.reward_worker import FSDPRewardWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="reward_training")
def main(cfg) -> None:
    print("=" * 60)
    print("ResNet Reward Model Training")
    print("=" * 60)

    # Validate config (same as SFT)
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create reward worker group (same pattern as SFT actor)
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPRewardWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Use SFTRunner to drive training
    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
