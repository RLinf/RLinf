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
from rlinf.workers.inference.utils import get_inference_backend_worker
from rlinf.workers.reward.reward_worker import RewardWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start GRPO training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_worker_cls = get_rollout_backend_worker(cfg)
    inference_worker_cls = get_inference_backend_worker(cfg)

    # Rollout group
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )



if __name__ == "__main__":
    main()
