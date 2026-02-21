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

"""
Entry point for vla_lib Value Model SFT training.

Usage:
    python train_vla_lib_sft.py --config-path config --config-name libero_value_sft
"""

import json
import logging
import os

# Suppress libdav1d/ffmpeg verbose logging (must be before imports that trigger av)
os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

import hydra  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.runners.vla_lib_sft_runner import VlaLibSFTRunner  # noqa: E402
from rlinf.scheduler import Cluster  # noqa: E402
from rlinf.utils.placement import HybridComponentPlacement  # noqa: E402
from rlinf.workers.vla_lib_sft.fsdp_value_sft_worker import (  # noqa: E402
    FSDPValueSftWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="libero_value_sft")
def main(cfg) -> None:
    # Set HF_LEROBOT_HOME if data_root is provided
    data_root = cfg.data.get("data_root", None)
    if data_root:
        os.environ["HF_LEROBOT_HOME"] = data_root

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPValueSftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    runner = VlaLibSFTRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
