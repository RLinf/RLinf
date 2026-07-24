# Copyright 2026 The RLinf Authors.
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

"""Entry point for STEAM value model SFT training.

Usage:
    python train_steam.py --config-path examples/offline_rl/config --config-name steam_value_model_sft
"""

import json
import logging
import os

# Quiet libav / ffmpeg before any importer triggers PyAV.
os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

import hydra  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.runners.sft_runner import SFTRunner  # noqa: E402
from rlinf.scheduler import Cluster  # noqa: E402
from rlinf.utils.placement import HybridComponentPlacement  # noqa: E402
from rlinf.workers.sft.fsdp_steam_sft_worker import (  # noqa: E402
    FSDPSteamSftWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path=None,
    config_name="steam_value_model_sft",
)
def main(cfg) -> None:
    data_root = cfg.data.get("data_root", None)
    if data_root:
        os.environ["HF_LEROBOT_HOME"] = data_root

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPSteamSftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    runner = SFTRunner(cfg=cfg, actor=actor_group)
    # 让 actor worker 初始化模型、optimizer、dataloader；如果有 resume_dir 就加载 checkpoint。
    runner.init_workers()
    #   
    runner.run()


if __name__ == "__main__":
    main()
