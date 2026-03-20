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

import os
from dataclasses import dataclass
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor

from rlinf.models.embodiment.lingbotvla.data.vla_data.base_dataset import (
    RobotwinDataset,
)


@dataclass
class LingbotDataConfig:
    norm_type: Literal[
        "meanstd", "bounds_99", "bounds_98", "bounds_98_woclip", "bounds_99_woclip"
    ] = "bounds_99_woclip"
    img_size: int = 224
    norm_stats_file: str = ""


def build_lingbot_sft_dataloader(cfg, world_size, global_rank, data_paths):
    lingbotvla_cfg = getattr(
        cfg.actor.model,
        "lingbotvla",
        getattr(cfg.actor.model, "lingbot", cfg.actor.model),
    )
    config_path = getattr(
        lingbotvla_cfg,
        "config_path",
        os.path.join(os.environ.get("LINGBOT_VLA_PATH", ""), "lingbot-vla-4b"),
    )

    dataset_qwen_config = PreTrainedConfig.from_pretrained(config_path)
    dataset_qwen_config.n_action_steps = cfg.actor.model.num_action_chunks

    processor = AutoProcessor.from_pretrained(cfg.actor.model.tokenizer_path)

    stats_path = getattr(
        lingbotvla_cfg,
        "stats_path",
        os.path.join(
            os.environ.get("LINGBOT_VLA_PATH", ""), "assets/norm_stats/robotwin_50.json"
        ),
    )
    data_config = LingbotDataConfig(norm_stats_file=stats_path)

    repo_id = data_paths if isinstance(data_paths, str) else data_paths[0]
    dataset = RobotwinDataset(
        repo_id=repo_id,
        config=dataset_qwen_config,
        tokenizer=processor.tokenizer,
        data_config=data_config,
        image_processor=processor.image_processor,
        use_depth_align=False,
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return data_loader, {}
