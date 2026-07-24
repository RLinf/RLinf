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

"""SFT dataloader for FastWAM.

Wraps the upstream ``RobotVideoDataset`` (LeRobot-format LIBERO data) in an RLinf
:class:`~torchdata.stateful_dataloader.StatefulDataLoader`. Each sample is already
in the exact dict layout consumed by ``FastWAM.training_loss`` (``video``,
``action``, ``context``, ``context_mask``, ``proprio`` and ``*_is_pad`` masks), so
the default tensor collate is sufficient.

Text embeddings must be pre-computed with FastWAM's ``scripts/precompute_text_embeds.py``
into ``text_embedding_cache_dir`` (the dataset raises if a cache entry is missing).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import default_collate
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.fastwam import _compose_fastwam_cfg, _default_fastwam_config_dir
from rlinf.utils.logging import get_logger

logger = get_logger()


def build_fastwam_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
):
    """Build a distributed FastWAM SFT dataloader.

    Args:
        cfg: full RLinf config (uses ``cfg.actor.model.fastwam`` + ``cfg.data`` +
            ``cfg.actor.micro_batch_size``).
        world_size / rank: distributed layout.
        data_paths: local path(s) to the LeRobot LIBERO dataset directory(ies).
        eval_dataset: build a (non-shuffled) eval split.

    Returns:
        (StatefulDataLoader, info_dict)
    """
    from hydra.utils import instantiate

    model_cfg = cfg.actor.model
    fw = model_cfg.get("fastwam", {}) or {}
    config_dir = fw.get("config_dir", None) or _default_fastwam_config_dir()
    config_name = fw.get("config_name", "sim_libero")
    overrides = list(fw.get("overrides", None) or [])

    fcfg = _compose_fastwam_cfg(config_dir, config_name, overrides)
    data_train = fcfg.data.train

    # Normalize dataset dir(s).
    if isinstance(data_paths, (list, tuple)):
        dataset_dirs = [str(p) for p in data_paths]
    else:
        dataset_dirs = [str(data_paths)]

    text_cache = cfg.data.get("text_embedding_cache_dir", None) or data_train.get(
        "text_embedding_cache_dir", None
    )
    # Use shipped dataset stats so we don't recompute (and don't need the full scan).
    norm_stats = cfg.data.get("pretrained_norm_stats", None) or model_cfg.get(
        "dataset_stats_path", None
    )

    # RobotVideoDataset writes a copy of the stats to misc.get_work_dir()
    # (default "./runs/", which may not exist). Point it at a real directory.
    import os as _os

    from fastwam.utils import misc as _fw_misc

    work_dir = str(cfg.runner.logger.get("log_path", "/workspace/runs"))
    _os.makedirs(work_dir, exist_ok=True)
    _fw_misc.register_work_dir(work_dir)

    overrides_kw = dict(
        dataset_dirs=dataset_dirs,
        is_training_set=not eval_dataset,
        text_embedding_cache_dir=text_cache,
    )
    if norm_stats:
        overrides_kw["pretrained_norm_stats"] = str(norm_stats)

    dataset = instantiate(data_train, **overrides_kw)
    logger.info(
        "FastWAM SFT dataset: %d samples from %s", len(dataset), dataset_dirs
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )

    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=cfg.data.get("num_workers", 2),
        collate_fn=default_collate,
        pin_memory=True,
        drop_last=not eval_dataset,
    )
    return data_loader, {"num_samples": len(dataset)}
