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

import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler, default_collate
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.fastwam import (
    _compose_fastwam_cfg,
    _default_fastwam_config_dir,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class _OfficialFastWAMEpochSampler(Sampler[int]):
    """Match the upstream epoch sampler and Accelerate batch sharding.

    The upstream trainer builds one globally shuffled sample stream, forms local
    batches, and Accelerate assigns contiguous local batches to ranks. With
    even_batches enabled, a final partial batch is completed with the initial
    samples and missing rank batches are filled from the same initial stream.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        original_batches = max(1, math.ceil(len(dataset) / self.batch_size))
        self.num_batches_per_rank = max(
            1,
            original_batches // self.num_replicas
            + int(original_batches % self.num_replicas != 0),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        batches = [
            indices[offset : offset + self.batch_size]
            for offset in range(0, len(indices), self.batch_size)
        ]
        if not batches:
            raise RuntimeError("FastWAM dataset must not be empty")

        initial_data = []
        batch_to_yield = []
        batch = None
        for idx, batch in enumerate(batches):
            if idx < self.num_replicas:
                initial_data.extend(batch)
            if idx % self.num_replicas == self.rank:
                batch_to_yield = batch
            if (
                idx % self.num_replicas == self.num_replicas - 1
                and len(batch) == self.batch_size
            ):
                yield from batch_to_yield
                batch_to_yield = []

        if not initial_data:
            raise RuntimeError("FastWAM dataset must not be empty")
        while len(initial_data) < self.num_replicas * self.batch_size:
            initial_data += initial_data

        # This is the same tail-padding procedure used by Accelerate's
        # BatchSamplerShard(even_batches=True).
        if len(batch_to_yield) == self.batch_size:
            yield from batch_to_yield
        if len(batch) == self.batch_size:
            batch = []
            idx += 1
        cycle_index = 0
        while idx % self.num_replicas != 0 or len(batch) > 0:
            end_index = cycle_index + self.batch_size - len(batch)
            batch += initial_data[cycle_index:end_index]
            if idx % self.num_replicas == self.rank:
                yield from batch
            cycle_index = end_index
            batch = []
            idx += 1

    def __len__(self) -> int:
        return self.num_batches_per_rank * self.batch_size


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
    if not data_paths:
        raise ValueError("FastWAM SFT data_paths is empty; set data.train_data_paths.")
    if isinstance(data_paths, str):
        dataset_dirs = [data_paths]
    else:
        # OmegaConf supplies a ListConfig for the official four-suite list.
        dataset_dirs = [str(p) for p in list(data_paths)]

    text_cache = cfg.data.get("text_embedding_cache_dir", None) or data_train.get(
        "text_embedding_cache_dir", None
    )
    # Use shipped dataset stats so we don't recompute (and don't need the full scan).
    norm_stats = cfg.data.get("pretrained_norm_stats", None) or model_cfg.get(
        "dataset_stats_path", None
    )

    logger.info(
        "FastWAM SFT resources: data_paths=%s; data.text_embedding_cache_dir=%s; "
        "normalization_stats=%s; cwd=%s",
        dataset_dirs,
        text_cache,
        norm_stats,
        Path.cwd(),
    )
    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir).absolute()
        if not root.is_dir():
            raise FileNotFoundError(
                f"FastWAM SFT dataset directory not found: {root}. "
                "Check data.train_data_paths (or the validation data_paths); "
                "download and extract the LIBERO LeRobot dataset first."
            )
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(
                f"FastWAM SFT dataset metadata not found: {root / 'meta/info.json'}. "
                "Each data_paths entry must point to an extracted LeRobot dataset, "
                "not the archive or its parent directory."
            )
    if not text_cache:
        raise ValueError(
            "FastWAM SFT data.text_embedding_cache_dir is not set. "
            "Run FastWAM scripts/precompute_text_embeds.py and set its output directory."
        )
    cache_path = Path(text_cache).absolute()
    if not cache_path.is_dir() or not any(cache_path.glob("*.t5_len*.wan22ti2v5b.pt")):
        raise FileNotFoundError(
            f"FastWAM SFT text embeddings not found: "
            f"data.text_embedding_cache_dir={cache_path}. "
            "Expected *.t5_len*.wan22ti2v5b.pt; run FastWAM "
            "scripts/precompute_text_embeds.py for the dataset task instructions."
        )
    if norm_stats and not Path(norm_stats).is_file():
        raise FileNotFoundError(
            f"FastWAM SFT normalization statistics not found: "
            f"{Path(norm_stats).absolute()}. Check data.pretrained_norm_stats "
            "or actor.model.dataset_stats_path."
        )

    # RobotVideoDataset writes a copy of the stats to misc.get_work_dir()
    # (default "./runs/", which may not exist). Point it at a real directory.
    import os as _os

    from fastwam.utils import misc as _fw_misc
    from fastwam.utils.pytorch_utils import worker_init_function as _fw_worker_init_fn

    work_dir = str(cfg.runner.logger.get("log_path", "../results"))
    _os.makedirs(work_dir, exist_ok=True)
    _fw_misc.register_work_dir(work_dir)

    overrides_kw = {
        "dataset_dirs": dataset_dirs,
        "is_training_set": not eval_dataset,
        "text_embedding_cache_dir": text_cache,
    }
    if norm_stats:
        overrides_kw["pretrained_norm_stats"] = str(norm_stats)

    try:
        dataset = instantiate(data_train, **overrides_kw)
    except Exception:
        logger.exception(
            "FastWAM SFT dataset initialization failed: data_paths=%s; "
            "data.text_embedding_cache_dir=%s; normalization_stats=%s. "
            "See the original exception below for the failing file.",
            dataset_dirs,
            text_cache,
            norm_stats,
        )
        raise
    logger.info("FastWAM SFT dataset: %d samples from %s", len(dataset), dataset_dirs)

    sampler = _OfficialFastWAMEpochSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        batch_size=cfg.actor.micro_batch_size,
        seed=int(cfg.actor.get("seed", 42)),
        shuffle=not eval_dataset,
    )

    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=cfg.data.get("num_workers", 2),
        collate_fn=default_collate,
        pin_memory=True,
        # Match the upstream FastWAM DataLoader's worker RNG initialization.
        worker_init_fn=_fw_worker_init_fn,
        # The sampler already emits only complete, equally sized batches.
        drop_last=False,
    )
    return data_loader, {"num_samples": len(dataset)}
