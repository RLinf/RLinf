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

"""LingBot-VA SFT dataloader builder for the Libero suite.

Wraps lingbot-va's ``MultiLatentLeRobotDataset`` (which reads pre-extracted
video latents + cached UMT5 text embeddings) and feeds it through a standard
PyTorch ``DataLoader`` so that RLinf's ``FSDPVlaSftWorker`` can consume it
the same way it consumes the DreamZero dataset.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.utils.data
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.lingbotva._utils import extend_import_path
from rlinf.utils.logging import get_logger

logger = get_logger()


class _SynchronousMultiLatentLeRobotDataset(torch.utils.data.Dataset):
    """Drop-in replacement for ``MultiLatentLeRobotDataset`` without ``Pool``.

    The upstream class spawns a ``multiprocessing.Pool`` of workers to load
    each per-task ``LatentLeRobotDataset`` in parallel. That forks the current
    process after CUDA has already been initialized (RLinf calls
    ``torch.cuda.set_device`` early in the worker), which deadlocks. We
    construct the underlying datasets sequentially instead — at most ~10
    sub-datasets for Libero-Object, each ~1s to load.
    """

    def __init__(self, config: Any):
        from wan_va.dataset.lerobot_latent_dataset import (
            LatentLeRobotDataset,
            recursive_find_file,
        )

        info_paths = recursive_find_file(config.dataset_path, "info.json")
        if not info_paths:
            raise FileNotFoundError(
                "No LingBot-VA LeRobot dataset found under "
                f"{config.dataset_path}. Expected at least one "
                "meta/info.json."
            )
        repo_ids = [p.split("/meta/info.json")[0] for p in info_paths]
        logger.info(
            "LingBot-VA SFT dataset: loading %d LeRobot sub-dataset(s) from %s.",
            len(repo_ids),
            config.dataset_path,
        )

        self._datasets = [
            LatentLeRobotDataset(repo_id=repo_id, config=config)
            for repo_id in repo_ids
        ]
        self._item_to_dataset, self._acc = self._build_index()

    def _build_index(self):
        item_to_dataset: dict[int, int] = {}
        acc: dict[int, int] = {}
        acc_nums = [0]
        idx = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_to_dataset[idx] = dset_id
                idx += 1
        for did in range(len(self._datasets)):
            acc[did] = acc_nums[did]
        return item_to_dataset, acc

    def __len__(self) -> int:
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, idx: int) -> dict:
        assert idx < len(self), f"idx {idx} out of range for dataset len {len(self)}"
        dset_id = self._item_to_dataset[idx]
        local_idx = idx - self._acc[dset_id]
        return self._datasets[dset_id][local_idx]


def _build_job_config(cfg) -> Any:
    """Clone the lingbot-va Libero training config and patch dataset paths."""
    from wan_va.configs import VA_CONFIGS

    model_cfg = cfg.actor.model
    repo_path = model_cfg.lingbotva.repo_path
    extend_import_path(repo_path)

    config_name = getattr(model_cfg.lingbotva, "config_name", "libero")
    # The lingbot-va training script uses the ``libero_train`` variant which
    # extends the eval config with optimizer + data fields. Fall back to the
    # eval-side ``libero`` config if the train variant is missing.
    base_name = (
        f"{config_name}_train" if f"{config_name}_train" in VA_CONFIGS else config_name
    )
    job_config = copy.deepcopy(VA_CONFIGS[base_name])

    dataset_path = cfg.data.train_data_paths
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"LingBot-VA SFT dataset_path does not exist: {dataset_path}. "
            "Run the lingbot-va data-prep scripts (HDF5 → LeRobot → subset → "
            "latent extract) before launching SFT."
        )
    job_config.dataset_path = dataset_path

    empty_emb_path = model_cfg.lingbotva.get(
        "empty_emb_path", None
    ) or os.path.join(dataset_path, "empty_emb.pt")
    if not os.path.isfile(empty_emb_path):
        raise FileNotFoundError(
            f"LingBot-VA empty UMT5 embedding not found: {empty_emb_path}. "
            "Re-run scripts/extract_latents.py to regenerate it."
        )
    job_config.empty_emb_path = empty_emb_path

    job_config.cfg_prob = float(model_cfg.lingbotva.get("cfg_prob", 0.1))
    return job_config


def build_lingbotva_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str,
    eval_dataset: bool = False,
):
    """Build the LingBot-VA SFT dataloader.

    Args:
        cfg: full RLinf DictConfig.
        world_size: distributed world size.
        rank: distributed rank.
        data_paths: unused (kept for signature parity with DreamZero); we read
            ``cfg.data.train_data_paths`` instead so the prepared LeRobot
            directory comes from the SFT YAML.
        eval_dataset: when True, disables shuffling and ``drop_last``.

    Returns:
        ``(StatefulDataLoader, metadata_dict)``.
    """
    del data_paths  # we pull from cfg.data.train_data_paths

    job_config = _build_job_config(cfg)
    dataset = _SynchronousMultiLatentLeRobotDataset(job_config)
    assert len(dataset) > 0, (
        "LingBot-VA SFT dataset is empty after construction. Verify that "
        f"latents/*.pth exist under {job_config.dataset_path}/latents."
    )
    logger.info("LingBot-VA SFT dataset built with %d samples.", len(dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
        seed=int(cfg.actor.get("seed", 42)),
    )

    num_workers = int(cfg.data.get("num_workers", 4))
    prefetch_factor = cfg.data.get("prefetch_factor", 4) if num_workers > 0 else None

    loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    return loader, {"num_samples": len(dataset)}
