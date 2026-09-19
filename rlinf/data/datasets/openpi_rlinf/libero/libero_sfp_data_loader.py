# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO SFT data loader for Streaming Flow Policy training."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.training import data_loader as openpi_data_loader
from openpi.training.config import DataConfig, TrainConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    validate_openpi_rlinf_model_shape,
)
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.dataconfig.sfp_transforms import SfpNormalize


def _collate(items: list[dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor]:
    """Stack transformed samples into the ``(observation, actions)`` SFT batch."""
    batch = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *items,
    )
    batch = jax.tree.map(torch.as_tensor, batch)
    return batch, batch["actions"]


def _transform_dataset(dataset: Any, data_config: DataConfig) -> Any:
    """Apply the SFP pipeline.

    This is :func:`openpi.training.data_loader.transform_dataset` with one step
    replaced: SFP normalizes by scale alone so that summing normalized action
    deltas reproduces the normalized trajectory. See
    :class:`~rlinf.models.embodiment.openpi.dataconfig.sfp_transforms.SfpNormalize`.
    """
    if data_config.norm_stats is None:
        raise ValueError(
            "SFP training needs normalization statistics. Compute them with "
            "toolkits/lerobot/calculate_norm_stats.py and set "
            "actor.model.openpi_data.norm_stats_path to the result."
        )
    return openpi_data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            SfpNormalize(
                data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def build_libero_sfp_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[StatefulDataLoader, DataConfig]:
    """Build the LIBERO SFP loader for the openpi_rlinf SFT worker.

    Yields ``(observation, actions)`` batches whose observation carries the
    ``action_states`` field :meth:`~rlinf.models.embodiment.openpi_rlinf.pi0.Pi0.compute_sfp_loss`
    integrates the action trajectory from. The loader is stateful so the SFT
    worker can save and restore its position with the rest of a checkpoint.
    """
    if eval_dataset:
        raise NotImplementedError(
            "LIBERO SFP validation is not implemented; set "
            "runner.val_check_interval to -1."
        )

    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "LIBERO SFP training requires data.train_data_paths to be set to a "
            "local dataset path or LeRobot repo id."
        )

    model_cfg = cfg.actor.model
    config: TrainConfig = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=int(cfg.actor.micro_batch_size) * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    seed = int(OmegaConf.select(cfg, "actor.seed", default=config.seed))
    num_workers = int(
        OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
    )
    config = dataclasses.replace(config, num_workers=num_workers, seed=seed)
    validate_openpi_rlinf_model_shape(model_cfg, config)

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    dataset = _transform_dataset(dataset, data_config)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=int(cfg.actor.micro_batch_size),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    return data_loader, data_config
