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

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import torch
from openpi.training import data_loader as openpi_data_loader
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.streamingvla.dataconfig import get_streamingvla_config
from rlinf.models.embodiment.streamingvla.observation import StreamingVLAObservation
from rlinf.models.embodiment.streamingvla.transforms import StreamingVLANormalize


def _collate_streamingvla_batch(
    items: list[dict[str, Any]],
) -> tuple[StreamingVLAObservation, torch.Tensor]:
    """Stack transformed samples into a StreamingVLA training batch."""
    batch = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *items,
    )
    batch = jax.tree.map(torch.as_tensor, batch)
    return StreamingVLAObservation.from_dict(batch), batch["actions"]


def _transform_dataset(dataset: Any, data_config: Any) -> Any:
    """Apply StreamingVLA-only transforms without modifying OpenPI globals."""
    norm_stats = {}
    if data_config.repo_id != "fake":
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found for StreamingVLA data.")
        norm_stats = data_config.norm_stats
    return openpi_data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            StreamingVLANormalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def create_streamingvla_data_loader(
    config: Any,
    *,
    framework: str = "pytorch",
    shuffle: bool = True,
) -> StatefulDataLoader:
    """Create the stateful PyTorch loader used by StreamingVLA SFT."""
    if framework != "pytorch":
        raise ValueError("StreamingVLA's RLinf data loader only supports PyTorch.")

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    if bool(getattr(config.data, "use_action_states", False)):
        sample = dataset[0]
        if "action_states" not in sample:
            raise KeyError(
                "StreamingVLA requires the real episode field "
                "'action_states'; it was absent from sample 0."
            )
    dataset = _transform_dataset(dataset, data_config)

    sampler = None
    if framework == "pytorch" and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            seed=int(config.seed),
            drop_last=True,
        )
        local_batch_size = config.batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = config.batch_size

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=_collate_streamingvla_batch,
        drop_last=True,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    return data_loader


def build_streamingvla_dataloader(
    cfg: Any, world_size: int
) -> tuple[StatefulDataLoader, Any]:
    """Build a StreamingVLA loader through RLinf's VLA worker hook."""
    config = get_streamingvla_config(
        cfg.actor.model.streamingvla.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=cfg.actor.micro_batch_size * world_size,
        data_kwargs=cfg.actor.model.streamingvla.get("data", None),
        seed=cfg.actor.seed,
    )
    return create_streamingvla_data_loader(
        config, framework="pytorch", shuffle=True
    ), config.data.create(config.assets_dirs, config.model)
