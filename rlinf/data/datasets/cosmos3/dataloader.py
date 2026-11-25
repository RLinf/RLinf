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

"""Cosmos3 SFT dataloader builder.

The batch structure ``OmniMoTModel.training_step`` consumes is produced by
cosmos's own ``PackingDataLoader``, using ``actor.micro_batch_size`` as its
sample-count limit.
"""

from __future__ import annotations

import contextlib

from rlinf.models.embodiment.cosmos3.cosmos3_config import (
    DATA_EXPERIMENTS,
    LIBERO_SUITES,
    build_cosmos3_config,
    cosmos_framework_root,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

_VALID_DATA_TYPES = frozenset(DATA_EXPERIMENTS)


def _detect_data_type(data_paths: str) -> str:
    """Auto-detect data_type from the dataset directory structure."""
    import os as _os

    if all(_os.path.isdir(_os.path.join(data_paths, s)) for s in LIBERO_SUITES):
        return "libero_all"
    return "libero_10"


def build_cosmos3_sft_dataloader(
    cfg,
    data_paths,
    eval_dataset: bool = False,
):
    """Build the Cosmos3 action-policy SFT dataloader."""
    from cosmos_framework.utils.lazy_config import instantiate

    if eval_dataset:
        raise NotImplementedError("Cosmos3 SFT eval dataloader is not supported yet.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    data_type = data_cfg.get("data_type", None)
    if data_type is None:
        # Auto-detect the data type from the dataset directory structure.
        data_type = _detect_data_type(str(data_paths))
        logger.info(
            f"Auto-detected data_type='{data_type}' from dataset at {data_paths}"
        )

    assert data_type in _VALID_DATA_TYPES, (
        f"Invalid data_type='{data_type}'. Valid: {sorted(_VALID_DATA_TYPES)}"
    )

    micro_batch_size = int(cfg.actor.micro_batch_size)
    if micro_batch_size < 1:
        raise ValueError("Cosmos3 SFT requires actor.micro_batch_size >= 1")
    num_workers = int(data_cfg.get("num_workers", 4))

    with contextlib.chdir(cosmos_framework_root()):
        cosmos_data_cfg = build_cosmos3_config(model_cfg, data_type, data_paths)

        dl_cfg = cosmos_data_cfg.dataloader_train
        # Packing and gradient accumulation share the same per-rank batch size.
        dl_cfg.max_samples_per_batch = micro_batch_size
        dl_cfg.dataloader.num_workers = num_workers
        if num_workers > 0:
            dl_cfg.dataloader.prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
        else:
            dl_cfg.dataloader.prefetch_factor = None
            dl_cfg.dataloader.persistent_workers = False

        loader = instantiate(dl_cfg)

    try:
        num_batches = len(loader)
    except Exception:  # noqa: BLE001 - iterable loader may not expose len
        num_batches = -1

    logger.info(
        "Cosmos3 SFT dataloader (cosmos PackingDataLoader): "
        "variant='%s' data_type='%s' -> experiment='%s' dataset=%s "
        "len=%s micro_batch_size=%s",
        model_cfg.get("model_variant", "nano"),
        data_type,
        DATA_EXPERIMENTS[data_type],
        data_paths,
        num_batches,
        micro_batch_size,
    )
    return loader, {"num_samples": num_batches if num_batches > 0 else 1}
