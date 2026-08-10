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

from __future__ import annotations

from typing import Any, Callable

SftDataLoaderBuilder = Callable[..., tuple[Any, Any]]


def _load_behavior_sft_dataloader() -> SftDataLoaderBuilder:
    from rlinf.data.datasets.openpi_rlinf.behavior import (
        build_behavior_sft_dataloader,
    )

    return build_behavior_sft_dataloader


def _load_dual_franka_sft_dataloader() -> SftDataLoaderBuilder:
    from rlinf.data.datasets.openpi_rlinf.dual_franka import (
        build_dual_franka_sft_dataloader,
    )

    return build_dual_franka_sft_dataloader


def _load_official_openpi_sft_dataloader() -> SftDataLoaderBuilder:
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )

    return build_official_openpi_sft_dataloader


# Environment name -> lazy SFT dataloader builder.
_SFT_DATALOADER_BUILDERS = {
    "behavior": _load_behavior_sft_dataloader,
    "dualfranka": _load_dual_franka_sft_dataloader,
    "robotwin": _load_official_openpi_sft_dataloader,
}


def _resolve_env(config_name: str) -> str:
    """Resolve the registered environment named by ``config_name``."""
    for env_type in _SFT_DATALOADER_BUILDERS:
        if env_type in config_name:
            return env_type
    raise ValueError(
        f"No openpi_rlinf SFT dataloader registered matching "
        f"config_name={config_name!r}; known envs: {list(_SFT_DATALOADER_BUILDERS)}."
    )


def build_openpi_rlinf_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the environment-specific openpi_rlinf SFT dataloader."""
    if bool(cfg.actor.model.openpi.get("use_rlt", False)):
        return _load_official_openpi_sft_dataloader()(
            cfg, world_size, rank, data_paths, eval_dataset
        )

    env_type = _resolve_env(str(cfg.actor.model.openpi.config_name))
    builder = _SFT_DATALOADER_BUILDERS[env_type]()
    return builder(cfg, world_size, rank, data_paths, eval_dataset)


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the official OpenPI loader for the legacy OpenPI model type."""
    return _load_official_openpi_sft_dataloader()(
        cfg, world_size, rank, data_paths, eval_dataset
    )


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the batch count of an official OpenPI SFT loader."""
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_official_openpi_sft_num_batches as _get_num_batches,
    )

    return _get_num_batches(data_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether the loader was created by the official OpenPI path."""
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        is_official_openpi_sft_dataloader as _is_official,
    )

    return _is_official(data_loader)


__all__ = [
    "build_official_openpi_sft_dataloader",
    "build_openpi_rlinf_sft_dataloader",
    "get_official_openpi_sft_num_batches",
    "is_official_openpi_sft_dataloader",
]
