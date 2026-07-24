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
    from rlinf.data.datasets.openpi_pytorch.behavior import (
        build_behavior_sft_dataloader,
    )

    return build_behavior_sft_dataloader


def _load_dual_franka_sft_dataloader() -> SftDataLoaderBuilder:
    from rlinf.data.datasets.openpi_pytorch.dual_franka import (
        build_dual_franka_sft_dataloader,
    )

    return build_dual_franka_sft_dataloader


# Environment name -> lazy SFT dataloader builder.
_SFT_DATALOADER_BUILDERS = {
    "behavior": _load_behavior_sft_dataloader,
    "dualfranka": _load_dual_franka_sft_dataloader,
}


def _resolve_env(config_name: str) -> str:
    """Resolve the registered environment named by ``config_name``."""
    for env_type in _SFT_DATALOADER_BUILDERS:
        if env_type in config_name:
            return env_type
    raise ValueError(
        f"No openpi_pytorch SFT dataloader registered matching "
        f"config_name={config_name!r}; known envs: {list(_SFT_DATALOADER_BUILDERS)}."
    )


def build_openpi_pytorch_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the environment-specific openpi_pytorch SFT dataloader."""
    env_type = _resolve_env(str(cfg.actor.model.openpi.config_name))
    builder = _SFT_DATALOADER_BUILDERS[env_type]()
    return builder(cfg, world_size, rank, data_paths, eval_dataset)


__all__ = ["build_openpi_pytorch_sft_dataloader"]
