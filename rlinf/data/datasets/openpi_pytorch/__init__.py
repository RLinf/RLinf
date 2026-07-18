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


def _load_behavior_sft_dataloader():
    from rlinf.data.datasets.openpi_pytorch.behavior import (
        build_behavior_sft_dataloader,
    )

    return build_behavior_sft_dataloader


# env name -> zero-arg loader returning the build_<env>_sft_dataloader function.
_SFT_DATALOADER_BUILDERS = {
    "behavior": _load_behavior_sft_dataloader,
}


def _resolve_env(config_name: str) -> str:
    """Resolve the registered env whose name appears in ``config_name``.

    Mirrors the ``if "<env>" in config_name`` dispatch the eval repack uses
    (config names look like ``pi05_behavior`` / ``pi0_libero``).
    """
    for env_type in _SFT_DATALOADER_BUILDERS:
        if env_type in config_name:
            return env_type
    raise ValueError(
        f"No openpi_pytorch SFT dataloader registered matching "
        f"config_name={config_name!r}; known envs: {list(_SFT_DATALOADER_BUILDERS)}."
    )


def build_openpi_pytorch_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    """Build the openpi_pytorch SFT dataloader for the env ``config_name`` selects.

    Returns ``(loader, data_config)`` — the same 2-tuple the SFT worker expects.
    """
    env_type = _resolve_env(str(cfg.actor.model.openpi.config_name))
    builder = _SFT_DATALOADER_BUILDERS[env_type]()
    return builder(cfg, world_size, rank, data_paths, eval_dataset)


__all__ = ["build_openpi_pytorch_sft_dataloader"]
