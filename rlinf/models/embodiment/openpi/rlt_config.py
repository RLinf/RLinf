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

"""RLT-token sidecar config used by SFT ``Pi0`` and Stage2 ``Pi0Eval``.

Not a ``tasks/`` entry: YAML is still ``task: sft|eval`` plus ``use_rlt``.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchRLTConfig:
    """RLT-token knobs from ``actor.model.openpi``."""

    use_rlt: bool = False
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_prefix_seq_len: int = 768
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = True
    rlt_use_mask: bool = False
    stage2_z_source: str = "rlt_token"
    prefix_pool: str | None = None


def build_rlt_config(
    model_cfg: Any, *, parent_cfg: Any | None = None
) -> OpenPiPytorchRLTConfig:
    """Build RLT and Prefix-FT settings from an OpenPI model config."""
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.prefix.config import resolve_prefix_pool

    parent = parent_cfg if parent_cfg is not None else model_cfg
    use_rlt = bool(OmegaConf.select(model_cfg, "use_rlt", default=False))
    rlt_use_mask = bool(OmegaConf.select(model_cfg, "rlt_use_mask", default=False))
    stage2_z_source = str(
        OmegaConf.select(model_cfg, "stage2_z_source", default="rlt_token")
    )
    prefix_pool = OmegaConf.select(parent, "prefix.pool", default=None)
    if prefix_pool is None:
        prefix_pool = OmegaConf.select(model_cfg, "prefix_pool", default=None)
    image_only = OmegaConf.select(parent, "prefix.image_only", default=None)
    if image_only is None:
        image_only = OmegaConf.select(model_cfg, "rlt_image_only", default=True)

    return OpenPiPytorchRLTConfig(
        use_rlt=use_rlt,
        rlt_alpha=float(OmegaConf.select(model_cfg, "rlt_alpha", default=1.0)),
        rlt_input_dim=int(OmegaConf.select(model_cfg, "rlt_input_dim", default=2048)),
        rlt_embed_dim=int(OmegaConf.select(model_cfg, "rlt_embed_dim", default=2048)),
        rlt_prefix_seq_len=int(
            OmegaConf.select(model_cfg, "rlt_prefix_seq_len", default=768)
        ),
        rlt_num_layers=int(OmegaConf.select(model_cfg, "rlt_num_layers", default=2)),
        rlt_num_heads=int(OmegaConf.select(model_cfg, "rlt_num_heads", default=8)),
        rlt_mlp_ratio=float(OmegaConf.select(model_cfg, "rlt_mlp_ratio", default=4.0)),
        rlt_image_only=bool(image_only),
        rlt_use_mask=rlt_use_mask,
        stage2_z_source=stage2_z_source,
        prefix_pool=resolve_prefix_pool(
            use_rlt=use_rlt,
            prefix_pool=str(prefix_pool) if prefix_pool is not None else None,
            stage2_z_source=stage2_z_source,
            rlt_use_mask=rlt_use_mask,
        ),
    )
