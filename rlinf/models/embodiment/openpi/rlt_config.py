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
    """RLT-token knobs from ``openpi`` and pooling from the sibling ``prefix``."""

    use_rlt: bool = False
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_prefix_seq_len: int = 768
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_use_mask: bool = False
    prefix_image_only: bool = True
    prefix_pool: str | None = None


# Former OpenPI-level pooling fields and the ``prefix`` field replacing each.
_LEGACY_PREFIX_FIELDS = {
    "stage2_z_source": "prefix.pool",
    "prefix_pool": "prefix.pool",
    "rlt_image_only": "prefix.image_only",
}


def build_rlt_config(model_cfg: Any, *, parent_cfg: Any) -> OpenPiPytorchRLTConfig:
    """Build RLT and Prefix-FT settings from an OpenPI model config.

    Args:
        model_cfg: The ``openpi`` sub-config holding the RLT-token fields.
        parent_cfg: The model config that contains ``model_cfg`` as ``openpi``
            and owns ``prefix.pool`` and ``prefix.image_only``.
    """
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.prefix.config import resolve_prefix_pool

    for legacy_key, replacement in _LEGACY_PREFIX_FIELDS.items():
        if OmegaConf.select(model_cfg, legacy_key, default=None) is not None:
            raise ValueError(
                f"openpi.{legacy_key} is no longer supported; set {replacement} "
                "beside the openpi block instead."
            )
    use_rlt = bool(OmegaConf.select(model_cfg, "use_rlt", default=False))
    prefix_pool = OmegaConf.select(parent_cfg, "prefix.pool", default=None)
    image_only = OmegaConf.select(parent_cfg, "prefix.image_only", default=True)

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
        rlt_use_mask=bool(OmegaConf.select(model_cfg, "rlt_use_mask", default=False)),
        prefix_image_only=bool(image_only),
        prefix_pool=resolve_prefix_pool(
            use_rlt=use_rlt,
            prefix_pool=str(prefix_pool) if prefix_pool is not None else None,
        ),
    )
