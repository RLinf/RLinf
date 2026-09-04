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

from typing import Any

from rlinf.models.embodiment.prefix_ft.history import StateHistoryBuffer
from rlinf.models.embodiment.prefix_ft.pool import PREFIX_POOL_MODES

PREFIX_AC_LOSS_TYPES = frozenset({"rlt_ac", "prefix_ac"})
RLT_ENV_LOSS_TYPES = PREFIX_AC_LOSS_TYPES | {"rlt_td3"}


def is_prefix_ac_loss(loss_type: Any) -> bool:
    return str(loss_type) in PREFIX_AC_LOSS_TYPES


def is_rlt_env_loss(loss_type: Any) -> bool:
    return str(loss_type) in RLT_ENV_LOSS_TYPES


def resolve_prefix_pool(
    *,
    use_rlt: bool = False,
    prefix_pool: str | None = None,
    stage2_z_source: str | None = None,
    rlt_use_mask: bool = False,
) -> str:
    """Resolve the Stage2 z source.

    Explicit ``prefix.pool`` wins. Otherwise ``stage2_z_source=vlm_prefix``
    maps to masked/mean pooling (preserving the old ablation), ``use_rlt=True``
    keeps ``rlt_token``, and Prefix-FT (``use_rlt=False``) defaults to
    ``masked_mean``.
    """
    if prefix_pool:
        pool = str(prefix_pool)
        if pool not in PREFIX_POOL_MODES:
            raise ValueError(
                "prefix.pool must be one of "
                f"{PREFIX_POOL_MODES}, got {pool!r}."
            )
        return pool
    if stage2_z_source == "vlm_prefix":
        return "masked_mean" if rlt_use_mask else "mean"
    if use_rlt:
        return "rlt_token"
    return "masked_mean"


def resolve_prefix_feature_model_config(cfg: Any) -> Any | None:
    """``rollout.prefix_feature_model`` falls back to ``rollout.rlt_feature_model``."""
    from omegaconf import OmegaConf

    feature = OmegaConf.select(cfg, "rollout.prefix_feature_model", default=None)
    if feature is None:
        feature = OmegaConf.select(cfg, "rollout.rlt_feature_model", default=None)
    return feature


def extra_z_dim_from_cfg(cfg: Any) -> int:
    """History extra width. YAML ``actor.model.z_dim`` stays the prefix width."""
    from omegaconf import OmegaConf

    hist = OmegaConf.select(cfg, "algorithm.state_history", default=None)
    if hist is None:
        hist = OmegaConf.select(cfg, "state_history", default=None)
    if hist is None:
        return 0
    if not bool(OmegaConf.select(hist, "enable", default=False)):
        return 0
    steps = int(OmegaConf.select(hist, "steps", default=4) or 4)
    proprio_dim = OmegaConf.select(cfg, "actor.model.proprio_dim", default=None)
    if proprio_dim is None:
        proprio_dim = OmegaConf.select(cfg, "proprio_dim", default=0)
    return steps * int(proprio_dim or 0)


def apply_prefix_head_z_dim(model_cfg: Any, full_cfg: Any) -> Any:
    """Expand ``model_cfg.z_dim`` by history extra width once, in place."""
    extra = extra_z_dim_from_cfg(full_cfg)
    if extra <= 0 or model_cfg is None:
        return model_cfg
    from omegaconf import OmegaConf, open_dict

    if OmegaConf.select(model_cfg, "z_dim", default=None) is None:
        return model_cfg
    if bool(OmegaConf.select(model_cfg, "_prefix_z_dim_expanded", default=False)):
        return model_cfg
    with open_dict(model_cfg):
        model_cfg.z_dim = int(model_cfg.z_dim) + extra
        model_cfg._prefix_z_dim_expanded = True
    return model_cfg


def build_state_history_buffer(cfg: Any) -> StateHistoryBuffer:
    from omegaconf import OmegaConf

    hist = OmegaConf.select(cfg, "algorithm.state_history", default=None)
    enable = bool(OmegaConf.select(hist, "enable", default=False)) if hist else False
    steps = int(OmegaConf.select(hist, "steps", default=4) or 4) if hist else 4
    pad = str(OmegaConf.select(hist, "pad", default="zero") or "zero") if hist else "zero"
    proprio_dim = OmegaConf.select(cfg, "actor.model.proprio_dim", default=None)
    if proprio_dim is None:
        proprio_dim = OmegaConf.select(cfg, "rollout.model.proprio_dim", default=0)
    return StateHistoryBuffer(
        enable=enable,
        steps=steps,
        proprio_dim=int(proprio_dim or 0),
        pad=pad,
    )
