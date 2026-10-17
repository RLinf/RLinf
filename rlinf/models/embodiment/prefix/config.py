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

from dataclasses import dataclass
from typing import Any, Literal

from omegaconf import OmegaConf

from rlinf.models.embodiment.prefix.history import StateHistoryBuffer
from rlinf.models.embodiment.prefix.pool import PREFIX_POOL_MODES

PREFIX_OFF_POLICY_LOSS_TYPE = "prefix_off_policy"


@dataclass(frozen=True)
class StateHistoryConfig:
    """Static state-history shape shared by rollout and policy construction."""

    enabled: bool
    steps: int
    proprio_dim: int
    pad: Literal["zero", "repeat"]

    @property
    def extra_dim(self) -> int:
        """Return the feature width contributed by state history."""
        return self.steps * self.proprio_dim if self.enabled else 0


def is_prefix_off_policy_loss(loss_type: Any) -> bool:
    """Return whether a config selects the prefix off-policy runtime."""
    return str(loss_type) == PREFIX_OFF_POLICY_LOSS_TYPE


def resolve_prefix_pool(
    *,
    use_rlt: bool = False,
    prefix_pool: str | None = None,
    stage2_z_source: str | None = None,
    rlt_use_mask: bool = False,
) -> str:
    """Resolve the pooling operation used to construct frozen prefix features."""
    if prefix_pool:
        pool = str(prefix_pool)
        if pool not in PREFIX_POOL_MODES:
            raise ValueError(
                f"prefix.pool must be one of {PREFIX_POOL_MODES}, got {pool!r}."
            )
        return pool
    if stage2_z_source == "vlm_prefix":
        return "masked_mean" if rlt_use_mask else "mean"
    if use_rlt:
        return "rlt_token"
    return "masked_mean"


def get_prefix_feature_model_config(cfg: Any) -> Any | None:
    """Return the canonical frozen feature-model config, if configured."""
    return OmegaConf.select(cfg, "rollout.prefix_feature_model", default=None)


def get_state_history_config(model_cfg: Any) -> StateHistoryConfig:
    """Parse state-history settings without mutating the Hydra config."""
    history_cfg = OmegaConf.select(model_cfg, "state_history", default=None)
    if history_cfg is None:
        enabled = False
        steps = 4
        pad = "zero"
    else:
        enabled = bool(OmegaConf.select(history_cfg, "enable", default=False))
        steps = int(OmegaConf.select(history_cfg, "steps", default=4) or 4)
        pad = str(OmegaConf.select(history_cfg, "pad", default="zero") or "zero")
    proprio_dim = int(OmegaConf.select(model_cfg, "proprio_dim", default=0) or 0)
    if steps < 1:
        raise ValueError(f"state_history.steps must be >= 1, got {steps}.")
    if pad not in {"zero", "repeat"}:
        raise ValueError(f"state_history.pad must be 'zero' or 'repeat', got {pad!r}.")
    if enabled and proprio_dim <= 0:
        raise ValueError(
            "state_history.enable=true requires actor.model.proprio_dim > 0."
        )
    return StateHistoryConfig(
        enabled=enabled,
        steps=steps,
        proprio_dim=proprio_dim,
        pad=pad,
    )


def get_prefix_feature_dim(model_cfg: Any) -> int:
    """Return the runtime ``z_rl`` width expected by the trainable policy."""
    prefix_dim = int(OmegaConf.select(model_cfg, "z_dim", default=0) or 0)
    if prefix_dim <= 0:
        raise ValueError("prefix_policy requires actor.model.z_dim > 0.")
    return prefix_dim + get_state_history_config(model_cfg).extra_dim


def build_state_history_buffer(model_cfg: Any) -> StateHistoryBuffer:
    """Build rollout history from the same model config that owns its shape."""
    history = get_state_history_config(model_cfg)
    return StateHistoryBuffer(
        enable=history.enabled,
        steps=history.steps,
        proprio_dim=history.proprio_dim,
        pad=history.pad,
    )
