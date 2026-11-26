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

"""Prefix features emitted by a frozen model before it generates actions.

Any model that runs before action generation can expose these features by
implementing :class:`PrefixFeatureModel`. This contract is orthogonal to model
architecture: OpenPI, LingBot-VLA, and a pooled MLP all produce the same
``{z_rl, proprio, ref_chunk}`` observation, which a trainable off-policy policy
consumes instead of raw images.
"""

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import torch
from omegaconf import OmegaConf

PREFIX_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")

PrefixPoolMode = Literal["masked_mean", "mean", "last"]

PREFIX_POOL_MODES = ("masked_mean", "mean", "last", "rlt_token")


class PrefixObs(TypedDict):
    """Observation consumed by a trainable prefix policy.

    ``z_rl`` contains the frozen feature model output. ``proprio`` is the current
    robot state, and ``ref_chunk`` is the feature model's reference action chunk.
    """

    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor


@runtime_checkable
class PrefixFeatureModel(Protocol):
    """Frozen model that extracts prefix features and reference actions."""

    @property
    def z_dim(self) -> int: ...

    def extract_prefix_obs(self, env_obs: dict[str, Any]) -> PrefixObs:
        """Extract the compact observation consumed by the trainable policy."""
        ...


def extract_prefix_obs(
    feature_model: PrefixFeatureModel,
    env_obs: dict[str, Any],
) -> PrefixObs:
    """Extract a prefix observation through the canonical feature-model API."""
    extractor = getattr(feature_model, "extract_prefix_obs", None)
    if not callable(extractor):
        raise TypeError(
            f"{type(feature_model).__name__} must implement extract_prefix_obs()."
        )
    prefix_obs = extractor(env_obs)
    z_width = prefix_obs["z_rl"].shape[-1]
    if z_width != feature_model.z_dim:
        raise ValueError(
            f"{type(feature_model).__name__}.extract_prefix_obs() returned z_rl "
            f"with last dimension {z_width}, expected z_dim={feature_model.z_dim}."
        )
    return prefix_obs


def check_prefix_feature_dim(feature_model: PrefixFeatureModel, z_dim: int) -> None:
    """Reject a feature model whose ``z_rl`` width differs from the policy input."""
    if int(feature_model.z_dim) != int(z_dim):
        raise ValueError(
            "rollout.prefix_feature_model produces z_rl of width "
            f"{feature_model.z_dim}, but actor.model.z_dim is {z_dim}."
        )


def pool_prefix(
    hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    mode: PrefixPoolMode = "masked_mean",
) -> torch.Tensor:
    """Pool prefix hidden states ``[B, T, D]`` into features ``[B, D]``."""
    if hidden.ndim != 3:
        raise ValueError(
            f"pool_prefix expects hidden of shape [B, T, D], got {tuple(hidden.shape)}."
        )
    hidden = hidden.to(dtype=torch.float32)
    batch_size, seq_len, _ = hidden.shape
    if mode == "mean":
        return hidden.mean(dim=1)

    if mask is not None:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"pool_prefix expects mask [B={batch_size}, T={seq_len}] or "
                f"[B, T, 1], got {tuple(mask.shape)}."
            )

    if mode == "last":
        if mask is None:
            return hidden[:, -1]
        keep = mask.to(device=hidden.device, dtype=torch.bool)
        positions = torch.arange(seq_len, device=hidden.device)
        last_index = positions.expand_as(keep).masked_fill(~keep, 0).amax(dim=1)
        batch_index = torch.arange(batch_size, device=hidden.device)
        return hidden[batch_index, last_index]

    if mode != "masked_mean":
        raise ValueError(
            f"pool_prefix mode must be 'masked_mean', 'mean', or 'last', got {mode!r}."
        )

    if mask is None:
        return hidden.mean(dim=1)

    keep = mask.to(device=hidden.device, dtype=hidden.dtype)
    if keep.ndim == 2:
        keep = keep[..., None]
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (hidden * keep).sum(dim=1) / denom


def resolve_prefix_pool(
    *,
    use_rlt: bool = False,
    prefix_pool: str | None = None,
) -> str:
    """Return ``prefix.pool``, defaulting to RLT tokens when the module exists."""
    if prefix_pool:
        pool = str(prefix_pool)
        if pool not in PREFIX_POOL_MODES:
            raise ValueError(
                f"prefix.pool must be one of {PREFIX_POOL_MODES}, got {pool!r}."
            )
        return pool
    return "rlt_token" if use_rlt else "masked_mean"


def get_prefix_feature_dim(model_cfg: Any) -> int:
    """Return the ``z_rl`` width expected by the trainable policy."""
    prefix_dim = int(OmegaConf.select(model_cfg, "z_dim", default=0) or 0)
    if prefix_dim <= 0:
        raise ValueError("prefix_policy requires actor.model.z_dim > 0.")
    return prefix_dim
