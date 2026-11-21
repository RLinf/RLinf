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

"""Synchronous PEAR objective: REINFORCE, one sequence per episode, DAPO."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from rlinf.algorithms.registry import register_policy_loss
from rlinf.utils.utils import masked_mean


def last_valid_step_indices(
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(last_index, has_valid)`` of shape ``[batch]`` for ``[time, batch, ...]``."""
    valid = loss_mask.to(dtype=torch.bool)
    if valid.ndim > 2:
        valid = valid.flatten(start_dim=2).any(dim=-1)
    if valid.ndim != 2:
        raise ValueError(
            f"loss_mask must be at least [time, batch], got {tuple(loss_mask.shape)}"
        )
    time = torch.arange(valid.shape[0], device=valid.device).unsqueeze(1)
    last = torch.where(valid, time, torch.full_like(time, -1)).amax(dim=0)
    has_valid = last >= 0
    return last.clamp(min=0), has_valid


def _gather_time_step(
    value: Any,
    indices: torch.Tensor,
    time_steps: int,
    *,
    key_name: str = "",
) -> Any:
    """Gather per-env step ``indices[b]`` along dim 0 of a ``[T, B, ...]`` tree."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: _gather_time_step(item, indices, time_steps, key_name=key)
            for key, item in value.items()
        }
    if not torch.is_tensor(value):
        return value
    if value.ndim < 2 or value.shape[1] != indices.shape[0]:
        return value
    idx = indices.to(device=value.device, dtype=torch.long)
    batch_arange = torch.arange(idx.shape[0], device=value.device)
    if value.shape[0] == time_steps:
        if key_name == "rewards":
            return value.sum(dim=0, keepdim=True)
        return value[idx, batch_arange].unsqueeze(0)
    if value.shape[0] == time_steps + 1:
        # Preserve [T+1, B, ...] layout as [2, B, ...]: pre-episode slot 0 and
        # post-episode slot at last_step + 1 so dones/values consumers still work.
        first = torch.zeros_like(value[:1])
        last = value[idx + 1, batch_arange].unsqueeze(0)
        return torch.cat([first, last], dim=0)
    return value


def compact_episode_sequence_batch(
    rollout_batch: dict[str, Any],
) -> dict[str, Any]:
    """Compress ``[T, B, ...]`` to ``[1, B, ...]`` keeping each env's final valid step.

    With ``episode_sequence=True``, the terminating step's ``forward_inputs``
    already contains the full multi-turn episode sequence. Without this
    compaction, ``run_training`` flattens all ``T * B`` steps and executes ``T``
    optimizer steps per RL iteration while running full forward/backward passes
    on ``T - 1`` zero-masked intermediate steps.
    """
    loss_mask = rollout_batch.get("loss_mask")
    if loss_mask is None or not torch.is_tensor(loss_mask) or loss_mask.ndim < 2:
        return rollout_batch
    time_steps = int(loss_mask.shape[0])
    if time_steps <= 1:
        return rollout_batch
    indices, has_valid = last_valid_step_indices(loss_mask)
    compacted: dict[str, Any] = {}
    for key, value in rollout_batch.items():
        compacted[key] = _gather_time_step(value, indices, time_steps, key_name=key)
    if "loss_mask" in compacted and torch.is_tensor(compacted["loss_mask"]):
        mask = compacted["loss_mask"]
        valid_env = has_valid.to(device=mask.device, dtype=mask.dtype).view(
            1, -1, *([1] * (mask.ndim - 2))
        )
        compacted["loss_mask"] = mask * valid_env
    if "loss_mask_sum" in compacted and torch.is_tensor(compacted.get("loss_mask")):
        refreshed = compacted["loss_mask"]
        if refreshed.ndim == 3:
            total = refreshed.to(dtype=torch.float32).sum(dim=(0, 2), keepdim=True)
        else:
            total = refreshed.to(dtype=torch.float32).sum(dim=0, keepdim=True)
        compacted["loss_mask_sum"] = total.expand_as(refreshed)
    return compacted


def count_valid_sequence_tokens(
    rollout_batch: dict[str, Any],
) -> float:
    """Count total valid response tokens across a flattened ``[N, ...]`` batch."""
    forward_inputs = rollout_batch.get("forward_inputs")
    if not isinstance(forward_inputs, dict):
        return 0.0
    response_mask = forward_inputs.get("response_mask")
    if not torch.is_tensor(response_mask):
        return 0.0
    token_mask = response_mask.to(dtype=torch.bool)
    loss_mask = rollout_batch.get("loss_mask")
    if torch.is_tensor(loss_mask):
        env_mask = loss_mask.to(device=token_mask.device, dtype=torch.bool)
        while env_mask.ndim > 1 and env_mask.shape[-1] == 1:
            env_mask = env_mask.squeeze(-1)
        while env_mask.ndim < token_mask.ndim:
            env_mask = env_mask.unsqueeze(-1)
        token_mask = token_mask & env_mask
    return float(token_mask.sum().item())


def is_trajectory_stale(
    trajectory: Any,
    current_version: int,
    max_off_policy_threshold: int | None,
) -> bool:
    """Return whether a trajectory exceeds PEAR's off-policy staleness threshold.

    Matches PEAR's ``max_off_policy_threshold`` (default 5 => staleness <= 4):
    a group whose oldest policy version satisfies
    ``current_version - version >= max_off_policy_threshold`` is dropped.
    """
    if max_off_policy_threshold is None or max_off_policy_threshold <= 0:
        return False
    versions = getattr(trajectory, "versions", None)
    if torch.is_tensor(versions) and versions.numel() > 0:
        oldest = int(versions.min().item())
        return (int(current_version) - oldest) >= int(max_off_policy_threshold)
    weight_id = getattr(trajectory, "model_weights_id", None)
    if weight_id is not None:
        try:
            return (int(current_version) - int(weight_id)) >= int(
                max_off_policy_threshold
            )
        except (TypeError, ValueError):
            return False
    return False


def keep_last_valid_step(loss_mask: torch.Tensor) -> torch.Tensor:
    """Keep the last valid step of each environment.

    ``loss_mask`` is ``[time, batch, ...]``. The last valid step is the
    terminating action: its forward already contains every earlier assistant
    span, so earlier steps must not enter the token mean a second time.
    """
    valid = loss_mask.to(dtype=torch.bool)
    if valid.ndim > 2:
        valid = valid.flatten(start_dim=2).any(dim=-1)
    if valid.ndim != 2:
        raise ValueError(
            f"loss_mask must be at least [time, batch], got {tuple(loss_mask.shape)}"
        )
    time = torch.arange(valid.shape[0], device=valid.device).unsqueeze(1)
    last = torch.where(valid, time, torch.full_like(time, -1)).amax(dim=0)
    keep = (time == last.unsqueeze(0)) & valid
    while keep.ndim < loss_mask.ndim:
        keep = keep.unsqueeze(-1)
    return loss_mask * keep.to(dtype=loss_mask.dtype)


def dapo_group_mask(
    rewards: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask groups whose episode returns have no variance.

    ``rewards`` has shape ``[time, batch, chunks]``. The episode return is the
    sum over time and chunks. Groups of identical returns are the ones DAPO
    drops and replaces.
    """
    if rewards.ndim != 3:
        raise ValueError(
            f"rewards must be [time, batch, chunks], got {tuple(rewards.shape)}"
        )
    batch = rewards.shape[1]
    if group_size <= 0 or batch % group_size != 0:
        raise ValueError(f"batch {batch} is not divisible by group_size {group_size}")
    scores = rewards.to(dtype=torch.float32).sum(dim=(0, 2))
    grouped = scores.view(-1, group_size)
    variance = grouped.std(dim=-1, unbiased=False)
    keep_group = variance > 0
    keep = keep_group.repeat_interleave(group_size)
    return keep, {
        "dapo/kept_groups": float(keep_group.sum().item()),
        "dapo/dropped_groups": float((~keep_group).sum().item()),
    }


def variable_group_index(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Env indices of groups whose returns are neither all equal nor empty."""
    keep, _ = dapo_group_mask(rewards, group_size)
    return keep.nonzero(as_tuple=False).flatten().to(dtype=torch.long)


def narrow_batch_dim(value, index: torch.Tensor):
    """Index a ``[time, batch, ...]`` tree on the batch axis. The index is copied."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim < 2:
            return value
        return value.index_select(1, index.to(device=value.device))
    if isinstance(value, dict):
        return {key: narrow_batch_dim(item, index) for key, item in value.items()}
    raise TypeError(f"cannot narrow {type(value)}")


def _pad_cat_batch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate on the batch axis, right-padding every later axis with zeros."""
    if left.ndim != right.ndim or left.shape[0] != right.shape[0]:
        raise ValueError(
            f"cannot concat batch tensors {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.ndim < 2:
        return left
    width = [max(left.shape[dim], right.shape[dim]) for dim in range(2, left.ndim)]
    padded = []
    for tensor in (left, right):
        pad: list[int] = []
        for dim in range(left.ndim - 1, 1, -1):
            pad.extend([0, width[dim - 2] - tensor.shape[dim]])
        padded.append(functional.pad(tensor, pad) if any(pad) else tensor)
    return torch.cat(padded, dim=1)


def concat_batch_dim(left, right):
    """Append ``right`` to ``left`` on the batch axis. ``None`` is the empty buffer."""
    if left is None:
        return right
    if right is None:
        return left
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise TypeError(f"cannot concat {type(left)} with {type(right)}")
        return _pad_cat_batch(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise KeyError(f"batch keys {sorted(left)} != {sorted(right)}")
        return {key: concat_batch_dim(left[key], right[key]) for key in left}
    raise TypeError(f"cannot concat {type(left)} with {type(right)}")


@dataclass(frozen=True)
class DapoPending:
    """One actor's signal that the shared pool is still short of a training batch."""

    kept_groups: int
    target_groups: int


@dataclass(frozen=True)
class TrajectoryDrain:
    """Tells every actor that this generation has no more groups to deliver."""


def _trajectory_fields(trajectory):
    return [
        name
        for name in trajectory.__dataclass_fields__
        if name not in {"max_episode_length", "model_weights_id"}
    ]


def narrow_trajectory(trajectory, index: torch.Tensor):
    """Copy one trajectory with its batch axis reduced to ``index``."""
    from rlinf.data.schema.embodied_types import Trajectory

    narrowed = Trajectory(
        max_episode_length=trajectory.max_episode_length,
        model_weights_id=trajectory.model_weights_id,
    )
    for name in _trajectory_fields(trajectory):
        value = getattr(trajectory, name)
        if value is None or value == {} or value == []:
            continue
        setattr(narrowed, name, narrow_batch_dim(value, index))
    return narrowed


def concat_trajectories(left, right):
    """Pad and append two trajectories on the batch axis."""
    if left is None:
        return right
    if right is None:
        return left
    from rlinf.data.schema.embodied_types import Trajectory

    merged = Trajectory(
        max_episode_length=max(left.max_episode_length, right.max_episode_length),
        model_weights_id=right.model_weights_id or left.model_weights_id,
    )
    for name in _trajectory_fields(left):
        setattr(
            merged,
            name,
            concat_batch_dim(getattr(left, name), getattr(right, name)),
        )
    return merged


def absorb_variable_groups(
    buffer,
    incoming: Sequence,
    *,
    group_size: int,
    target_groups: int,
    shard_count: int,
):
    """Add every variable group to the shared pool, then deal a full batch.

    Constant groups are dropped. Until the pool holds ``target_groups``, every
    shard is a :class:`DapoPending` carrying the same global count. A full pool
    deals the oldest groups into ``shard_count`` equal trajectories and keeps
    the surplus for the next update.
    """
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if target_groups <= 0 or (target_groups * group_size) % shard_count != 0:
        raise ValueError(
            "target_groups * group_size must be a positive multiple of "
            f"shard_count, got {target_groups}, {group_size}, {shard_count}"
        )
    pool = buffer
    for trajectory in incoming:
        if trajectory is None:
            continue
        index = variable_group_index(trajectory.rewards, group_size)
        if index.numel() == 0:
            continue
        pool = concat_trajectories(pool, narrow_trajectory(trajectory, index))
    have = 0 if pool is None else pool._batch_size() // group_size
    if have < target_groups:
        pending = DapoPending(kept_groups=have, target_groups=target_groups)
        return pool, [pending] * shard_count
    keep_count = target_groups * group_size
    train = narrow_trajectory(pool, torch.arange(keep_count))
    surplus = None
    if pool._batch_size() > keep_count:
        surplus = narrow_trajectory(pool, torch.arange(keep_count, pool._batch_size()))
    return surplus, train.split(shard_count)


class DapoGroupAccumulator:
    """Keep variable-reward groups until a target count is filled.

    This is the replacement half of DAPO: zero-variance groups are discarded
    and the caller samples another batch.
    """

    def __init__(self, target_groups: int, group_size: int):
        if target_groups <= 0 or group_size <= 0:
            raise ValueError("target_groups and group_size must be positive")
        self.target_groups = target_groups
        self.group_size = group_size
        self.kept: list[torch.Tensor] = []

    @property
    def ready(self) -> bool:
        return len(self.kept) >= self.target_groups

    def add_batch(self, episode_returns: torch.Tensor) -> bool:
        """Add a ``[num_groups, group_size]`` return tensor. Return whether full."""
        if episode_returns.ndim != 2 or episode_returns.shape[1] != self.group_size:
            raise ValueError(
                "episode_returns must be [num_groups, group_size], got "
                f"{tuple(episode_returns.shape)}"
            )
        variance = episode_returns.to(dtype=torch.float32).std(dim=-1, unbiased=False)
        for index in range(episode_returns.shape[0]):
            if float(variance[index]) <= 0:
                continue
            self.kept.append(episode_returns[index].detach().clone())
            if self.ready:
                break
        return self.ready

    def stacked(self) -> torch.Tensor:
        """Return the first ``target_groups`` kept rows."""
        rows = self.kept[: self.target_groups]
        if len(rows) != self.target_groups:
            raise RuntimeError(
                f"need {self.target_groups} variable groups, have {len(rows)}"
            )
        return torch.stack(rows, dim=0)


def fill_dapo_groups(
    target_groups: int,
    group_size: int,
    sample_batch: Callable[[], torch.Tensor],
    max_batches: int,
) -> torch.Tensor:
    """Sample until ``target_groups`` variable groups exist, or raise."""
    accumulator = DapoGroupAccumulator(target_groups, group_size)
    for _ in range(max_batches):
        if accumulator.add_batch(sample_batch()):
            return accumulator.stacked()
    raise RuntimeError(
        f"DAPO collected {len(accumulator.kept)} variable groups in "
        f"{max_batches} batches, target is {target_groups}"
    )


def pear_reinforce_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    loss_agg_func: Callable[..., torch.Tensor] | None = None,
    sc_correction: torch.Tensor | None = None,
    sc_sampler_head_mass: torch.Tensor | None = None,
    sc_train_head_mass: torch.Tensor | None = None,
    global_valid_tokens: float | torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Token-mean REINFORCE with optional PEAR Score Centering.

    Matches PEAR's bypass-mode REINFORCE loss (verl patch 0011):
    ``pg_losses = -advantages * logprobs + advantages * sc_correction``.
    No PPO clip, KL term, or entropy bonus.

    When ``global_valid_tokens`` is provided (global token count across all
    micro-batches and DP ranks), the loss is normalized by that denominator so
    gradient accumulation yields an exact batch-wide token-mean rather than a
    sequence-mean of per-microbatch token-means.
    """
    del kwargs
    if loss_agg_func is None:
        loss_agg_func = masked_mean
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs, dtype=torch.bool)
    mask_bool = loss_mask.to(device=logprobs.device, dtype=torch.bool)
    token_loss = -advantages * logprobs
    if sc_correction is not None:
        token_loss = token_loss + advantages * sc_correction
    if global_valid_tokens is not None:
        denom = (
            global_valid_tokens.to(device=token_loss.device, dtype=token_loss.dtype)
            if torch.is_tensor(global_valid_tokens)
            else token_loss.new_tensor(float(global_valid_tokens))
        ).clamp_min(1.0)
        policy_loss = (token_loss * mask_bool.to(dtype=token_loss.dtype)).sum() / denom
    else:
        policy_loss = loss_agg_func(token_loss, mask_bool, None)
    metrics = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/clip_fraction": torch.zeros((), device=policy_loss.device),
        "actor/ratio": torch.ones((), device=policy_loss.device),
    }
    if sc_correction is not None:
        metrics["actor/sc_correction"] = masked_mean(sc_correction.detach(), mask_bool)
    if sc_sampler_head_mass is not None:
        metrics["actor/sc_sampler_head_mass"] = masked_mean(
            sc_sampler_head_mass.detach(), mask_bool
        )
    if sc_train_head_mass is not None:
        metrics["actor/sc_train_head_mass"] = masked_mean(
            sc_train_head_mass.detach(), mask_bool
        )
    return policy_loss, metrics


@register_policy_loss("pear_reinforce")
def compute_pear_reinforce_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """Registered synchronous PEAR actor loss."""
    return pear_reinforce_loss(**kwargs)


def episode_returns(
    rewards: Sequence[Sequence[float]], group_size: int
) -> torch.Tensor:
    """Sum each trajectory and reshape to ``[num_groups, group_size]``."""
    flat = torch.tensor([sum(traj) for traj in rewards], dtype=torch.float32)
    if flat.numel() % group_size != 0:
        raise ValueError(
            f"{flat.numel()} returns are not divisible by group_size {group_size}"
        )
    return flat.view(-1, group_size)
