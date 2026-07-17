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

from dataclasses import dataclass, field, fields, replace
from typing import Any, Self, Sequence, TypeAlias

import torch


@dataclass(kw_only=True)
class TrajectoryData:
    """Metadata shared by every trajectory transition record."""

    global_step: int
    rank: int
    current_epoch: int
    current_step: int = 0
    stage_id: int = 0
    slot_ids: tuple[int, ...] | None = None

    @property
    def batch_size(self) -> int:
        """Infer the leading batch size from this record's payload fields."""
        if self.slot_ids is not None:
            return len(self.slot_ids)
        for item in fields(self):
            size = _batch_size_of(getattr(self, item.name))
            if size is not None:
                return size
        raise ValueError(f"Cannot infer batch size for {type(self).__name__}.")

    def select(
        self,
        indices: Sequence[int],
        *,
        slot_ids: Sequence[int] | None = None,
    ) -> Self:
        """Return a batch shard while preserving trajectory metadata."""
        indices = tuple(indices)
        selected_values = {
            item.name: _select_batch_value(getattr(self, item.name), indices)
            for item in fields(self)
        }
        if slot_ids is not None:
            if len(slot_ids) != len(indices):
                raise ValueError("slot_ids must have the same length as indices.")
            selected_values["slot_ids"] = tuple(slot_ids)
        return replace(self, **selected_values)


@dataclass(frozen=True)
class TrajectoryEnvelope:
    """Preserve an existing decoupled-mode routing envelope around one record."""

    record: TrajectoryData
    batch_index: str | None = None

    @classmethod
    def from_channel_item(cls, item: Any) -> "TrajectoryEnvelope":
        """Normalize normal and decoupled channel payloads."""
        if isinstance(item, TrajectoryData):
            return cls(record=item)
        if isinstance(item, dict) and set(item) == {"batch_index", "batch"}:
            if not isinstance(item["batch"], TrajectoryData):
                raise TypeError(
                    "Decoupled trajectory payload must contain TrajectoryData."
                )
            return cls(record=item["batch"], batch_index=item["batch_index"])
        raise TypeError(
            f"Unsupported trajectory channel payload: {type(item).__name__}."
        )

    def select(
        self, indices: Sequence[int], *, slot_ids: Sequence[int]
    ) -> "TrajectoryEnvelope":
        """Shard only the batched record, retaining its route identity."""
        return type(self)(
            record=self.record.select(indices, slot_ids=slot_ids),
            batch_index=self.batch_index,
        )

    def to_channel_item(self) -> TrajectoryData | dict[str, Any]:
        """Restore the channel payload shape expected by ``recv_from``."""
        if self.batch_index is None:
            return self.record
        return {"batch_index": self.batch_index, "batch": self.record}


@dataclass(kw_only=True)
class Observations(TrajectoryData):
    """Environment output for rollout/reward relay and trajectory storage."""

    obs: dict[str, torch.Tensor] = field(default_factory=dict)
    task_descriptions: list[str] | None = None
    next_obs: dict[str, torch.Tensor] = field(default_factory=dict)
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    final_obs: dict[str, Any] | None = None
    intervene_actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None
    reward_inputs: dict[str, Any] | None = None

    def to_rollout_input(self) -> dict[str, Any]:
        """Return the existing rollout worker input payload."""
        obs = self.obs
        if self.task_descriptions is not None:
            obs = {**obs, "task_descriptions": self.task_descriptions}
        return {
            "obs": obs,
            "final_obs": self.final_obs,
            "rlt_switch_flags": self.rlt_switch_flags,
            "intervene_flags": self.intervene_flags,
        }


@dataclass(kw_only=True)
class Actions(TrajectoryData):
    """Action and rollout statistics produced by the rollout worker."""

    actions: torch.Tensor | None = None
    prev_logprobs: torch.Tensor | None = None
    prev_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    bootstrap_values: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None

    @classmethod
    def from_rollout_result(
        cls,
        rollout_result: Any,
        *,
        num_action_chunks: int,
        **metadata: Any,
    ) -> "Actions":
        """Wrap a rollout result with chunked actions for trajectory storage."""
        actions = cls._canonicalize_action_chunks(
            rollout_result.actions, num_action_chunks
        )
        prev_logprobs = cls._canonicalize_action_chunks(
            rollout_result.prev_logprobs, num_action_chunks
        )
        versions = cls._canonicalize_action_chunks(
            rollout_result.versions, num_action_chunks
        )
        forward_inputs = dict(rollout_result.forward_inputs)
        for name in ("action", "model_action"):
            if name in forward_inputs:
                forward_inputs[name] = cls._canonicalize_action_chunks(
                    forward_inputs[name], num_action_chunks
                )
        return cls(
            actions=actions,
            prev_logprobs=prev_logprobs,
            prev_values=rollout_result.prev_values,
            versions=versions,
            forward_inputs=forward_inputs,
            bootstrap_values=rollout_result.bootstrap_values,
            intervene_flags=rollout_result.intervene_flags,
            **metadata,
        )

    @staticmethod
    def _canonicalize_action_chunks(
        actions: torch.Tensor | None, num_action_chunks: int
    ) -> torch.Tensor | None:
        """Normalize flattened and chunked action tensors to ``[B, C, A]``."""
        if actions is None:
            return None
        if actions.dim() == 2:
            if actions.shape[1] % num_action_chunks:
                raise ValueError(
                    "Flattened action width must be divisible by num_action_chunks."
                )
            return actions.reshape(actions.shape[0], num_action_chunks, -1)
        if actions.dim() == 3 and actions.shape[1] == num_action_chunks:
            return actions
        raise ValueError(
            "Trajectory actions must have shape [batch, num_action_chunks, action_dim]."
        )

    def to_rollout_result(self) -> Any:
        """Rebuild the existing live rollout payload for EnvWorker."""
        from rlinf.data.embodied_io_struct import RolloutResult

        return RolloutResult(
            actions=self.actions,
            prev_logprobs=self.prev_logprobs,
            prev_values=self.prev_values,
            versions=self.versions,
            forward_inputs=self.forward_inputs,
            bootstrap_values=self.bootstrap_values,
            intervene_flags=self.intervene_flags,
        )


@dataclass(kw_only=True)
class Rewards(TrajectoryData):
    """Reward result produced by the reward worker."""

    rewards: torch.Tensor | None = None
    reward_mode: str = "per_step"
    history_lengths: dict[str, list[int]] | None = None


@dataclass(kw_only=True)
class EnvBootstrap(TrajectoryData):
    """Final environment state needed to bootstrap a rollout segment."""

    observations: dict[str, torch.Tensor] = field(default_factory=dict)
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    final_obs: dict[str, Any] | None = None
    intervene_actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None

    def to_rollout_input(self) -> dict[str, Any]:
        """Return the final EnvWorker payload consumed by RolloutWorker."""
        return {
            "obs": self.observations,
            "final_obs": self.final_obs,
            "rlt_switch_flags": self.rlt_switch_flags,
            "intervene_flags": self.intervene_flags,
        }


@dataclass(kw_only=True)
class RolloutBootstrap(TrajectoryData):
    """Final value prediction needed to bootstrap a rollout segment."""

    prev_values: torch.Tensor | None = None
    bootstrap_values: torch.Tensor | None = None


TrajectoryRecord: TypeAlias = (
    Observations | Actions | Rewards | EnvBootstrap | RolloutBootstrap
)


def _batch_size_of(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim else None
    if isinstance(value, dict):
        for nested_value in value.values():
            size = _batch_size_of(nested_value)
            if size is not None:
                return size
    return None


def merge_trajectory_data(items: Sequence[TrajectoryData]) -> TrajectoryData:
    """Merge trajectory-worker shards back into one component-local batch."""
    if not items:
        raise ValueError("At least one trajectory data shard is required.")
    first = items[0]
    if any(type(item) is not type(first) for item in items[1:]):
        raise ValueError("Trajectory data shards must have the same datatype.")
    values = {}
    for item_field in fields(first):
        field_values = [getattr(item, item_field.name) for item in items]
        if item_field.name == "slot_ids":
            values[item_field.name] = tuple(
                slot_id
                for slot_ids in field_values
                if slot_ids is not None
                for slot_id in slot_ids
            )
        else:
            values[item_field.name] = _merge_batch_values(field_values)
    return replace(first, **values)


def merge_trajectory_envelopes(
    items: Sequence[TrajectoryEnvelope],
) -> TrajectoryEnvelope:
    """Merge relay shards while preserving their shared batch route identity."""
    if not items:
        raise ValueError("At least one trajectory envelope is required.")
    batch_index = items[0].batch_index
    if any(item.batch_index != batch_index for item in items[1:]):
        raise ValueError("Trajectory envelope shards have different batch indices.")
    return TrajectoryEnvelope(
        record=merge_trajectory_data([item.record for item in items]),
        batch_index=batch_index,
    )


def _select_batch_value(value: Any, indices: tuple[int, ...]) -> Any:
    if isinstance(value, torch.Tensor):
        return value[list(indices)].contiguous()
    if isinstance(value, dict):
        return {
            key: _select_batch_value(nested_value, indices)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [value[index] for index in indices]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    return value


def _merge_batch_values(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if any(value is None for value in values):
            raise ValueError("Tensor trajectory fields must be present in every shard.")
        return torch.cat(values, dim=0).contiguous()
    if isinstance(first, dict):
        keys = first.keys()
        if any(value is None or value.keys() != keys for value in values):
            raise ValueError("Trajectory dictionary shards must have matching keys.")
        return {
            key: _merge_batch_values([value[key] for value in values]) for key in keys
        }
    if isinstance(first, list):
        return [element for value in values for element in value]
    if isinstance(first, tuple):
        return tuple(element for value in values for element in value)
    if any(value != first for value in values[1:]):
        raise ValueError("Non-batched trajectory metadata differs across shards.")
    return first
