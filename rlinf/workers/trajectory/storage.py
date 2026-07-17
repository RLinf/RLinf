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

"""Per-trajectory-worker storage for assembling embodied trajectories."""

from dataclasses import dataclass, fields
from typing import Any, Iterable, Sequence

import torch

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id

from .data import (
    Actions,
    EnvBootstrap,
    Observations,
    Rewards,
    RolloutBootstrap,
)


@dataclass(frozen=True)
class TrajectoryStorageConfig:
    """Static shape information for one trajectory-worker shard."""

    num_envs: int
    rollout_epoch: int
    max_steps_per_rollout_epoch: int
    max_episode_length: int
    requires_values: bool
    action_chunk_shape: tuple[int, int] | None = None
    requires_external_rewards: bool = False
    env_reward_weight: float = 1.0
    reward_weight: float = 1.0
    gamma: float = 1.0
    auto_reset: bool = False
    bootstrap_type: str = "standard"


@dataclass(frozen=True)
class TrajectoryTensorSpec:
    """Canonical allocation policy for one trajectory tensor leaf."""

    length: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class TrajectoryStorageSchema:
    """Own tensor layouts and allocations for one trajectory storage shard.

    Known fields are registered from configuration.  Observation and optional
    model-input leaves are bound by their first complete typed record: all
    leaves in that record are validated before any allocation or write occurs.
    Subsequent records only use this schema; they never infer a new layout.
    """

    _ACTION_LAYOUT_PATHS = frozenset(
        {
            "actions",
            "prev_logprobs",
            "versions",
            "forward_inputs.action",
            "forward_inputs.model_action",
        }
    )

    def __init__(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._specs: dict[str, TrajectoryTensorSpec] = {}
        self._tensors: dict[str, torch.Tensor] = {}

    @property
    def specs(self) -> dict[str, TrajectoryTensorSpec]:
        """Return a shallow copy of the registered field specifications."""
        return dict(self._specs)

    def register_static(
        self, path: str, *, length: int, shape: Sequence[int], dtype: torch.dtype
    ) -> None:
        """Register a configuration-derived layout before any record arrives."""
        spec = TrajectoryTensorSpec(length, tuple(shape), dtype)
        existing = self._specs.get(path)
        if existing is not None and existing != spec:
            raise ValueError(
                f"Conflicting trajectory schema for {path!r}: {existing} vs {spec}."
            )
        self._specs[path] = spec

    def allocate(self, path: str) -> torch.Tensor:
        """Allocate the tensor owned by a registered field exactly once."""
        try:
            spec = self._specs[path]
        except KeyError as error:
            raise ValueError(f"Trajectory field {path!r} is not registered.") from error
        if path not in self._tensors:
            self._tensors[path] = torch.empty(
                (spec.length, self._num_envs, *spec.shape), dtype=spec.dtype
            )
        return self._tensors[path]

    def tensor(self, path: str) -> torch.Tensor | None:
        """Return an allocated field, if the record type actually uses it."""
        return self._tensors.get(path)

    def register_record(
        self, fields: Iterable[tuple[str, torch.Tensor | None, int]]
    ) -> None:
        """Atomically bind all tensor leaves of one typed record.

        The validation pass intentionally precedes allocation.  A malformed
        late observation therefore cannot leave half of its sibling leaves in
        the schema, which is what made the old first-write allocator fragile.
        """
        candidates: dict[str, TrajectoryTensorSpec] = {}
        for path, value, length in fields:
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Trajectory field {path!r} must be a tensor, got {type(value).__name__}."
                )
            if value.dim() == 0:
                raise ValueError(
                    f"Trajectory field {path!r} must include a batch axis."
                )
            spec = TrajectoryTensorSpec(length, tuple(value.shape[1:]), value.dtype)
            existing = self._specs.get(path)
            if existing is None:
                previous = candidates.get(path)
                if previous is not None and previous != spec:
                    raise ValueError(
                        f"Conflicting fields for {path!r} in one trajectory record."
                    )
                candidates[path] = spec
                continue
            self._validate_layout(path, value, existing, length)

        self._specs.update(candidates)
        for path in candidates:
            self.allocate(path)

    def write(
        self, path: str, value: torch.Tensor | None, index: int, slots: list[int]
    ) -> torch.Tensor | None:
        """Write a registered tensor after canonical shape/dtype conversion."""
        if value is None:
            return self.tensor(path)
        try:
            spec = self._specs[path]
        except KeyError as error:
            raise ValueError(
                f"Trajectory field {path!r} was written before schema registration."
            ) from error
        value = self._canonicalize(path, value, spec)
        if value.shape[0] != len(slots):
            raise ValueError(
                f"Trajectory field {path!r} has batch size {value.shape[0]}, "
                f"expected {len(slots)} for routed slots."
            )
        target = self.allocate(path)
        target[index, slots] = value
        return target

    def _validate_layout(
        self, path: str, value: torch.Tensor, spec: TrajectoryTensorSpec, length: int
    ) -> None:
        if length != spec.length:
            raise ValueError(
                f"Trajectory field {path!r} has time length {length}, expected {spec.length}."
            )
        if tuple(value.shape[1:]) != spec.shape:
            expected_numel = value.shape[0]
            for dimension in spec.shape:
                expected_numel *= dimension
            if path not in self._ACTION_LAYOUT_PATHS or value.numel() != expected_numel:
                raise ValueError(
                    f"Trajectory field {path!r} has shape {tuple(value.shape)}; "
                    f"expected [batch, {', '.join(map(str, spec.shape))}]."
                )
        if value.dtype != spec.dtype and not (
            value.is_floating_point()
            and torch.empty((), dtype=spec.dtype).is_floating_point()
        ):
            raise ValueError(
                f"Trajectory field {path!r} has dtype {value.dtype}, expected {spec.dtype}."
            )

    def _canonicalize(
        self, path: str, value: torch.Tensor, spec: TrajectoryTensorSpec
    ) -> torch.Tensor:
        value = value.detach().cpu().contiguous()
        if tuple(value.shape[1:]) != spec.shape:
            expected_numel = value.shape[0]
            for dimension in spec.shape:
                expected_numel *= dimension
            if path in self._ACTION_LAYOUT_PATHS and value.numel() == expected_numel:
                value = value.reshape(value.shape[0], *spec.shape)
            else:
                raise ValueError(
                    f"Trajectory field {path!r} has shape {tuple(value.shape)}; "
                    f"expected [batch, {', '.join(map(str, spec.shape))}]."
                )
        if value.dtype == spec.dtype:
            return value
        if (
            value.is_floating_point()
            and torch.empty((), dtype=spec.dtype).is_floating_point()
        ):
            return value.to(dtype=spec.dtype)
        raise ValueError(
            f"Trajectory field {path!r} has dtype {value.dtype}, expected {spec.dtype}."
        )


class TrajectoryStorage:
    """Assemble records for a fixed set of environment slots.

    ``Observations(current_step=t)`` contains the state at ``t`` and, when
    ``t > 0``, the reward produced by transition ``t - 1``.  ``EnvBootstrap``
    supplies the final state and the reward of the final transition.  This is
    the same time alignment used by :class:`EmbodiedRolloutResult`.
    """

    def __init__(
        self,
        num_envs: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        max_episode_length: int,
        requires_values: bool,
        action_chunk_shape: tuple[int, int] | None = None,
        requires_external_rewards: bool = False,
        env_reward_weight: float = 1.0,
        reward_weight: float = 1.0,
        gamma: float = 1.0,
        auto_reset: bool = False,
        bootstrap_type: str = "standard",
    ) -> None:
        if min(num_envs, rollout_epoch, max_steps_per_rollout_epoch) <= 0:
            raise ValueError("Trajectory storage dimensions must be positive.")
        self.num_envs = num_envs
        self.rollout_epoch = rollout_epoch
        self.max_steps_per_rollout_epoch = max_steps_per_rollout_epoch
        self.max_episode_length = max_episode_length
        self.requires_values = requires_values
        self.action_chunk_shape = action_chunk_shape
        self.requires_external_rewards = requires_external_rewards
        self.env_reward_weight = env_reward_weight
        self.reward_weight = reward_weight
        self.gamma = gamma
        self.auto_reset = auto_reset
        self.bootstrap_type = bootstrap_type
        self.transition_length = rollout_epoch * max_steps_per_rollout_epoch
        self.state_length = rollout_epoch * (max_steps_per_rollout_epoch + 1)
        self.schema = TrajectoryStorageSchema(num_envs)
        self._register_builtin_schema()

        self.actions: torch.Tensor | None = self.schema.tensor("actions")
        self.intervene_flags: torch.Tensor | None = self.schema.tensor(
            "intervene_flags"
        )
        self.rewards: torch.Tensor | None = None
        self._environment_rewards: torch.Tensor | None = None
        self._reward_model_rewards: torch.Tensor | None = None
        self.terminations: torch.Tensor | None = None
        self.truncations: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
        self.prev_logprobs: torch.Tensor | None = None
        self.prev_values: torch.Tensor | None = None
        self.versions: torch.Tensor | None = None
        self.forward_inputs: dict[str, Any] = {}
        self.curr_obs: dict[str, Any] = {}
        self.next_obs: dict[str, Any] = {}

        self._actions_ready = torch.zeros(
            self.transition_length, num_envs, dtype=torch.bool
        )
        self._rewards_ready = torch.zeros_like(self._actions_ready)
        self._environment_rewards_ready = torch.zeros_like(self._actions_ready)
        self._reward_model_rewards_ready = torch.zeros_like(self._actions_ready)
        self._values_ready = torch.zeros(self.state_length, num_envs, dtype=torch.bool)
        self._pending_interventions: dict[
            tuple[int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._terminal_bootstrap_applied = torch.zeros(
            rollout_epoch, num_envs, dtype=torch.bool
        )

    def _register_builtin_schema(self) -> None:
        """Register every configuration-defined trajectory field.

        Scalar rewards, termination flags, and values have fixed embodied-RL
        layouts.  Action-derived fields share the configured chunk layout.
        Optional fields are allocated only when their record type uses them,
        preserving ``None`` in exported trajectories without sacrificing a
        deterministic schema.
        """
        if self.action_chunk_shape is None:
            return
        chunk_shape = self.action_chunk_shape[:1]
        for path in ("rewards", "environment_rewards"):
            self.schema.register_static(
                path,
                length=self.transition_length,
                shape=chunk_shape,
                dtype=torch.float32,
            )
        # Image reward models score one observation per chunk.  The resulting
        # [batch, 1] value is deliberately broadcast over environment action
        # steps when it is combined with [batch, num_action_chunks] rewards.
        self.schema.register_static(
            "reward_model_rewards",
            length=self.transition_length,
            shape=(1,),
            dtype=torch.float32,
        )
        for path in ("dones", "terminations", "truncations"):
            self.schema.register_static(
                path,
                length=self.state_length,
                shape=chunk_shape,
                dtype=torch.bool,
            )
        if self.requires_values:
            self.schema.register_static(
                "prev_values", length=self.state_length, shape=(1,), dtype=torch.float32
            )
        for path in ("actions", "prev_logprobs", "versions"):
            self.schema.register_static(
                path,
                length=self.transition_length,
                shape=self.action_chunk_shape,
                dtype=torch.float32,
            )
        # ``action`` is the environment action and shares the configured
        # [num_action_chunks, action_dim] layout.  ``model_action`` is an
        # internal model representation (OpenPI uses [num_action_chunks, 320])
        # and is atomically bound from the first complete Actions record.
        self.schema.register_static(
            "forward_inputs.action",
            length=self.transition_length,
            shape=self.action_chunk_shape,
            dtype=torch.float32,
        )
        self.schema.register_static(
            "intervene_flags",
            length=self.transition_length,
            shape=self.action_chunk_shape,
            dtype=torch.bool,
        )
        self.actions = self.schema.allocate("actions")
        self.intervene_flags = self.schema.allocate("intervene_flags")

    @classmethod
    def from_config(cls, config: TrajectoryStorageConfig) -> "TrajectoryStorage":
        return cls(**config.__dict__)

    def write(self, item: object, slot_indices: Iterable[int]) -> None:
        """Write one typed record into this worker's local slot shard."""
        if isinstance(item, Observations):
            self.write_observations(item, slot_indices)
        elif isinstance(item, Actions):
            self.write_actions(item, slot_indices)
        elif isinstance(item, Rewards):
            self.write_rewards(item, slot_indices)
        elif isinstance(item, EnvBootstrap):
            self.write_env_bootstrap(item, slot_indices)
        elif isinstance(item, RolloutBootstrap):
            self.write_rollout_bootstrap(item, slot_indices)
        else:
            raise ValueError(f"Unsupported trajectory record: {type(item).__name__}.")

    def write_observations(
        self, item: Observations, slot_indices: Iterable[int]
    ) -> None:
        slots = self._slots(slot_indices)
        state_index = self._state_index(item.current_epoch, item.current_step)
        transition_index = self._transition_index(item.current_epoch, item.current_step)
        fields = [
            *self._dict_fields("curr_obs", item.obs, self.transition_length),
            *self._dict_fields("next_obs", item.next_obs, self.transition_length),
            ("dones", item.dones, self.state_length),
            ("terminations", item.terminations, self.state_length),
            ("truncations", item.truncations, self.state_length),
        ]
        if item.current_step > 0:
            fields.append(
                (
                    "environment_rewards"
                    if self.requires_external_rewards
                    else "rewards",
                    item.rewards,
                    self.transition_length,
                )
            )
        self._register_record(fields)
        self.curr_obs = self._write_dict(
            self.curr_obs,
            item.obs,
            "curr_obs",
            transition_index,
            slots,
        )
        self.next_obs = self._write_dict(
            self.next_obs,
            item.next_obs,
            "next_obs",
            transition_index,
            slots,
        )
        self._write_state_flags(item, state_index, slots)

        if item.current_step > 0:
            previous_transition = transition_index - 1
            self._write_environment_rewards_at(item.rewards, previous_transition, slots)

    def write_actions(self, item: Actions, slot_indices: Iterable[int]) -> None:
        if item.actions is None:
            raise ValueError("Actions record must include actions.")
        slots = self._slots(slot_indices)
        index = self._transition_index(item.current_epoch, item.current_step)
        self._register_record(
            [
                ("actions", item.actions, self.transition_length),
                ("prev_logprobs", item.prev_logprobs, self.transition_length),
                ("prev_values", item.prev_values, self.state_length),
                ("versions", item.versions, self.transition_length),
                *self._dict_fields(
                    "forward_inputs", item.forward_inputs, self.transition_length
                ),
            ]
        )
        self.actions = self._write_tensor("actions", item.actions, index, slots)
        if self.intervene_flags is None:
            self.schema.register_static(
                "intervene_flags",
                length=self.transition_length,
                shape=tuple(self.actions.shape[2:]),
                dtype=torch.bool,
            )
            self.intervene_flags = self.schema.allocate("intervene_flags")
            self.intervene_flags.zero_()
        self.prev_logprobs = self._write_tensor(
            "prev_logprobs", item.prev_logprobs, index, slots
        )
        self.prev_values = self._write_tensor(
            "prev_values",
            item.prev_values,
            self._state_index(item.current_epoch, item.current_step),
            slots,
        )
        if item.prev_values is not None:
            self._values_ready[
                self._state_index(item.current_epoch, item.current_step), slots
            ] = True
        self.versions = self._write_tensor("versions", item.versions, index, slots)
        self.forward_inputs = self._write_dict(
            self.forward_inputs,
            item.forward_inputs,
            "forward_inputs",
            index,
            slots,
        )
        self._actions_ready[index, slots] = True
        self._apply_pending_intervention(index, slots)

    def write_rewards(self, item: Rewards, slot_indices: Iterable[int]) -> None:
        if item.reward_mode != "per_step":
            raise ValueError(f"Unsupported reward mode: {item.reward_mode!r}.")
        slots = self._slots(slot_indices)
        # Reward workers consume the observation emitted after an environment
        # step.  Its ``current_step`` therefore names the following rollout
        # request, while the reward belongs to the preceding action chunk.
        if item.current_step <= 0:
            raise ValueError("Rewards records must follow an environment step.")
        index = self._transition_index(item.current_epoch, item.current_step - 1)
        self._register_record(
            [
                (
                    "reward_model_rewards"
                    if self.requires_external_rewards
                    else "rewards",
                    item.rewards,
                    self.transition_length,
                )
            ]
        )
        if not self.requires_external_rewards:
            self._write_rewards_at(item.rewards, index, slots)
            self._rewards_ready[index, slots] = True
            return
        self._reward_model_rewards = self._write_tensor(
            "reward_model_rewards",
            item.rewards,
            index,
            slots,
        )
        self._reward_model_rewards_ready[index, slots] = True
        self._refresh_external_rewards(index, slots)

    def write_env_bootstrap(
        self, item: EnvBootstrap, slot_indices: Iterable[int]
    ) -> None:
        slots = self._slots(slot_indices)
        state_index = self._bootstrap_state_index(item.current_epoch)
        transition_index = self._bootstrap_transition_index(item.current_epoch)
        self._register_record(
            [
                *self._dict_fields(
                    "next_obs", item.observations, self.transition_length
                ),
                (
                    "environment_rewards"
                    if self.requires_external_rewards
                    else "rewards",
                    item.rewards,
                    self.transition_length,
                ),
                ("dones", item.dones, self.state_length),
                ("terminations", item.terminations, self.state_length),
                ("truncations", item.truncations, self.state_length),
            ]
        )
        self.next_obs = self._write_dict(
            self.next_obs,
            item.observations,
            "next_obs",
            transition_index,
            slots,
        )
        self._write_environment_rewards_at(item.rewards, transition_index, slots)
        self._write_state_flags(item, state_index, slots)

    def write_rollout_bootstrap(
        self, item: RolloutBootstrap, slot_indices: Iterable[int]
    ) -> None:
        slots = self._slots(slot_indices)
        index = self._bootstrap_state_index(item.current_epoch)
        self._register_record([("prev_values", item.prev_values, self.state_length)])
        self.prev_values = self._write_tensor(
            "prev_values", item.prev_values, index, slots
        )
        self._values_ready[index, slots] = True

    def apply_intervention(
        self,
        current_epoch: int,
        current_step: int,
        slot_indices: Iterable[int],
        intervene_actions: torch.Tensor,
        intervene_flags: torch.Tensor,
    ) -> None:
        """Replace selected action chunks, including records arriving later."""
        slots = self._slots(slot_indices)
        index = self._transition_index(current_epoch, current_step)
        if self.actions is None or not self._actions_ready[index, slots].all():
            self._pending_interventions[(index, tuple(slots))] = (
                intervene_actions.detach().cpu().contiguous(),
                intervene_flags.detach().cpu().contiguous(),
            )
            return
        self._apply_intervention(index, slots, intervene_actions, intervene_flags)

    def apply_terminal_bootstrap_reward(
        self,
        current_epoch: int,
        gamma: float,
        auto_reset: bool,
        bootstrap_type: str,
    ) -> None:
        """Add terminal-state bootstrap values once for truncated trajectories."""
        if bootstrap_type != "standard" or not auto_reset:
            return
        if self.rewards is None or self.prev_values is None or self.truncations is None:
            return
        transition_index = self._bootstrap_transition_index(current_epoch)
        state_index = self._bootstrap_state_index(current_epoch)
        if not (
            self._rewards_ready[transition_index].all()
            and self._values_ready[state_index].all()
        ):
            return
        truncations = self.truncations[state_index].to(torch.bool)
        truncated_slots = truncations.reshape(self.num_envs, -1).any(dim=1)
        apply_mask = truncated_slots & ~self._terminal_bootstrap_applied[current_epoch]
        while apply_mask.dim() < self.prev_values[state_index].dim():
            apply_mask = apply_mask.unsqueeze(-1)
        self.rewards[transition_index] += (
            gamma
            * self.prev_values[state_index]
            * apply_mask.to(self.prev_values.dtype)
        )
        self._terminal_bootstrap_applied[current_epoch] |= truncated_slots

    def complete(self) -> bool:
        """Whether every local slot has a trainable rollout segment."""
        if not self._actions_ready.all() or not self._rewards_ready.all():
            return False
        return not self.requires_values or bool(self._values_ready.all())

    def to_trajectory(self, slot_indices: Iterable[int] | None = None) -> Trajectory:
        """Export the assembled trajectory in actor-consumable time layout."""
        if not self.complete():
            raise RuntimeError("Trajectory is not complete.")
        trajectory = Trajectory(max_episode_length=self.max_episode_length)
        for name in (
            "actions",
            "intervene_flags",
            "rewards",
            "terminations",
            "truncations",
            "dones",
            "prev_logprobs",
            "prev_values",
            "versions",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(trajectory, name, value.contiguous())
        trajectory.forward_inputs = self.forward_inputs
        trajectory.curr_obs = self.curr_obs
        trajectory.next_obs = self.next_obs
        trajectory.model_weights_id = get_model_weights_id(
            self.versions if self.versions is not None else torch.zeros(1)
        )
        if slot_indices is None:
            return trajectory
        return slice_trajectory(trajectory, slot_indices)

    def _write_rewards_at(
        self, value: torch.Tensor | None, index: int, slots: list[int]
    ) -> None:
        self.rewards = self._write_tensor("rewards", value, index, slots)

    def _write_environment_rewards_at(
        self, value: torch.Tensor | None, index: int, slots: list[int]
    ) -> None:
        if not self.requires_external_rewards:
            self._write_rewards_at(value, index, slots)
            self._rewards_ready[index, slots] = True
            return
        self._environment_rewards = self._write_tensor(
            "environment_rewards",
            value,
            index,
            slots,
        )
        self._environment_rewards_ready[index, slots] = True
        self._refresh_external_rewards(index, slots)

    def _refresh_external_rewards(self, index: int, slots: list[int]) -> None:
        if not (
            self._environment_rewards_ready[index, slots].all()
            and self._reward_model_rewards_ready[index, slots].all()
        ):
            return
        if self._reward_model_rewards is None:
            raise RuntimeError("Reward model readiness has no reward tensor.")
        model_rewards = self._reward_model_rewards[index, slots]
        if self._environment_rewards is None:
            environment_rewards = torch.zeros_like(model_rewards)
        else:
            environment_rewards = self._environment_rewards[index, slots]
        combined_rewards = (
            self.env_reward_weight * environment_rewards.to(model_rewards.dtype)
            + self.reward_weight * model_rewards
        )
        self._register_record([("rewards", combined_rewards, self.transition_length)])
        self._write_rewards_at(combined_rewards, index, slots)
        self._rewards_ready[index, slots] = True

    def _write_state_flags(
        self, item: Observations | EnvBootstrap, index: int, slots: list[int]
    ) -> None:
        self.dones = self._write_tensor("dones", item.dones, index, slots)
        self.terminations = self._write_tensor(
            "terminations", item.terminations, index, slots
        )
        self.truncations = self._write_tensor(
            "truncations", item.truncations, index, slots
        )

    def _apply_pending_intervention(self, index: int, slots: list[int]) -> None:
        pending = self._pending_interventions.pop((index, tuple(slots)), None)
        if pending is not None:
            self._apply_intervention(index, slots, *pending)

    def _apply_intervention(
        self,
        index: int,
        slots: list[int],
        intervene_actions: torch.Tensor,
        intervene_flags: torch.Tensor,
    ) -> None:
        assert self.actions is not None and self.intervene_flags is not None
        actions = intervene_actions.detach().cpu().contiguous()
        flags = intervene_flags.detach().cpu().to(torch.bool).contiguous()
        if flags.dim() == self.actions[index, slots].dim() - 1:
            flags = flags.unsqueeze(-1)
        expanded_flags = flags.expand_as(self.actions[index, slots])
        self.actions[index, slots] = torch.where(
            expanded_flags, actions, self.actions[index, slots]
        )
        self.intervene_flags[index, slots] = expanded_flags
        if "action" in self.forward_inputs:
            self.forward_inputs["action"][index, slots] = self.actions[index, slots]
        self.forward_inputs.pop("model_action", None)

    def _write_tensor(
        self,
        path: str,
        value: torch.Tensor | None,
        index: int,
        slots: list[int],
    ) -> torch.Tensor | None:
        return self.schema.write(path, value, index, slots)

    def _write_dict(
        self,
        target: dict[str, Any],
        value: dict[str, Any],
        root: str,
        index: int,
        slots: list[int],
    ) -> dict[str, Any]:
        for key, nested_value in value.items():
            # Text task descriptions are needed by the live rollout request,
            # but the actor trajectory path only supports batched tensors.
            # This mirrors EmbodiedRolloutResult.append_transitions().
            if key == "task_descriptions":
                continue
            path = f"{root}.{key}"
            if nested_value is None:
                continue
            if isinstance(nested_value, torch.Tensor):
                target[key] = self._write_tensor(path, nested_value, index, slots)
            elif isinstance(nested_value, dict):
                target[key] = self._write_dict(
                    target.get(key, {}), nested_value, path, index, slots
                )
            else:
                raise TypeError(
                    f"Unsupported trajectory dictionary value for {path!r}: "
                    f"{type(nested_value).__name__}."
                )
        return target

    def _register_record(
        self, fields: Iterable[tuple[str, torch.Tensor | None, int]]
    ) -> None:
        """Bind one typed record's complete tensor schema before writing it."""
        self.schema.register_record(fields)

    def _dict_fields(
        self, root: str, value: dict[str, Any], length: int
    ) -> list[tuple[str, torch.Tensor | None, int]]:
        fields: list[tuple[str, torch.Tensor | None, int]] = []
        for key, nested_value in value.items():
            if key == "task_descriptions" or nested_value is None:
                continue
            path = f"{root}.{key}"
            if isinstance(nested_value, torch.Tensor):
                fields.append((path, nested_value, length))
            elif isinstance(nested_value, dict):
                fields.extend(self._dict_fields(path, nested_value, length))
            else:
                raise TypeError(
                    f"Unsupported trajectory dictionary value for {path!r}: "
                    f"{type(nested_value).__name__}."
                )
        return fields

    def _slots(self, slot_indices: Iterable[int]) -> list[int]:
        slots = list(slot_indices)
        if not slots or min(slots) < 0 or max(slots) >= self.num_envs:
            raise ValueError("Slot indices are outside this trajectory-worker shard.")
        return slots

    def _transition_index(self, epoch: int, step: int) -> int:
        self._validate_epoch_step(epoch, step)
        return epoch * self.max_steps_per_rollout_epoch + step

    def _state_index(self, epoch: int, step: int) -> int:
        self._validate_epoch_step(epoch, step)
        return epoch * (self.max_steps_per_rollout_epoch + 1) + step

    def _bootstrap_transition_index(self, epoch: int) -> int:
        self._validate_epoch(epoch)
        return (epoch + 1) * self.max_steps_per_rollout_epoch - 1

    def _bootstrap_state_index(self, epoch: int) -> int:
        self._validate_epoch(epoch)
        return (epoch + 1) * (self.max_steps_per_rollout_epoch + 1) - 1

    def _validate_epoch_step(self, epoch: int, step: int) -> None:
        self._validate_epoch(epoch)
        if not 0 <= step < self.max_steps_per_rollout_epoch:
            raise ValueError(f"Step {step} is outside the rollout segment.")

    def _validate_epoch(self, epoch: int) -> None:
        if not 0 <= epoch < self.rollout_epoch:
            raise ValueError(f"Epoch {epoch} is outside the rollout segment.")


def slice_trajectory(trajectory: Trajectory, slot_indices: Iterable[int]) -> Trajectory:
    """Select a contiguous or discontiguous batch shard from a trajectory."""
    slots = list(slot_indices)
    values = {}
    for field in fields(trajectory):
        value = getattr(trajectory, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = value[:, slots].contiguous()
        elif isinstance(value, dict):
            values[field.name] = _slice_trajectory_dict(value, slots)
        else:
            values[field.name] = value
    return Trajectory(**values)


def merge_trajectories(trajectories: Iterable[Trajectory]) -> Trajectory:
    """Merge trajectory shards for one actor along their batch dimension."""
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("At least one trajectory shard is required.")
    values = {}
    for field in fields(Trajectory):
        field_values = [getattr(trajectory, field.name) for trajectory in trajectories]
        first = field_values[0]
        if isinstance(first, torch.Tensor):
            if any(value is None for value in field_values):
                raise ValueError(
                    f"Trajectory field {field.name!r} is missing in a shard."
                )
            values[field.name] = torch.cat(field_values, dim=1).contiguous()
        elif isinstance(first, dict):
            values[field.name] = _merge_trajectory_dicts(field_values)
        else:
            values[field.name] = first
    return Trajectory(**values)


def _slice_trajectory_dict(value: dict[str, Any], slots: list[int]) -> dict[str, Any]:
    return {
        key: (
            nested_value[:, slots].contiguous()
            if isinstance(nested_value, torch.Tensor)
            else _slice_trajectory_dict(nested_value, slots)
        )
        for key, nested_value in value.items()
    }


def _merge_trajectory_dicts(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = values[0].keys()
    if any(value.keys() != keys for value in values[1:]):
        raise ValueError("Trajectory dictionary shards have different keys.")
    merged = {}
    for key in keys:
        nested_values = [value[key] for value in values]
        if isinstance(nested_values[0], torch.Tensor):
            merged[key] = torch.cat(nested_values, dim=1).contiguous()
        else:
            merged[key] = _merge_trajectory_dicts(nested_values)
    return merged
