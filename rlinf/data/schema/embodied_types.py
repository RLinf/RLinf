# Copyright 2025 The RLinf Authors.
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

"""Transport and training data types for embodied rollouts.

Shape notation used by field comments:

* ``B`` is the routed environment batch. Before decoupled routing it is
  typically ``env.train.total_num_envs / env_world_size /
  rollout.pipeline_stage_num``; merged requests may have a larger ``B``.
* ``C`` is ``actor.model.num_action_chunks`` (``rollout.model`` in eval-only
  runs), ``A`` is ``actor.model.action_dim``, and ``D = C * A``.
* ``E`` is ``env.train.rollout_epoch``. Each epoch has
  ``env.train.max_steps_per_rollout_epoch / C`` chunks, so a full trajectory
  normally has ``T = E * max_steps_per_rollout_epoch / C`` chunks.

Observation shapes remain environment/model specific. Image and state tensors
always use ``B`` as their leading batch dimension during transport.
"""

import copy
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np
import torch

from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict,
    stack_list_of_dict_tensor,
)

# Source identity carried through the trajectory pipeline.


@dataclass(frozen=True)
class TrajectoryKey:
    """Identity of one action chunk produced by a logical environment source."""

    # Runner training step that requested this rollout.
    step_id: int
    # Rollout epoch in [0, env.train.rollout_epoch).
    epoch_id: int
    # Global rank of the environment worker that owns the source batch.
    env_rank: int
    # Pipeline stage in [0, rollout.pipeline_stage_num).
    stage_id: int
    # Action-chunk index in [0, max_steps_per_rollout_epoch / C).
    chunk_id: int


@dataclass(frozen=True)
class TrajectorySource:
    """Trajectory key and batch size carried by one routed source shard."""

    # Logical chunk identity shared by its policy and environment parts.
    key: TrajectoryKey
    # Number of environments in this shard; contributes to routed batch B.
    size: int
    # Leading-batch offset within the logical source before channel splitting.
    offset: int = 0

    @classmethod
    def split(
        cls,
        sources: list["TrajectorySource"],
        split_sizes: list[int],
    ) -> list[list["TrajectorySource"]]:
        """Split routed source metadata along the batch dimension."""
        if not sources:
            return [[] for _ in split_sizes]
        if sum(source.size for source in sources) != sum(split_sizes):
            raise ValueError("Source sizes do not match the requested batch splits.")

        shards = [[] for _ in split_sizes]
        source_index = 0
        source_offset = 0
        for shard, split_size in zip(shards, split_sizes):
            remaining = split_size
            while remaining:
                source = sources[source_index]
                take_size = min(remaining, source.size - source_offset)
                shard.append(
                    cls(
                        key=source.key,
                        size=take_size,
                        offset=source.offset + source_offset,
                    )
                )
                source_offset += take_size
                remaining -= take_size
                if source_offset == source.size:
                    source_index += 1
                    source_offset = 0
        return shards

    @classmethod
    def merge(
        cls,
        source_groups: list[list["TrajectorySource"]],
    ) -> list["TrajectorySource"]:
        """Merge adjacent shards while preserving logical source boundaries."""
        sources: list[TrajectorySource] = []
        for source in (item for group in source_groups for item in group):
            if (
                sources
                and sources[-1].key == source.key
                and sources[-1].offset + sources[-1].size == source.offset
            ):
                sources[-1] = cls(
                    key=source.key,
                    size=sources[-1].size + source.size,
                    offset=sources[-1].offset,
                )
            else:
                sources.append(source)
        return sources


# Collector inputs and their environment- or policy-owned payloads.


@dataclass(kw_only=True)
class EnvTransition:
    """Environment-owned outcome of one executed action chunk."""

    # Environment rewards, float [B, C]; absent on reset/bootstrap outputs.
    rewards: torch.Tensor | None = None
    # Combined terminal mask, bool [B, C].
    dones: torch.Tensor | None = None
    # Natural termination mask, bool [B, C].
    terminations: torch.Tensor | None = None
    # Time-limit/external truncation mask, bool [B, C].
    truncations: torch.Tensor | None = None
    # Expert actions for intervened slots, float [B, C, A] or [B, D].
    intervene_actions: torch.Tensor | None = None
    # Environment-side intervention mask, bool [B, C].
    intervene_flags: torch.Tensor | None = None
    # RLT route chosen per action, usually bool/int [B, C].
    rlt_switch_flags: torch.Tensor | None = None
    # External reward-model scores aligned with ``rewards``, float [B, C].
    reward_model_output: torch.Tensor | None = None
    # History lengths used for delayed reward assignment; one int per env [B].
    reward_assign_lengths: list[int] | None = None
    # Online LeRobot frame data for this chunk; batched entries start with [B, ...].
    episode_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "rewards",
            "dones",
            "terminations",
            "truncations",
            "intervene_actions",
            "intervene_flags",
            "rlt_switch_flags",
            "reward_model_output",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())

    def with_trajectory_data(
        self,
        *,
        reward_model_output: torch.Tensor | None = None,
        reward_assign_lengths: list[int] | None = None,
        episode_data: dict[str, Any] | None = None,
    ) -> "EnvTransition":
        """Return a copy enriched with collector-facing metadata."""
        return replace(
            self,
            reward_model_output=reward_model_output,
            reward_assign_lengths=reward_assign_lengths,
            episode_data=episode_data,
        )

    def to_dict(self) -> dict[str, Any]:
        """Expose transition fields for existing environment batch consumers."""
        return {
            name: getattr(self, name)
            for name in (
                "dones",
                "terminations",
                "truncations",
                "rewards",
                "intervene_actions",
                "intervene_flags",
                "rlt_switch_flags",
            )
        }

    def split(self, split_sizes: list[int]) -> list["EnvTransition"]:
        """Split this environment transition on its batch dimension."""
        fields = {
            name: split_batch_value(getattr(self, name), split_sizes)
            for name in self.__dataclass_fields__
            if name != "episode_data"
        }
        episodes = split_episode_data(self.episode_data, split_sizes)
        return [
            type(self)(
                **{name: values[index] for name, values in fields.items()},
                episode_data=episodes[index],
            )
            for index in range(len(split_sizes))
        ]

    @classmethod
    def merge(cls, transitions: list["EnvTransition"]) -> "EnvTransition":
        """Merge source fragments in batch order."""
        if not transitions:
            raise ValueError("Cannot merge an empty list of environment transitions.")
        episode_data = [transition.episode_data for transition in transitions]
        if any(value is not None for value in episode_data) and not all(
            value is not None for value in episode_data
        ):
            raise ValueError("Cannot merge partial episode data.")
        return cls(
            **{
                name: merge_batch_values(
                    [getattr(transition, name) for transition in transitions]
                )
                for name in transitions[0].__dataclass_fields__
                if name != "episode_data"
            },
            episode_data=(
                merge_episode_data(episode_data)
                if all(value is not None for value in episode_data)
                else None
            ),
        )

    def compute_rewards(
        self,
        *,
        bootstrap_values: torch.Tensor | None,
        env_reward_weight: float,
        reward_weight: float,
        auto_reset: bool,
        bootstrap_type: str,
        gamma: float,
    ) -> torch.Tensor | None:
        """Build the reward tensor consumed by trajectory training.

        Reward-model scores are weighted into environment rewards first.
        Auto-reset truncations then receive the configured bootstrap value.

        The result has independent storage for later in-place reward updates.
        """
        rewards = self.rewards
        if rewards is None:
            return None
        if self.reward_model_output is not None:
            rewards = (
                env_reward_weight * rewards
                + reward_weight * self.reward_model_output.to(rewards.dtype)
            )
        else:
            rewards = rewards.clone()
        if bootstrap_values is None or not auto_reset or self.dones is None:
            return rewards

        terminal = self.truncations if bootstrap_type == "standard" else self.dones
        if terminal is None or not terminal[:, -1].any():
            return rewards

        mask = terminal[:, -1]
        rewards[mask, -1] += gamma * bootstrap_values[mask].reshape(-1).to(
            rewards.dtype
        )
        return rewards

    def assign_history_rewards(
        self,
        steps: list["TrajectoryStep"],
        *,
        reward_weight: float,
    ) -> None:
        """Assign delayed model rewards to the requested preceding steps."""
        if self.reward_assign_lengths is None:
            return
        if self.reward_model_output is None:
            raise ValueError("History reward assignment requires model rewards.")

        reward_steps = [step for step in steps if step.rewards is not None]
        reward = reward_weight * self.reward_model_output
        for env_id, length in enumerate(self.reward_assign_lengths):
            for offset in range(2, min(length, len(reward_steps)) + 1):
                reward_steps[-offset].add_reward(env_id, reward[env_id])


@dataclass(kw_only=True)
class PolicyOutput:
    """Policy inference output retained until its environment transition arrives."""

    # Model-ready training inputs; batched leaves are [B, ...], and ``action``
    # is normally flattened [B, D] where D = num_action_chunks * action_dim.
    forward_inputs: dict[str, Any]
    # Behavior-policy log probabilities, float [B, ...]; common layouts are
    # [B, D] and [B, C, A], selected by the model/loss implementation.
    prev_logprobs: torch.Tensor | None = None
    # Behavior-policy values, usually [B, 1] for chunk-level or [B, C] for
    # action-level ``actor.model.value_type``.
    prev_values: torch.Tensor | None = None
    # Model-side intervention mask, bool [B, C] before action-slot expansion.
    intervene_flags: torch.Tensor | None = None
    # Actor weight version per log-probability slot; shape matches prev_logprobs.
    versions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Move inference outputs to contiguous CPU storage for transport."""
        self.forward_inputs = put_tensor_device(
            self.forward_inputs,
            "cpu",
            detach=True,
        )
        for field_name in (
            "prev_logprobs",
            "prev_values",
            "intervene_flags",
            "versions",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.detach().cpu().contiguous())

    def split(self, split_sizes: list[int]) -> list["PolicyOutput"]:
        """Split model outputs on their leading batch dimension."""
        fields = {
            name: split_batch_value(getattr(self, name), split_sizes)
            for name in self.__dataclass_fields__
        }
        return [
            type(self)(**{name: values[index] for name, values in fields.items()})
            for index in range(len(split_sizes))
        ]

    @classmethod
    def merge(cls, outputs: list["PolicyOutput"]) -> "PolicyOutput":
        """Merge routed model-output fragments in batch order."""
        if not outputs:
            raise ValueError("Cannot merge an empty list of policy outputs.")
        return cls(
            **{
                name: merge_batch_values([getattr(output, name) for output in outputs])
                for name in outputs[0].__dataclass_fields__
            }
        )


@dataclass(kw_only=True)
class PolicyPart:
    """Policy-owned data for one or more routed action chunks.

    Exactly one of ``output`` and ``external_actions`` must be present.
    The latter represents an online intervention that skipped model inference.
    """

    # Logical source shards represented by this payload; sizes sum to B.
    sources: list[TrajectorySource]
    # Pre-action observations; nested leaves start with [B, ...].
    obs: dict[str, Any]
    # Full model inference output for training; absent for external actions.
    output: PolicyOutput | None = None
    # Smooth-intervention actions, float [B, C, A] or [B, D].
    external_actions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the policy variant and move payloads to CPU."""
        if (self.output is None) == (self.external_actions is None):
            raise ValueError(
                "PolicyPart requires exactly one of output and external_actions."
            )
        self.obs = put_tensor_device(self.obs, "cpu")
        if self.external_actions is not None:
            self.external_actions = self.external_actions.cpu().contiguous()

    @property
    def inferred(self) -> bool:
        """Return whether this part came from model inference."""
        return self.output is not None

    def split(self) -> list["PolicyPart"]:
        """Split this routed payload into one fragment per logical source."""
        split_sizes = [source.size for source in self.sources]
        observations = split_batch_value(self.obs, split_sizes)
        if self.output is None:
            actions = split_batch_value(self.external_actions, split_sizes)
            return [
                type(self)(
                    sources=[source],
                    obs=observations[index],
                    external_actions=actions[index],
                )
                for index, source in enumerate(self.sources)
            ]

        outputs = self.output.split(split_sizes)
        return [
            type(self)(
                sources=[source],
                obs=observations[index],
                output=outputs[index],
            )
            for index, source in enumerate(self.sources)
        ]

    @classmethod
    def merge(cls, fragments: list["PolicyPart"]) -> "PolicyPart":
        """Merge single-source fragments in their validated batch order."""
        if not fragments:
            raise ValueError("Cannot merge an empty list of policy parts.")
        inferred = [fragment.inferred for fragment in fragments]
        if any(inferred) and not all(inferred):
            raise ValueError("Cannot merge inferred and external policy parts.")
        full_sources = TrajectorySource.merge(
            [fragment.sources for fragment in fragments]
        )
        if len(full_sources) != 1:
            raise ValueError("Policy fragments must belong to one logical source.")
        observations = merge_batch_values([fragment.obs for fragment in fragments])
        if fragments[0].output is None:
            return cls(
                sources=full_sources,
                obs=observations,
                external_actions=merge_batch_values(
                    [fragment.external_actions for fragment in fragments]
                ),
            )
        return cls(
            sources=full_sources,
            obs=observations,
            output=PolicyOutput.merge([fragment.output for fragment in fragments]),
        )


@dataclass(kw_only=True)
class EnvPart:
    """Environment-owned half of one or more routed action chunks.

    Env creates this object after stepping and attaches it to the next
    :class:`PolicyInput`. Rollout adds terminal inference data with
    :meth:`complete` before publishing the same type to the Actor channel.
    """

    # Logical source shards represented by this payload; sizes sum to B.
    sources: list[TrajectorySource]
    # Outcome of the executed actions; tensors are normally [B, C, ...].
    transition: EnvTransition
    # Post-action observation override; leaves [B, ...]. Env sends it only when
    # the accompanying PolicyInput.obs cannot represent this boundary.
    next_obs: dict[str, Any] | None = None
    # Whether Rollout must infer on the effective post-action observation.
    requires_inference: bool = False
    # RLT features for the next state (z_rl, proprio, ref_chunk); leaves [B, ...].
    next_rlt_obs: dict[str, Any] | None = None
    # Value used for truncation bootstrap, float [B, 1], derived from final values.
    bootstrap_values: torch.Tensor | None = None
    # Full terminal value output, usually float [B, 1], appended at epoch end.
    final_prev_values: torch.Tensor | None = None
    # Pre-action boundary for chunk zero; tensors [B, C], otherwise ``None``.
    initial_transition: EnvTransition | None = None

    def __post_init__(self) -> None:
        """Move model-derived fields to CPU for transport."""
        if self.next_obs is not None:
            self.next_obs = put_tensor_device(self.next_obs, "cpu")
        if self.next_rlt_obs is not None:
            self.next_rlt_obs = put_tensor_device(self.next_rlt_obs, "cpu")
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.final_prev_values is not None:
            self.final_prev_values = self.final_prev_values.cpu().contiguous()

    def complete(
        self,
        *,
        next_obs: dict[str, Any] | None,
        next_rlt_obs: dict[str, Any] | None,
        final_prev_values: torch.Tensor | None = None,
    ) -> "EnvPart":
        """Return the Actor-ready part after resolving rollout-owned fields."""
        final_prev_values = (
            final_prev_values.cpu().contiguous()
            if final_prev_values is not None
            else None
        )
        return replace(
            self,
            requires_inference=False,
            next_obs=next_obs,
            next_rlt_obs=next_rlt_obs,
            bootstrap_values=(
                final_prev_values[:, :1] if final_prev_values is not None else None
            ),
            final_prev_values=final_prev_values,
        )

    def split(self, split_sizes: list[int] | None = None) -> list["EnvPart"]:
        """Split by explicit route sizes or by the represented sources."""
        split_sizes = split_sizes or [source.size for source in self.sources]
        source_shards = TrajectorySource.split(self.sources, split_sizes)
        transitions = self.transition.split(split_sizes)
        next_observations = split_batch_value(self.next_obs, split_sizes)
        next_rlt_observations = split_batch_value(self.next_rlt_obs, split_sizes)
        bootstrap_values = split_batch_value(self.bootstrap_values, split_sizes)
        final_prev_values = split_batch_value(self.final_prev_values, split_sizes)
        initial_transitions = (
            self.initial_transition.split(split_sizes)
            if self.initial_transition is not None
            else [None] * len(split_sizes)
        )
        return [
            type(self)(
                sources=source_shards[index],
                transition=transitions[index],
                next_obs=next_observations[index],
                requires_inference=self.requires_inference,
                next_rlt_obs=next_rlt_observations[index],
                bootstrap_values=bootstrap_values[index],
                final_prev_values=final_prev_values[index],
                initial_transition=initial_transitions[index],
            )
            for index in range(len(split_sizes))
        ]

    @classmethod
    def merge(cls, fragments: list["EnvPart"]) -> "EnvPart":
        """Merge single-source fragments in their validated batch order."""
        if not fragments:
            raise ValueError("Cannot merge an empty list of environment parts.")
        source = fragments[0].sources[0]
        full_sources = TrajectorySource.merge(
            [fragment.sources for fragment in fragments]
        )
        if len(full_sources) != 1:
            raise ValueError("Environment fragments must belong to one logical source.")
        inference_modes = [fragment.requires_inference for fragment in fragments]
        if any(inference_modes) and not all(inference_modes):
            raise ValueError(f"Conflicting terminal-inference state for {source.key}.")
        initial_transitions = [fragment.initial_transition for fragment in fragments]
        has_initial = [transition is not None for transition in initial_transitions]
        if any(has_initial) and not all(has_initial):
            raise ValueError(f"Incomplete initial-state fragments for {source.key}.")
        return cls(
            sources=full_sources,
            transition=EnvTransition.merge(
                [fragment.transition for fragment in fragments]
            ),
            next_obs=merge_batch_values([fragment.next_obs for fragment in fragments]),
            requires_inference=fragments[0].requires_inference,
            next_rlt_obs=merge_batch_values(
                [fragment.next_rlt_obs for fragment in fragments]
            ),
            bootstrap_values=merge_batch_values(
                [fragment.bootstrap_values for fragment in fragments]
            ),
            final_prev_values=merge_batch_values(
                [fragment.final_prev_values for fragment in fragments]
            ),
            initial_transition=(
                EnvTransition.merge(initial_transitions) if all(has_initial) else None
            ),
        )

    def compute_rewards(
        self,
        *,
        env_reward_weight: float,
        reward_weight: float,
        auto_reset: bool,
        bootstrap_type: str,
        gamma: float,
    ) -> torch.Tensor | None:
        """Return the training rewards represented by this environment part."""
        return self.transition.compute_rewards(
            bootstrap_values=self.bootstrap_values,
            env_reward_weight=env_reward_weight,
            reward_weight=reward_weight,
            auto_reset=auto_reset,
            bootstrap_type=bootstrap_type,
            gamma=gamma,
        )


TrajectoryPart: TypeAlias = PolicyPart | EnvPart


# Actor-facing structures assembled by the collector.


@dataclass(kw_only=True)
class TrajectoryStep:
    """Complete training data for one joined action chunk."""

    # Executed training actions, float [B, C, A] or flattened [B, D].
    actions: torch.Tensor | None = None
    # Per-action expert mask expanded to match ``actions``.
    intervene_flags: torch.Tensor | None = None
    # Behavior-policy log probabilities, float [B, ...], model/loss specific.
    prev_logprobs: torch.Tensor | None = None
    # Current values, usually [B, 1] or [B, C] from actor.model.value_type.
    prev_values: torch.Tensor | None = None
    # Post-action done flags, bool [B, C].
    dones: torch.Tensor | None = None
    # Post-action truncation flags, bool [B, C].
    truncations: torch.Tensor | None = None
    # Post-action termination flags, bool [B, C].
    terminations: torch.Tensor | None = None
    # Combined environment/model rewards, float [B, C].
    rewards: torch.Tensor | None = None
    # Actor training inputs for the current chunk; leaves start with [B, ...].
    forward_inputs: dict[str, Any] = field(default_factory=dict)
    # Actor weight version for each policy statistic; shape matches prev_logprobs.
    versions: torch.Tensor | None = None
    # Current transition observation; leaves start with [B, ...].
    curr_obs: dict[str, Any] = field(default_factory=dict)
    # Next transition observation; leaves start with [B, ...].
    next_obs: dict[str, Any] = field(default_factory=dict)
    # Epoch-start done boundary, bool [B, C], present only on chunk zero.
    initial_dones: torch.Tensor | None = None
    # Epoch-start truncation boundary, bool [B, C], present only on chunk zero.
    initial_truncations: torch.Tensor | None = None
    # Epoch-start termination boundary, bool [B, C], present only on chunk zero.
    initial_terminations: torch.Tensor | None = None
    # Terminal value appended after the last chunk, usually [B, 1] or [B, C].
    final_prev_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Normalize all retained training data to contiguous CPU storage."""
        for field_name in (
            "actions",
            "intervene_flags",
            "prev_logprobs",
            "prev_values",
            "dones",
            "terminations",
            "truncations",
            "rewards",
            "versions",
            "initial_dones",
            "initial_truncations",
            "initial_terminations",
            "final_prev_values",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())
        for field_name in ("forward_inputs", "curr_obs", "next_obs"):
            value = getattr(self, field_name)
            if value:
                setattr(self, field_name, put_tensor_device(value, "cpu"))

        if self.actions is not None and self.intervene_flags is None:
            self.intervene_flags = torch.zeros_like(self.actions, dtype=torch.bool)

    @classmethod
    def from_parts(
        cls,
        policy: PolicyPart,
        env: EnvPart,
        *,
        rewards: torch.Tensor | None,
        collect_prev_infos: bool,
        collect_transitions: bool,
        enable_rlt: bool,
        include_final_value: bool,
    ) -> "TrajectoryStep":
        """Resolve one joined policy/env pair into self-contained step data.

        This is the single conversion boundary for policy statistics,
        environment boundaries, interventions, and optional transitions.
        """
        output = policy.output
        if output is None:
            raise ValueError("Training trajectories require inferred policy output.")
        initial = env.initial_transition
        step = cls(
            actions=output.forward_inputs.get("action"),
            prev_logprobs=output.prev_logprobs if collect_prev_infos else None,
            prev_values=output.prev_values if collect_prev_infos else None,
            dones=env.transition.dones,
            truncations=env.transition.truncations,
            terminations=env.transition.terminations,
            rewards=rewards,
            forward_inputs=dict(output.forward_inputs),
            versions=output.versions,
            initial_dones=initial.dones if initial is not None else None,
            initial_truncations=(initial.truncations if initial is not None else None),
            initial_terminations=(
                initial.terminations if initial is not None else None
            ),
            final_prev_values=(
                env.final_prev_values
                if include_final_value and collect_prev_infos
                else None
            ),
        )
        if env.transition.intervene_actions is not None:
            step.apply_interventions(
                env.transition.intervene_actions,
                env.transition.intervene_flags,
            )
        if output.intervene_flags is not None:
            step.set_intervention_flags(output.intervene_flags)
        step.set_transition_observations(
            policy,
            env,
            collect_transitions=collect_transitions,
            enable_rlt=enable_rlt,
        )
        return step

    def apply_interventions(
        self,
        intervene_actions: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> None:
        """Replace executed actions and synchronize their training metadata."""
        if self.actions is None:
            return
        if intervene_flags is None:
            raise ValueError("Intervention actions require intervention flags.")
        if self.actions.dim() != 2 or intervene_actions.dim() != 2:
            raise ValueError("Intervention actions must use flattened 2D chunks.")
        if intervene_flags.dim() == 1:
            intervene_flags = intervene_flags[:, None]
        if intervene_flags.dim() != 2:
            raise ValueError("Intervention flags must have shape [batch, chunk].")

        batch_size, chunk_count = intervene_flags.shape
        flags = intervene_flags.to(torch.bool).reshape(batch_size, chunk_count, 1)
        model_actions = self.actions.reshape(batch_size, chunk_count, -1)
        expert_actions = intervene_actions.reshape(batch_size, chunk_count, -1)
        actions = expert_actions * flags + model_actions * (~flags)
        self.actions = actions.reshape(batch_size, -1).cpu().contiguous()
        self.intervene_flags = flags.expand_as(actions).reshape(batch_size, -1)
        if "action" in self.forward_inputs:
            self.forward_inputs["action"] = self.actions
        self.forward_inputs.pop("model_action", None)

    def set_intervention_flags(self, intervene_flags: torch.Tensor) -> None:
        """Expand chunk-level flags to the flattened action layout."""
        if self.actions is None:
            return
        if intervene_flags.dim() == 1:
            intervene_flags = intervene_flags[:, None]
        if intervene_flags.dim() != 2:
            raise ValueError("Intervention flags must have shape [batch, chunk].")

        batch_size, chunk_count = intervene_flags.shape
        expanded = intervene_flags.reshape(batch_size, chunk_count, 1).expand_as(
            self.actions.reshape(batch_size, chunk_count, -1)
        )
        self.intervene_flags = expanded.reshape(batch_size, -1).to(torch.bool)

    def set_transition_observations(
        self,
        policy: PolicyPart,
        env: EnvPart,
        *,
        collect_transitions: bool,
        enable_rlt: bool,
    ) -> None:
        """Extract the transition representation required by the algorithm."""
        if enable_rlt:
            from rlinf.algorithms.rlt.transition import (
                apply_rlt_interventions,
                extract_rlt_obs_from_forward_inputs,
            )

            if env.next_rlt_obs is None:
                raise ValueError("RLT transitions require next-state features.")
            current_obs = extract_rlt_obs_from_forward_inputs(self.forward_inputs)
            apply_rlt_interventions(
                current_obs,
                env.transition.intervene_actions,
                env.transition.intervene_flags,
            )
            next_obs = env.next_rlt_obs
        elif collect_transitions:
            if env.next_obs is None:
                raise ValueError("Raw transitions require a next observation.")
            current_obs = policy.obs
            next_obs = env.next_obs
        else:
            return

        # Task strings are rollout metadata, not tensor training inputs.
        self.curr_obs = {
            key: value
            for key, value in current_obs.items()
            if key != "task_descriptions"
        }
        self.next_obs = {
            key: value for key, value in next_obs.items() if key != "task_descriptions"
        }

    def add_reward(self, env_id: int, reward: torch.Tensor) -> None:
        """Add one delayed reward to an environment in this step."""
        if self.rewards is None:
            raise ValueError("Cannot assign history reward to a rewardless step.")
        self.rewards[env_id] += reward.to(self.rewards.dtype)


@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """

    max_episode_length: int = 0
    model_weights_id: str = ""
    actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    prev_logprobs: torch.Tensor | None = None
    prev_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)
    curr_obs: dict[str, Any] = field(default_factory=dict)
    next_obs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_steps(
        cls,
        steps: list[TrajectoryStep],
        *,
        max_episode_length: int = 0,
    ) -> "Trajectory":
        """Stack complete chunk steps into one actor-facing trajectory.

        Epoch-start boundaries and final bootstrap values are inserted in
        temporal order; ordinary fields retain one entry per action chunk.
        """
        trajectory = cls(max_episode_length=max_episode_length)

        def stack_step_field(name: str) -> torch.Tensor | None:
            values = [getattr(step, name) for step in steps]
            present = [value for value in values if value is not None]
            return torch.stack(present, dim=0).cpu().contiguous() if present else None

        for name in (
            "actions",
            "intervene_flags",
            "rewards",
            "prev_logprobs",
            "versions",
        ):
            setattr(trajectory, name, stack_step_field(name))

        # Boundary fields insert the state preceding every rollout epoch.
        for name, initial_name in (
            ("dones", "initial_dones"),
            ("truncations", "initial_truncations"),
            ("terminations", "initial_terminations"),
        ):
            values = []
            for step in steps:
                initial_value = getattr(step, initial_name)
                if initial_value is not None:
                    values.append(initial_value)
                value = getattr(step, name)
                if value is not None:
                    values.append(value)
            if values:
                setattr(
                    trajectory,
                    name,
                    torch.stack(values, dim=0).cpu().contiguous(),
                )

        values = []
        for step in steps:
            if step.prev_values is not None:
                values.append(step.prev_values)
            if step.final_prev_values is not None:
                values.append(step.final_prev_values)
        if values:
            trajectory.prev_values = torch.stack(values, dim=0).cpu().contiguous()

        for name in ("forward_inputs", "curr_obs", "next_obs"):
            values = [getattr(step, name) for step in steps if getattr(step, name)]
            if values:
                setattr(
                    trajectory,
                    name,
                    put_tensor_device(stack_list_of_dict_tensor(values), "cpu"),
                )

        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )
        return trajectory

    def split(self, split_sizes: int | list[int]) -> list["Trajectory"]:
        """Split the batch dimension into actor-consumable shards.

        An integer requests equal shards. A list provides each shard size
        explicitly, as required by pipeline actor routing.
        """
        if isinstance(split_sizes, int):
            batch_size = self._batch_size()
            if batch_size % split_sizes:
                raise ValueError(
                    f"Trajectory batch size {batch_size} is not divisible by "
                    f"{split_sizes} shards."
                )
            split_sizes = [batch_size // split_sizes] * split_sizes
        elif sum(split_sizes) != self._batch_size():
            raise ValueError("Trajectory split sizes do not match its batch size.")

        trajectories = [type(self)() for _ in split_sizes]
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, (int, str)):
                for trajectory in trajectories:
                    setattr(trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                for trajectory, shard in zip(
                    trajectories,
                    torch.split(value, split_sizes, dim=1),
                ):
                    setattr(trajectory, field_name, shard.contiguous())
            elif isinstance(value, dict):
                for trajectory, shard in zip(
                    trajectories,
                    split_dict(value, split_sizes, dim=1),
                ):
                    setattr(trajectory, field_name, shard)
            else:
                raise TypeError(
                    f"Unsupported trajectory field {field_name}: {type(value)}"
                )
        return trajectories

    @classmethod
    def to_batch(cls, trajectories: list["Trajectory"]) -> dict[str, Any]:
        """Concatenate trajectories into one ``[T, B, ...]`` Actor batch."""
        if not trajectories:
            return {}

        batch: dict[str, Any] = {}
        for field_name in ("curr_obs", "next_obs", "forward_inputs"):
            all_keys = {
                key
                for trajectory in trajectories
                for key in getattr(trajectory, field_name)
            }
            if all_keys:
                batch[field_name] = {
                    key: torch.cat(
                        [
                            getattr(trajectory, field_name)[key]
                            for trajectory in trajectories
                            if key in getattr(trajectory, field_name)
                        ],
                        dim=1,
                    )
                    for key in all_keys
                }

        for field_name in trajectories[0].__dataclass_fields__:
            if not isinstance(getattr(trajectories[0], field_name), torch.Tensor):
                continue
            values = [
                getattr(trajectory, field_name)
                for trajectory in trajectories
                if getattr(trajectory, field_name) is not None
            ]
            if values:
                batch[field_name] = torch.cat(values, dim=1)
        return batch

    def extract_intervene_traj(self, mode="any"):
        if self.intervene_flags is None or (~self.intervene_flags).all():
            return None
        if mode == "any":
            mask = self.intervene_flags.any(dim=-1)
        elif mode == "all":
            mask = self.intervene_flags.all(dim=-1)
        else:
            raise NotImplementedError(
                f"Unsupported extract_intervene_traj mode: {mode}"
            )
        assert mask.dim() == 2, (
            f"Expected 2D mask after processing (traj len, bsz), got {mask.shape=}"
        )
        traj_len = int(mask.shape[0])

        def apply_mask(tensor, i):
            return tensor[:, i][mask[:, i]].unsqueeze(1) if tensor is not None else None

        def apply_mask_to_dict(d, i):
            return (
                {k: v[:, i][mask[:, i]].unsqueeze(1) for k, v in d.items()} if d else {}
            )

        filtered_trajectories = []
        for i in range(mask.shape[1]):
            if not mask[:, i].any():
                continue
            actions = apply_mask(self.actions, i)
            rewards = apply_mask(self.rewards, i)
            prev_logprobs = apply_mask(self.prev_logprobs, i)
            prev_values = apply_mask(self.prev_values, i)
            intervene_flags = apply_mask(self.intervene_flags, i)
            forward_inputs = apply_mask_to_dict(self.forward_inputs, i)
            curr_obs = apply_mask_to_dict(self.curr_obs, i)
            next_obs = apply_mask_to_dict(self.next_obs, i)
            terminations = truncations = dones = None
            if self.terminations is not None:
                field_mask = self._generate_field_mask(
                    self.terminations[:, i : i + 1], mask[:, i], traj_len
                )
                terminations = self.terminations[:, i : i + 1][field_mask]
                truncations = self.truncations[:, i : i + 1][field_mask]
                dones = self.dones[:, i : i + 1][field_mask]
            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=actions,
                    intervene_flags=intervene_flags,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    dones=dones,
                    prev_logprobs=prev_logprobs,
                    prev_values=prev_values,
                    forward_inputs=forward_inputs,
                    curr_obs=curr_obs,
                    next_obs=next_obs,
                )
            )
        return filtered_trajectories if filtered_trajectories else None

    def _batch_size(self) -> int:
        """Return the trajectory batch dimension."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                return value.shape[1]
            if isinstance(value, dict):
                for nested_value in value.values():
                    if isinstance(nested_value, torch.Tensor):
                        return nested_value.shape[1]
        raise ValueError("Cannot infer the batch size of an empty trajectory.")

    @staticmethod
    def _generate_field_mask(
        ref_tensor: torch.Tensor, mask: torch.Tensor, traj_len: int
    ) -> torch.Tensor:
        """
        Generate a mask for terminations/truncations/dones based on their original shape.
        """
        assert mask.dim() == 1, f"Expected 1D mask, got {mask.shape=}"
        if ref_tensor.shape[0] == traj_len:
            return mask
        if ref_tensor.shape[0] > traj_len:
            extra = int(ref_tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} "
                "for terminations/truncations/dones"
            )
            epoch_len = traj_len // extra
            field_mask = torch.zeros(
                ref_tensor.shape[0], dtype=torch.bool, device=mask.device
            )
            original_indices = torch.arange(ref_tensor.shape[0], device=mask.device)
            epoch_idx = original_indices // (epoch_len + 1)
            step_idx = original_indices % (epoch_len + 1)
            field_mask[step_idx == 0] = True
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)
            return field_mask
        raise ValueError(
            f"Reference tensor length {ref_tensor.shape[0]} < traj_len {traj_len}"
        )


# Canonical online LeRobot data used by the episode accumulator.


@dataclass(kw_only=True)
class LeRobotStep:
    """One environment step decoded from a vectorized action chunk."""

    # Environment index in [0, B).
    env_index: int
    # Observation associated with the executed action; environment-specific.
    observation: Any
    # Metadata for the same step after selecting this environment.
    info: Any
    # Executed policy action, float [A], before an intervention override.
    action: np.ndarray | None
    # Whether the environment terminated naturally after this action.
    terminated: bool
    # Whether a time limit or external condition truncated the environment.
    truncated: bool
    # Auto-reset observation for the next episode, if returned in the same step.
    reset_observation: Any | None = None
    # Metadata accompanying ``reset_observation``.
    reset_info: Any | None = None

    @property
    def done(self) -> bool:
        """Return whether this step closes the current episode."""
        return self.terminated or self.truncated

    def info_flag(self, key: str) -> bool:
        """Return whether one recording-control flag contains a true value."""
        if not isinstance(self.info, dict) or self.info.get(key) is None:
            return False
        value = self.info[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return bool(np.asarray(value).any())


@dataclass(kw_only=True)
class LeRobotChunk:
    """Canonical vectorized data for one online LeRobot action chunk."""

    # Executed actions, float [B, C, A], or ``None`` when no frame can be built.
    actions: np.ndarray | None
    # Observations for the C low-level steps; leaves are batched [B, ...].
    observations: tuple[Any, ...]
    # Natural termination flags, bool [B, C], [B], or scalar.
    terminations: np.ndarray
    # Truncation flags with the same layouts as ``terminations``.
    truncations: np.ndarray
    # Per-step environment metadata; nested leaves are batched [B, ...].
    infos: tuple[Any, ...]
    # Policy-side expert actions, float [B, C, A], used when Env omits them.
    intervention_actions: np.ndarray | None
    # Policy-side expert masks, bool [B, C].
    intervention_flags: np.ndarray | None
    # Number of parallel environments represented by the leading batch B.
    num_envs: int
    # Single-action width A from ``actor.model.action_dim``.
    action_dim: int

    @classmethod
    def from_data(
        cls,
        *,
        policy_output: PolicyOutput | None,
        chunk_actions: Any,
        obs_list: Any,
        terminations: Any,
        truncations: Any,
        infos_list: Any,
        num_envs: int,
        num_action_chunks: int,
        action_dim: int,
    ) -> "LeRobotChunk":
        """Normalize transport payloads to one canonical LeRobot chunk."""
        observations = (
            tuple(obs_list) if isinstance(obs_list, (list, tuple)) else (obs_list,)
        )
        if isinstance(infos_list, (list, tuple)):
            infos = tuple(infos_list)
        else:
            infos = (infos_list,) * len(observations)
        if len(infos) != len(observations):
            raise ValueError("LeRobot infos must contain one entry per chunk step.")

        intervention_flags = (
            policy_output.intervene_flags if policy_output is not None else None
        )
        if intervention_flags is not None:
            intervention_flags = cls._to_numpy(intervention_flags).reshape(
                num_envs, num_action_chunks
            )
        intervention_actions = (
            policy_output.forward_inputs.get("action")
            if policy_output is not None
            else None
        )
        return cls(
            actions=cls._reshape_actions(
                chunk_actions,
                num_envs=num_envs,
                num_action_chunks=num_action_chunks,
                action_dim=action_dim,
            ),
            observations=observations,
            terminations=cls._to_numpy(terminations),
            truncations=cls._to_numpy(truncations),
            infos=infos,
            intervention_actions=cls._reshape_actions(
                intervention_actions,
                num_envs=num_envs,
                num_action_chunks=num_action_chunks,
                action_dim=action_dim,
            ),
            intervention_flags=intervention_flags,
            num_envs=num_envs,
            action_dim=action_dim,
        )

    @property
    def step_count(self) -> int:
        """Return the number of low-level steps represented by this chunk."""
        return len(self.observations)

    def step(self, step_index: int, env_index: int) -> LeRobotStep:
        """Decode one environment step, including auto-reset terminal data."""
        step_observation = self.observations[step_index]
        step_info = self._step_info(step_index)
        terminated = self._flag_at(self.terminations, step_index, env_index)
        truncated = self._flag_at(self.truncations, step_index, env_index)
        done = terminated or truncated

        has_final_observation = (
            isinstance(step_info, dict) and "final_observation" in step_info
        )
        if has_final_observation and done:
            info_without_reset = copy.deepcopy(step_info)
            final_observation = info_without_reset.pop("final_observation")
            final_info = info_without_reset.pop("final_info")
            observation = self._slice_env(final_observation, env_index, self.num_envs)
            info = self._slice_env(final_info, env_index, self.num_envs)
            reset_observation = self._slice_env(
                step_observation, env_index, self.num_envs
            )
            reset_info = self._slice_env(info_without_reset, env_index, self.num_envs)
        else:
            observation = self._slice_env(step_observation, env_index, self.num_envs)
            info = self._slice_env(step_info, env_index, self.num_envs)
            if isinstance(info, dict):
                info = copy.deepcopy(info)
                info.pop("final_observation", None)
                info.pop("final_info", None)
            reset_observation = None
            reset_info = None

        action = None
        if self.actions is not None:
            action = self.actions[env_index, min(step_index, self.actions.shape[1] - 1)]
        return LeRobotStep(
            env_index=env_index,
            observation=observation,
            info=info,
            action=action,
            terminated=terminated,
            truncated=truncated,
            reset_observation=reset_observation,
            reset_info=reset_info,
        )

    def _step_info(self, step_index: int) -> Any:
        """Copy one info batch and add missing policy intervention metadata."""
        info = copy.deepcopy(self.infos[step_index])
        if (
            not isinstance(info, dict)
            or self.intervention_actions is None
            or self.intervention_flags is None
            or "intervene_action" in info
        ):
            return info

        step_actions = self.intervention_actions[:, step_index]
        step_flags = self.intervention_flags[:, step_index]
        if "final_info" in info:
            info["final_info"]["intervene_action"] = self.intervention_actions
            info["final_info"]["intervene_flag"] = self.intervention_flags
            info["intervene_action"] = step_actions
            terminations = self._step_values(self.terminations, step_index)
            truncations = self._step_values(self.truncations, step_index)
            info["intervene_flag"] = step_flags & ~(terminations | truncations)
        else:
            info["intervene_action"] = step_actions
            info["intervene_flag"] = step_flags
        return info

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        """Convert tensor-like data to a detached NumPy array."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _reshape_actions(
        cls,
        actions: Any,
        *,
        num_envs: int,
        num_action_chunks: int,
        action_dim: int,
    ) -> np.ndarray | None:
        """Normalize flat or chunked actions to [B, C, A]."""
        array = cls._to_numpy(actions)
        if array is None:
            return None
        flat_dim = num_action_chunks * action_dim
        if array.ndim == 3:
            return array
        if array.ndim == 2 and array.shape[-1] == flat_dim:
            return array.reshape(array.shape[0], num_action_chunks, action_dim)
        if array.ndim == 2 and array.shape[-1] == action_dim:
            return array[:, None, :]
        raise ValueError(
            f"Unexpected chunk action shape {array.shape}; expected "
            f"[{num_envs}, {num_action_chunks}, {action_dim}] or flat dim "
            f"{flat_dim}."
        )

    @staticmethod
    def _slice_env(value: Any, env_index: int, num_envs: int) -> Any:
        """Select one environment from a recursively nested batch."""
        if isinstance(value, torch.Tensor):
            return (
                value[env_index]
                if value.dim() > 0 and value.shape[0] == num_envs
                else value
            )
        if isinstance(value, np.ndarray):
            return (
                value[env_index]
                if value.ndim > 0 and value.shape[0] == num_envs
                else value
            )
        if isinstance(value, dict):
            return {
                key: LeRobotChunk._slice_env(item, env_index, num_envs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return value[env_index] if len(value) == num_envs else value
        return value

    @staticmethod
    def _step_values(values: np.ndarray, step_index: int) -> np.ndarray:
        """Select one low-level step while retaining its environment batch."""
        return values[:, step_index] if values.ndim > 1 else values

    def _flag_at(self, values: np.ndarray, step_index: int, env_index: int) -> bool:
        """Read one environment flag from scalar, [B], or [B, C] layouts."""
        step_values = self._step_values(values, step_index)
        if step_values.ndim > 0 and step_values.shape[0] == self.num_envs:
            return bool(step_values[env_index])
        return bool(step_values.item())


@dataclass(kw_only=True)
class LeRobotFrame:
    """Canonical frame stored in one online LeRobot episode."""

    # Proprioceptive state, float32 [state_dim].
    state: np.ndarray
    # Actual executed action after intervention, float32 [A].
    action: np.ndarray
    # Language task description associated with this frame.
    task: str
    # Camera fields such as image, wrist_image, or extra_view_image-N.
    images: dict[str, np.ndarray] = field(default_factory=dict)
    # Whether an expert action replaced the policy action.
    intervene_flag: bool = False
    # Real-world recording segment local to the current episode.
    segment_id: int = 0
    # Success reported by this frame, or ``None`` when unavailable.
    step_success: bool | None = None

    @classmethod
    def from_step(
        cls,
        *,
        observation: Any,
        action: np.ndarray | None,
        info: Any,
        segment_id: int,
        action_dim: int,
    ) -> "LeRobotFrame | None":
        """Build one canonical frame from environment-specific step data."""
        if action is None:
            return None
        action, intervene_flag = cls._executed_action(action, info, action_dim)
        return cls.from_values(
            observation=observation,
            action=action,
            task=None,
            intervene_flag=intervene_flag,
            segment_id=segment_id,
            step_success=cls.success_from_info(info),
        )

    @classmethod
    def from_values(
        cls,
        *,
        observation: Any,
        action: Any,
        task: str | None,
        intervene_flag: bool,
        segment_id: int,
        step_success: bool | None = None,
    ) -> "LeRobotFrame | None":
        """Build a frame from an already resolved action and intervention flag."""
        image, wrist_image, extra_view_image, state, observed_task = (
            cls._extract_observation(observation)
        )
        if state is None or action is None:
            return None

        images = {}
        if image is not None:
            images["image"] = cls._to_uint8(np.asarray(image))
        for key, value in cls._expand_images("wrist_image", wrist_image).items():
            images[key] = cls._to_uint8(np.asarray(value))
        for key, value in cls._expand_images(
            "extra_view_image", extra_view_image
        ).items():
            images[key] = cls._to_uint8(np.asarray(value))

        return cls(
            state=np.asarray(state).astype(np.float32),
            action=np.asarray(action).astype(np.float32).flatten(),
            task=observed_task if task is None else task,
            images=images,
            intervene_flag=intervene_flag,
            segment_id=segment_id,
            step_success=step_success,
        )

    def to_dict(self, *, episode_success: bool, done: bool) -> dict[str, Any]:
        """Return the frame schema consumed by the LeRobot dataset writer."""
        return {
            "state": self.state,
            "actions": self.action,
            "task": self.task,
            "is_success": np.array([episode_success], dtype=bool),
            "done": np.array([done], dtype=bool),
            "intervene_flag": np.array([self.intervene_flag], dtype=bool),
            "segment_id": np.array([self.segment_id], dtype=np.uint8),
            **self.images,
        }

    @staticmethod
    def success_from_info(info: Any) -> bool | None:
        """Read success from common top-level or episode info fields."""
        if not isinstance(info, dict):
            return None
        for source in (info.get("episode"), info):
            if not isinstance(source, dict):
                continue
            for key in ("success_once", "success_at_end", "success"):
                value = source.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                return bool(np.asarray(value).reshape(-1).any())
        return None

    @staticmethod
    def _extract_observation(
        observation: Any,
    ) -> tuple[Any, Any, Any, Any, str]:
        """Extract camera, state, and task fields from known observation schemas."""
        if not isinstance(observation, dict):
            return None, None, None, None, "unknown task"
        image = observation.get(
            "main_images", observation.get("image", observation.get("full_image"))
        )
        wrist_image = observation.get("wrist_images", observation.get("wrist_image"))
        extra_view_image = observation.get(
            "extra_view_images", observation.get("extra_view_image")
        )
        state = observation.get("states", observation.get("state"))
        task = observation.get("task_descriptions", "unknown task")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else "unknown task"
        return (
            LeRobotChunk._to_numpy(image),
            LeRobotChunk._to_numpy(wrist_image),
            LeRobotChunk._to_numpy(extra_view_image),
            LeRobotChunk._to_numpy(state),
            str(task),
        )

    @staticmethod
    def _executed_action(
        action: np.ndarray, info: Any, action_dim: int
    ) -> tuple[np.ndarray, bool]:
        """Apply an environment-recorded expert action when intervention is active."""
        if not isinstance(info, dict) or not {
            "intervene_action",
            "intervene_flag",
        }.issubset(info):
            return action, False

        intervention_action = LeRobotChunk._to_numpy(info["intervene_action"])
        intervention_flag = LeRobotChunk._to_numpy(info["intervene_flag"])
        if intervention_action.size > action_dim:
            chunk_count = intervention_action.reshape(-1, action_dim).shape[0]
            intervention_action = intervention_action.reshape(-1, action_dim)[-1]
            intervention_flag = intervention_flag.reshape(chunk_count, -1)[-1, 0]
        else:
            intervention_action = intervention_action.reshape(-1)[:action_dim]
        is_intervened = bool(
            np.asarray(intervention_flag, dtype=bool).reshape(-1).any()
        )
        return (
            intervention_action if is_intervened else action,
            is_intervened,
        )

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        """Convert normalized or byte-scale image data to uint8."""
        if array.dtype == np.uint8:
            return array
        return (
            (array * 255).astype(np.uint8)
            if array.max() <= 1.0
            else array.astype(np.uint8)
        )

    @staticmethod
    def _expand_images(
        base_key: str, images: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Map one or several camera arrays to stable LeRobot frame keys."""
        if images is None:
            return {}
        if images.ndim == 3:
            return {base_key: images}
        if images.ndim == 4:
            if images.shape[0] == 1:
                return {base_key: images[0]}
            return {f"{base_key}-{index}": image for index, image in enumerate(images)}
        return {base_key: images}


# Standard env-to-rollout transport messages.


@dataclass(kw_only=True)
class EnvOutput:
    """Observation and transition returned by one environment chunk step."""

    # Post-step observations; nested tensors/arrays start with [B, ...].
    obs: dict[str, Any]
    # Outcome of the actions that produced ``obs``; tensors are usually [B, C].
    transition: EnvTransition = field(default_factory=EnvTransition)
    # Pre-reset observations for finished envs; same schema/shape as ``obs``.
    final_obs: dict[str, Any] | None = None
    # Raw environment metadata; batched values start with [B, ...].
    env_infos: dict[str, Any] | None = None

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )

    @property
    def rewards(self) -> torch.Tensor | None:
        """Return per-action environment rewards from the transition."""
        return self.transition.rewards

    @property
    def dones(self) -> torch.Tensor | None:
        """Return combined termination/truncation flags."""
        return self.transition.dones

    @property
    def terminations(self) -> torch.Tensor | None:
        """Return natural terminal flags."""
        return self.transition.terminations

    @property
    def truncations(self) -> torch.Tensor | None:
        """Return time-limit or external truncation flags."""
        return self.transition.truncations

    @property
    def intervene_actions(self) -> torch.Tensor | None:
        """Return expert actions recorded by the environment."""
        return self.transition.intervene_actions

    @property
    def intervene_flags(self) -> torch.Tensor | None:
        """Return which action slots used an expert action."""
        return self.transition.intervene_flags

    @property
    def rlt_switch_flags(self) -> torch.Tensor | None:
        """Return per-action RLT route choices."""
        return self.transition.rlt_switch_flags

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs": self.prepare_observations(self.obs),
            "final_obs": (
                self.prepare_observations(self.final_obs)
                if self.final_obs is not None
                else None
            ),
            "env_infos": self.env_infos,
            **self.transition.to_dict(),
        }


@dataclass(kw_only=True)
class PolicyInput:
    """One routed policy request, optionally carrying completed env data."""

    # Policy observations; nested leaves start with [B, ...].
    obs: dict[str, Any]
    # RLT route selected per env/action slot, usually bool/int [B] or [B, C].
    rlt_switch_flags: torch.Tensor | None = None
    # Requested expert-intervention slots, bool [B, C].
    intervene_flags: torch.Tensor | None = None
    # Logical source shards spanning this request; sizes sum to B.
    sources: list[TrajectorySource] = field(default_factory=list)
    # Previous environment parts awaiting rollout-side completion; one per source
    # request before merge, with ``None`` for a bootstrap-only request.
    env_parts: list[EnvPart | None] = field(default_factory=list)
    # Batch sizes of merged producer requests; values sum to B.
    request_sizes: list[int] = field(default_factory=list)
    # Whether this message closes an epoch without requesting another action.
    is_last: bool = False
    # Externally supplied actions, float [B, C, A] or [B, D]. ``None`` means
    # Rollout must run model inference. Used by smooth intervention only.
    external_actions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.obs = put_tensor_device(self.obs, "cpu")
        for name in ("rlt_switch_flags", "intervene_flags"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.cpu().contiguous())
        if self.external_actions is not None:
            self.external_actions = self.external_actions.cpu().contiguous()

    @property
    def requires_inference(self) -> bool:
        """Return whether Rollout must infer actions for this request."""
        return self.external_actions is None

    def split(self, split_sizes: list[int]) -> list["PolicyInput"]:
        """Split this producer request on its batch dimension."""
        if len(self.env_parts) > 1:
            raise ValueError(
                "A producer policy input cannot contain merged environment parts."
            )

        source_shards = TrajectorySource.split(self.sources, split_sizes)
        rlt_shards = split_batch_value(self.rlt_switch_flags, split_sizes)
        intervene_shards = split_batch_value(self.intervene_flags, split_sizes)
        env_part = self.env_parts[0] if self.env_parts else None
        env_part_shards = (
            env_part.split(split_sizes)
            if env_part is not None
            else [None] * len(split_sizes)
        )
        action_shards = split_batch_value(self.external_actions, split_sizes)
        return [
            type(self)(
                obs=obs,
                rlt_switch_flags=rlt_shards[index],
                intervene_flags=intervene_shards[index],
                sources=source_shards[index],
                env_parts=[env_part_shards[index]],
                request_sizes=[split_sizes[index]],
                is_last=self.is_last,
                external_actions=action_shards[index],
            )
            for index, obs in enumerate(split_batch_value(self.obs, split_sizes))
        ]

    @classmethod
    def merge(cls, policy_inputs: list["PolicyInput"]) -> "PolicyInput":
        """Merge routed policy requests in source order."""
        if not policy_inputs:
            raise ValueError("Cannot merge an empty list of policy inputs.")
        inference_modes = [item.requires_inference for item in policy_inputs]
        if any(inference_modes) and not all(inference_modes):
            raise ValueError("Cannot merge inferred and external policy inputs.")

        observations = [policy_input.obs for policy_input in policy_inputs]

        def merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [getattr(item, field_name) for item in policy_inputs]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                reference = next(value for value in values if value is not None)
                values = [
                    value
                    if value is not None
                    else torch.zeros(
                        (_observation_batch_size(obs), *reference.shape[1:]),
                        dtype=reference.dtype,
                    )
                    for obs, value in zip(observations, values)
                ]
            return merge_batch_values(values)

        return cls(
            obs=merge_batch_values(observations),
            rlt_switch_flags=merge_optional_tensor("rlt_switch_flags"),
            intervene_flags=merge_optional_tensor("intervene_flags"),
            sources=TrajectorySource.merge(
                [policy_input.sources for policy_input in policy_inputs]
            ),
            env_parts=[
                env_part
                for policy_input in policy_inputs
                for env_part in policy_input.env_parts
            ],
            request_sizes=[
                size
                for policy_input in policy_inputs
                for size in policy_input.request_sizes
            ],
            is_last=policy_inputs[0].is_last,
            external_actions=(
                None
                if policy_inputs[0].requires_inference
                else merge_batch_values(
                    [item.external_actions for item in policy_inputs]
                )
            ),
        )


# Real-time correction transport messages.


@dataclass(kw_only=True)
class RTCRequest:
    """Real-time correction request sent from the env worker to rollout."""

    obs: dict[str, Any]
    request_type: str = "bootstrap"
    executed_horizon: int = 0
    predicted_delay_steps: int = 0
    chunk_id: int = 0

    def __post_init__(self):
        # Keep Ray channel payloads on CPU so the control node never receives
        # CUDA tensors from the rollout node.
        self.obs = put_tensor_device(self.obs, "cpu")


@dataclass(kw_only=True)
class RTCActionResponse:
    """RTC response carrying a fresh action chunk."""

    actions: torch.Tensor | None = None
    model_actions: torch.Tensor | None = None
    chunk_id: int = 0
    guidance_applied: bool = False

    def __post_init__(self):
        # Actions are executed by the env worker, while model_actions are kept
        # for the next RTC overlap constraint.
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.model_actions is not None:
            self.model_actions = self.model_actions.cpu().contiguous()


# Shared batch and identifier utilities.


def get_model_weights_id(versions: torch.Tensor) -> str:
    """
    Get the model weights id from the tensor.

    Args:
        versions (torch.Tensor): The tensor to get the model weights id from.

    Returns:
        str: The model weights id.
    """
    name_bytes = versions.cpu().numpy().tobytes()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_bytes.hex()))


def split_batch_value(value: Any, split_sizes: list[int]) -> list[Any]:
    """Split a recursively nested batch on its leading dimension."""
    if value is None:
        return [None] * len(split_sizes)
    if isinstance(value, torch.Tensor):
        return [chunk.contiguous() for chunk in torch.split(value, split_sizes, dim=0)]
    if isinstance(value, np.ndarray):
        return list(np.split(value, np.cumsum(split_sizes)[:-1], axis=0))
    if isinstance(value, dict):
        chunks = [{} for _ in split_sizes]
        for key, item in value.items():
            for chunk, split_item in zip(chunks, split_batch_value(item, split_sizes)):
                chunk[key] = split_item
        return chunks
    if isinstance(value, list):
        offset = 0
        chunks = []
        for size in split_sizes:
            chunks.append(value[offset : offset + size])
            offset += size
        return chunks
    if isinstance(value, (bool, float, int, str)):
        # Scalars describe the whole batch and remain identical across shards.
        return [value] * len(split_sizes)
    raise TypeError(f"Unsupported batch value: {type(value)}")


def merge_batch_values(values: list[Any]) -> Any:
    """Merge recursively nested batches on their leading dimension."""
    if not values:
        raise ValueError("Cannot merge an empty list of batch values.")
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Cannot merge present and absent batch values.")

    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values[1:]):
            raise ValueError("Cannot merge batch dictionaries with different keys.")
        return {
            key: merge_batch_values([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [item for value in values for item in value]
    if isinstance(first, (bool, float, int, str)):
        if any(value != first for value in values[1:]):
            raise ValueError(f"Cannot merge conflicting scalar batch values: {values}.")
        return first
    raise TypeError(f"Unsupported batch value: {type(first)}")


def split_episode_data(
    data: dict[str, Any] | None, split_sizes: list[int]
) -> list[dict[str, Any] | None]:
    """Split online LeRobot chunk data without splitting its time dimension."""
    if data is None:
        return [None] * len(split_sizes)

    def split_steps(values: list[Any]) -> list[list[Any]]:
        chunks = [[] for _ in split_sizes]
        for value in values:
            for chunk, split_item in zip(chunks, split_batch_value(value, split_sizes)):
                chunk.append(split_item)
        return chunks

    return [
        {
            "chunk_actions": chunk_actions,
            "obs_list": obs_list,
            "terminations": terminations,
            "truncations": truncations,
            "infos_list": infos_list,
        }
        for chunk_actions, obs_list, terminations, truncations, infos_list in zip(
            split_batch_value(data["chunk_actions"], split_sizes),
            split_steps(data["obs_list"]),
            split_batch_value(data["terminations"], split_sizes),
            split_batch_value(data["truncations"], split_sizes),
            split_steps(data["infos_list"]),
        )
    ]


def merge_episode_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge online LeRobot chunk data without merging its time dimension."""
    if not data:
        raise ValueError("Cannot merge an empty list of episode data.")

    def merge_steps(values: list[list[Any]]) -> list[Any]:
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise ValueError("Cannot merge episode data with different chunk lengths.")
        return [merge_batch_values(list(items)) for items in zip(*values)]

    return {
        "chunk_actions": merge_batch_values([value["chunk_actions"] for value in data]),
        "obs_list": merge_steps([value["obs_list"] for value in data]),
        "terminations": merge_batch_values([value["terminations"] for value in data]),
        "truncations": merge_batch_values([value["truncations"] for value in data]),
        "infos_list": merge_steps([value["infos_list"] for value in data]),
    }


def _observation_batch_size(obs: dict[str, Any]) -> int:
    """Infer the leading batch size from a policy observation."""
    for key in ("states", "main_images", "task_descriptions"):
        value = obs.get(key)
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return value.shape[0]
        if isinstance(value, list):
            return len(value)
    raise ValueError("Cannot infer batch size from policy input observations.")


__all__ = [
    "TrajectoryKey",
    "TrajectorySource",
    "EnvTransition",
    "PolicyOutput",
    "PolicyPart",
    "EnvPart",
    "TrajectoryPart",
    "TrajectoryStep",
    "Trajectory",
    "LeRobotStep",
    "LeRobotChunk",
    "LeRobotFrame",
    "EnvOutput",
    "PolicyInput",
    "RTCRequest",
    "RTCActionResponse",
    "get_model_weights_id",
    "merge_batch_values",
    "merge_episode_data",
    "split_batch_value",
    "split_episode_data",
]
