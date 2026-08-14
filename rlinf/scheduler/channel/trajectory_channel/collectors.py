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

"""Collection strategies for assembled embodied trajectory chunks."""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
    EmbodiedTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    TrajectoryKey,
    convert_trajectories_to_batch,
)
from rlinf.scheduler.channel.trajectory_channel.assembler import AssembledChunk
from rlinf.scheduler.channel.trajectory_channel.data import (
    DummyPolicyStep,
    PolicyStep,
)
from rlinf.scheduler.worker.routing import CommMapper
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)

SourceID: TypeAlias = tuple[int, int]
CollectionScope: TypeAlias = int | tuple[int, int]


@dataclass(kw_only=True)
class CollectorOutput:
    """One collector result and the worker queue that should receive it."""

    queue_key: str
    data: Any


class TrajectoryCollector(ABC):
    """Accumulate assembled chunks and emit consumer-ready data."""

    @abstractmethod
    def collect(self, chunk: AssembledChunk) -> list[CollectorOutput]:
        """Consume one complete chunk and emit outputs for completed scopes."""


class _TrajectoryBuilderCollector(TrajectoryCollector):
    """Shared trajectory-building behavior for rollout and pipeline modes."""

    def __init__(self, cfg: DictConfig, source_count: int, chunk_count: int):
        self._cfg = cfg
        self._source_count = source_count
        self._chunk_count = chunk_count
        self._completed_keys: set[TrajectoryKey] = set()
        self._collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)
        self._collect_transitions = cfg.rollout.get("collect_transitions", False)
        self._enable_rlt = cfg.algorithm.get("loss_type") == "rlt_ac"
        self._env_reward_weight = cfg.get("reward", {}).get("env_reward_weight", 1.0)
        self._reward_weight = cfg.get("reward", {}).get("reward_weight", 1.0)

    def collect(self, chunk: AssembledChunk) -> list[CollectorOutput]:
        """Append a policy chunk and flush its scope when complete."""
        if isinstance(chunk.policy, DummyPolicyStep):
            raise ValueError(
                "DummyPolicyStep is only supported for online LeRobot DAgger."
            )
        if chunk.key in self._completed_keys:
            raise ValueError(f"Received duplicate trajectory event for {chunk.key}.")

        scope = self._scope(chunk.key)
        builders = self._builders_for(scope)
        builder = builders.setdefault(
            chunk.source,
            EmbodiedTrajectoryBuilder(
                max_episode_length=self._cfg.env.train.max_episode_steps
            ),
        )
        self._append_chunk(builder, chunk)
        if chunk.key.chunk_id == self._chunk_count - 1 and self._collect_prev_infos:
            builder.append_final_value(chunk.env.final_prev_values)
        self._completed_keys.add(chunk.key)

        if self._scope_key_count(scope) != self._expected_key_count:
            return []
        if len(builders) != self._source_count:
            raise ValueError(
                f"Expected {self._source_count} builders, but got {len(builders)}."
            )

        outputs = self._flush_scope(scope, builders)
        self._clear_scope(scope)
        self._completed_keys = {
            key for key in self._completed_keys if self._scope(key) != scope
        }
        return outputs

    @property
    @abstractmethod
    def _expected_key_count(self) -> int:
        """Number of chunks required to flush one collection scope."""

    @abstractmethod
    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        """Return the collection scope containing a trajectory key."""

    @abstractmethod
    def _builders_for(
        self, scope: CollectionScope
    ) -> dict[SourceID, EmbodiedTrajectoryBuilder]:
        """Return the source builders for a collection scope."""

    @abstractmethod
    def _flush_scope(
        self,
        scope: CollectionScope,
        builders: dict[SourceID, EmbodiedTrajectoryBuilder],
    ) -> list[CollectorOutput]:
        """Convert a completed scope into consumer outputs."""

    @abstractmethod
    def _clear_scope(self, scope: CollectionScope) -> None:
        """Release a successfully converted scope."""

    def _scope_key_count(self, scope: CollectionScope) -> int:
        return sum(self._scope(key) == scope for key in self._completed_keys)

    def _append_chunk(
        self, builder: EmbodiedTrajectoryBuilder, chunk: AssembledChunk
    ) -> None:
        policy = chunk.policy
        assert isinstance(policy, PolicyStep)
        result = policy.rollout_result
        env_result = chunk.env.result
        result.bootstrap_values = chunk.env.bootstrap_values

        if chunk.initial_env_result is not None:
            builder.append_initial_state(
                dones=chunk.initial_env_result.dones,
                truncations=chunk.initial_env_result.truncations,
                terminations=chunk.initial_env_result.terminations,
            )
        builder.append_step_result(
            ChunkStepResult(
                actions=result.forward_inputs.get("action"),
                prev_logprobs=(
                    result.prev_logprobs if self._collect_prev_infos else None
                ),
                prev_values=(result.prev_values if self._collect_prev_infos else None),
                forward_inputs=result.forward_inputs,
                versions=result.versions,
                dones=env_result.dones,
                truncations=env_result.truncations,
                terminations=env_result.terminations,
                rewards=self._rewards(chunk),
            )
        )
        if env_result.reward_assign_lengths is not None:
            rewards = builder.rewards
            reward = self._reward_weight * env_result.reward_model_output
            for env_id, length in enumerate(env_result.reward_assign_lengths):
                for offset in range(2, min(length, len(rewards)) + 1):
                    rewards[-offset][env_id] += reward[env_id].to(
                        rewards[-offset].dtype
                    )
        if env_result.intervene_actions is not None:
            builder.update_last_actions(
                env_result.intervene_actions, env_result.intervene_flags
            )
        if result.intervene_flags is not None:
            builder.mark_last_step_with_intervene_flags(result.intervene_flags)
        if self._enable_rlt:
            from rlinf.algorithms.rlt.transition import (
                apply_rlt_interventions,
                extract_rlt_obs_from_forward_inputs,
            )

            current_obs = extract_rlt_obs_from_forward_inputs(result.forward_inputs)
            apply_rlt_interventions(
                current_obs,
                env_result.intervene_actions,
                env_result.intervene_flags,
            )
            builder.append_transitions(
                current_obs,
                extract_rlt_obs_from_forward_inputs(
                    chunk.env.forward_inputs, transition=True
                ),
            )
        elif self._collect_transitions:
            builder.append_transitions(policy.obs, chunk.env.next_obs)

    def _rewards(self, chunk: AssembledChunk) -> torch.Tensor | None:
        env_result = chunk.env.result
        rewards = env_result.rewards
        if rewards is None:
            return None
        if env_result.reward_model_output is not None:
            rewards = (
                self._env_reward_weight * rewards
                + self._reward_weight * env_result.reward_model_output.to(rewards.dtype)
            )
        values = chunk.env.bootstrap_values
        if (
            values is None
            or not self._cfg.env.train.auto_reset
            or env_result.dones is None
        ):
            return rewards
        truncations = env_result.truncations
        if self._cfg.algorithm.get("bootstrap_type", "standard") != "standard":
            truncations = env_result.dones
        if truncations is None or not truncations[:, -1].any():
            return rewards
        rewards = rewards.clone()
        mask = truncations[:, -1]
        rewards[mask, -1] += self._cfg.algorithm.get("gamma", 1.0) * values[
            mask
        ].reshape(-1).to(rewards.dtype)
        return rewards


class RolloutTrajectoryCollector(_TrajectoryBuilderCollector):
    """Collect complete rollout steps and emit trajectory shards."""

    def __init__(
        self,
        cfg: DictConfig,
        source_count: int,
        chunk_count: int,
        shards_per_source: int,
    ):
        """Initialize step-scoped trajectory builders."""
        super().__init__(cfg, source_count, chunk_count)
        self._shards_per_source = shards_per_source
        self._builders: dict[int, dict[SourceID, EmbodiedTrajectoryBuilder]] = {}

    @property
    def _expected_key_count(self) -> int:
        return (
            self._source_count * self._cfg.env.train.rollout_epoch * self._chunk_count
        )

    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        return key.step_id

    def _builders_for(
        self, scope: CollectionScope
    ) -> dict[SourceID, EmbodiedTrajectoryBuilder]:
        assert isinstance(scope, int)
        return self._builders.setdefault(scope, {})

    def _flush_scope(
        self,
        scope: CollectionScope,
        builders: dict[SourceID, EmbodiedTrajectoryBuilder],
    ) -> list[CollectorOutput]:
        return [
            CollectorOutput(queue_key="default", data=trajectory)
            for builder in builders.values()
            for trajectory in builder.to_splited_trajectories(self._shards_per_source)
        ]

    def _clear_scope(self, scope: CollectionScope) -> None:
        assert isinstance(scope, int)
        del self._builders[scope]


class PipelineTrajectoryCollector(_TrajectoryBuilderCollector):
    """Collect one rollout epoch and emit actor-routed training micro-batches."""

    def __init__(
        self,
        cfg: DictConfig,
        source_count: int,
        chunk_count: int,
        actor_world_size: int,
    ):
        """Initialize epoch-scoped builders and actor routing state."""
        super().__init__(cfg, source_count, chunk_count)
        self._actor_world_size = actor_world_size
        self._builders: dict[
            tuple[int, int], dict[SourceID, EmbodiedTrajectoryBuilder]
        ] = {}
        self._generators: dict[int, torch.Generator] = {}
        self._shuffle_rollout = cfg.algorithm.get("shuffle_rollout", True)

    @property
    def _expected_key_count(self) -> int:
        return self._source_count * self._chunk_count

    def _scope(self, key: TrajectoryKey) -> CollectionScope:
        return (key.step_id, key.epoch_id)

    def _builders_for(
        self, scope: CollectionScope
    ) -> dict[SourceID, EmbodiedTrajectoryBuilder]:
        assert isinstance(scope, tuple)
        return self._builders.setdefault(scope, {})

    def _flush_scope(
        self,
        scope: CollectionScope,
        builders: dict[SourceID, EmbodiedTrajectoryBuilder],
    ) -> list[CollectorOutput]:
        batches_by_actor: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
        for (rollout_rank, stage_id), builder in builders.items():
            logical_rank = (
                rollout_rank * self._cfg.rollout.pipeline_stage_num + stage_id
            )
            actor_splits = CommMapper.get_dst_ranks(
                batch_size=self._cfg.env.train.total_num_envs,
                src_world_size=self._source_count,
                dst_world_size=self._actor_world_size,
                src_rank=logical_rank,
            )
            trajectories = builder.to_splited_trajectories_by_sizes(
                [size for _, size in actor_splits]
            )
            for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                batches_by_actor[actor_rank].append(
                    self._prepare_pipeline_batch(trajectory)
                )

        if self._cfg.algorithm.get("normalize_advantages", True):
            for batches in batches_by_actor.values():
                stats = sum(
                    masked_stats(batch["advantages"], batch.get("loss_mask"))
                    for batch in batches
                )
                for batch in batches:
                    batch["advantages"] = normalize_from_stats(
                        batch["advantages"], stats
                    )

        return [
            CollectorOutput(queue_key=f"actor:{actor_rank}", data=micro_batch)
            for actor_rank, batches in batches_by_actor.items()
            for batch in batches
            for micro_batch in self._pipeline_micro_batches(batch, actor_rank)
        ]

    def _clear_scope(self, scope: CollectionScope) -> None:
        assert isinstance(scope, tuple)
        del self._builders[scope]

    def _prepare_pipeline_batch(self, trajectory) -> dict[str, torch.Tensor]:
        batch = preprocess_embodied_batch(
            convert_trajectories_to_batch([trajectory]),
            rollout_epoch=1,
            auto_reset=self._cfg.env.train.auto_reset,
            ignore_terminations=self._cfg.env.train.ignore_terminations,
            reward_type=self._cfg.algorithm.reward_type,
            filter_rewards=self._cfg.algorithm.get("filter_rewards", False),
            group_size=self._cfg.algorithm.group_size,
            rewards_lower_bound=self._cfg.algorithm.get("rewards_lower_bound"),
            rewards_upper_bound=self._cfg.algorithm.get("rewards_upper_bound"),
        )
        batch.update(
            calculate_adv_and_returns(
                task_type=self._cfg.runner.task_type,
                adv_type=self._cfg.algorithm.adv_type,
                rewards=batch["rewards"],
                dones=batch["dones"],
                values=batch.get("prev_values"),
                prev_logprobs=batch.get("prev_logprobs"),
                num_action_chunks=self._cfg.actor.model.num_action_chunks,
                gamma=self._cfg.algorithm.get("gamma", 1),
                gae_lambda=self._cfg.algorithm.get("gae_lambda", 1),
                group_size=self._cfg.algorithm.get("group_size", 8),
                reward_type=self._cfg.algorithm.reward_type,
                loss_mask=batch.get("loss_mask"),
                loss_mask_sum=batch.get("loss_mask_sum"),
                normalize_advantages=False,
            )
        )
        return batch

    def _pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        generator = self._generators.setdefault(
            actor_rank,
            torch.Generator().manual_seed(self._cfg.actor.seed + actor_rank),
        )
        indices = (
            torch.randperm(batch_size, generator=generator)
            if self._shuffle_rollout
            else torch.arange(batch_size)
        )
        flat_batch = flatten_embodied_batch(batch, indices)
        micro_batch_size = self._cfg.actor.micro_batch_size
        if batch_size % micro_batch_size:
            raise ValueError(
                f"Pipeline batch size {batch_size} is not divisible by "
                f"micro batch size {micro_batch_size}."
            )
        return [
            pack_batch(micro_batch)
            for micro_batch in split_dict_to_chunk(
                flat_batch, batch_size // micro_batch_size, dim=0
            )
        ]


class OnlineLerobotTrajectoryCollector(TrajectoryCollector):
    """Collect online LeRobot episodes and emit episode shards per rollout step."""

    def __init__(
        self,
        cfg: DictConfig,
        source_count: int,
        chunk_count: int,
        shards_per_source: int,
    ):
        """Initialize persistent per-source LeRobot episode builders."""
        self._cfg = cfg
        self._source_count = source_count
        self._chunk_count = chunk_count
        self._shards_per_source = shards_per_source
        self._completed_keys: set[TrajectoryKey] = set()
        self._builders: dict[SourceID, EmbodiedLerobotTrajectoryBuilder] = {}
        online_cfg = cfg.algorithm.dagger.online_lerobot
        self._only_success = bool(online_cfg.get("only_success", False))

    def collect(self, chunk: AssembledChunk) -> list[CollectorOutput]:
        """Append episode data and drain completed episodes at step boundaries."""
        if chunk.key in self._completed_keys:
            raise ValueError(f"Received duplicate trajectory event for {chunk.key}.")
        episode_data = chunk.env.result.episode_data
        if episode_data is None:
            raise ValueError("Online LeRobot segment is missing episode data.")

        builder = self._builders.setdefault(
            chunk.source,
            EmbodiedLerobotTrajectoryBuilder(
                max_episode_length=self._cfg.env.train.max_episode_steps,
                num_envs=self._cfg.env.train.total_num_envs // self._source_count,
                only_success=self._only_success,
                num_action_chunks=self._cfg.actor.model.num_action_chunks,
                action_dim=self._cfg.actor.model.action_dim,
            ),
        )
        policy_output = (
            chunk.policy.rollout_result
            if isinstance(chunk.policy, PolicyStep)
            else None
        )
        builder.append_chunk_episode_data(policy_output=policy_output, **episode_data)
        self._completed_keys.add(chunk.key)

        expected = (
            self._source_count * self._cfg.env.train.rollout_epoch * self._chunk_count
        )
        if (
            sum(key.step_id == chunk.key.step_id for key in self._completed_keys)
            != expected
        ):
            return []
        if len(self._builders) != self._source_count:
            raise ValueError(
                f"Expected {self._source_count} builders, but got {len(self._builders)}."
            )

        outputs = []
        for source_builder in self._builders.values():
            episodes = source_builder.drain_episodes()
            outputs.extend(
                CollectorOutput(
                    queue_key="default",
                    data=episodes[shard :: self._shards_per_source],
                )
                for shard in range(self._shards_per_source)
            )
        self._completed_keys = {
            key for key in self._completed_keys if key.step_id != chunk.key.step_id
        }
        return outputs


def create_trajectory_collector(
    cfg: DictConfig,
    *,
    source_count: int,
    chunk_count: int,
    shards_per_source: int,
    actor_world_size: int,
) -> TrajectoryCollector:
    """Create the collection strategy selected by the training configuration."""
    use_training_pipeline = cfg.runner.get("use_training_pipeline", False)
    online_lerobot = bool(
        cfg.algorithm.get("dagger", {}).get("online_lerobot", {}).get("enabled", False)
    )
    if use_training_pipeline and online_lerobot:
        raise ValueError(
            "Training pipeline does not support online LeRobot trajectory data."
        )
    if use_training_pipeline and cfg.runner.get("enable_decoupled_mode", False):
        raise ValueError(
            "Training pipeline does not support decoupled environment mode."
        )
    if online_lerobot:
        return OnlineLerobotTrajectoryCollector(
            cfg, source_count, chunk_count, shards_per_source
        )
    if use_training_pipeline:
        return PipelineTrajectoryCollector(
            cfg, source_count, chunk_count, actor_world_size
        )
    return RolloutTrajectoryCollector(cfg, source_count, chunk_count, shards_per_source)
