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

"""Join trajectory channel events into complete action chunks."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias

from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    TrajectoryKey,
    TrajectorySource,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    DummyPolicyStep,
    EnvStepResult,
    PolicyStep,
    TrajectoryData,
    TrajectoryStart,
)

PolicyEvent: TypeAlias = PolicyStep | DummyPolicyStep


@dataclass(kw_only=True)
class AssembledChunk:
    """Policy and environment data joined for one action chunk."""

    key: TrajectoryKey
    source: tuple[int, int]
    policy: PolicyEvent
    env: EnvStepResult
    initial_env_result: EnvResult | None = None


class TrajectoryEventAssembler:
    """Reassemble source fragments and join corresponding trajectory events."""

    def __init__(self, source_batch_size: int):
        """Initialize assembly state for one logical rollout source."""
        self._source_batch_size = source_batch_size
        self._policy_steps: dict[TrajectoryKey, PolicyEvent] = {}
        self._env_results: dict[TrajectoryKey, EnvStepResult] = {}
        self._initial_results: dict[tuple[int, int, int, int], EnvResult] = {}
        self._fragments: dict[tuple[type, TrajectoryKey], list[object]] = defaultdict(
            list
        )

    def push(self, event: TrajectoryData) -> list[AssembledChunk]:
        """Consume one channel event and return any newly completed chunks."""
        if isinstance(event, TrajectoryStart):
            key = event.source.key
            self._initial_results[self._initial_key(key)] = event.result
            chunk = self._try_complete(key)
            return [chunk] if chunk is not None else []
        if not isinstance(event, (PolicyStep, DummyPolicyStep, EnvStepResult)):
            raise ValueError(f"Unexpected data type: {type(event)}")

        if self._is_complete_source(event):
            chunk = self._accept_complete_event(event)
            return [chunk] if chunk is not None else []

        completed = []
        for fragment in self._split_event(event):
            merged = self._merge_fragments(fragment)
            if merged is None:
                continue
            chunk = self._accept_complete_event(merged)
            if chunk is not None:
                completed.append(chunk)
        return completed

    def _is_complete_source(self, event: PolicyEvent | EnvStepResult) -> bool:
        """Return whether an event already contains one complete source batch."""
        if len(event.sources) != 1:
            return False
        source = event.sources[0]
        return source.offset == 0 and source.size == self._source_batch_size

    def _accept_complete_event(
        self, event: PolicyEvent | EnvStepResult
    ) -> AssembledChunk | None:
        """Store one complete event and join it with its counterpart if present."""
        key = event.sources[0].key
        if isinstance(event, (PolicyStep, DummyPolicyStep)):
            self._policy_steps[key] = event
        else:
            self._env_results[key] = event
        return self._try_complete(key)

    def acknowledge(self, key: TrajectoryKey) -> None:
        """Release a chunk after a collector has accepted it."""
        del self._policy_steps[key]
        del self._env_results[key]
        if key.chunk_id == 0:
            del self._initial_results[self._initial_key(key)]

    @staticmethod
    def _initial_key(key: TrajectoryKey) -> tuple[int, int, int, int]:
        return (key.step_id, key.epoch_id, key.env_rank, key.stage_id)

    def _try_complete(self, key: TrajectoryKey) -> AssembledChunk | None:
        policy = self._policy_steps.get(key)
        env = self._env_results.get(key)
        if policy is None or env is None:
            return None

        initial_key = self._initial_key(key)
        if key.chunk_id == 0 and initial_key not in self._initial_results:
            return None

        initial_result = (
            self._initial_results[initial_key] if key.chunk_id == 0 else None
        )
        return AssembledChunk(
            key=key,
            source=(key.env_rank, key.stage_id),
            policy=policy,
            env=env,
            initial_env_result=initial_result,
        )

    def _split_event(self, event):
        sizes = [source.size for source in event.sources]
        if isinstance(event, PolicyStep):
            observations = split_batch_value(event.obs, sizes)
            fields = {
                name: split_batch_value(getattr(event.rollout_result, name), sizes)
                for name in event.rollout_result.__dataclass_fields__
            }
            for index, source in enumerate(event.sources):
                yield PolicyStep(
                    sources=[source],
                    obs=observations[index],
                    rollout_result=EmbodiedRolloutResult(
                        **{name: values[index] for name, values in fields.items()}
                    ),
                )
            return
        if isinstance(event, DummyPolicyStep):
            observations = split_batch_value(event.obs, sizes)
            actions = split_batch_value(event.actions, sizes)
            for index, source in enumerate(event.sources):
                yield DummyPolicyStep(
                    sources=[source],
                    obs=observations[index],
                    actions=actions[index],
                )
            return

        fields = {
            name: split_batch_value(getattr(event.result, name), sizes)
            for name in event.result.__dataclass_fields__
            if name != "episode_data"
        }
        episodes = split_episode_data(event.result.episode_data, sizes)
        next_observations = split_batch_value(event.next_obs, sizes)
        forward_inputs = split_batch_value(event.forward_inputs, sizes)
        bootstrap_values = split_batch_value(event.bootstrap_values, sizes)
        final_prev_values = split_batch_value(event.final_prev_values, sizes)
        for index, source in enumerate(event.sources):
            yield EnvStepResult(
                sources=[source],
                result=EnvResult(
                    **{name: values[index] for name, values in fields.items()},
                    episode_data=episodes[index],
                ),
                next_obs=next_observations[index],
                forward_inputs=forward_inputs[index],
                bootstrap_values=bootstrap_values[index],
                final_prev_values=final_prev_values[index],
            )

    def _merge_fragments(self, event):
        source = event.sources[0]
        key = (type(event), source.key)
        fragments = self._fragments[key]
        fragments.append(event)
        received_size = sum(item.sources[0].size for item in fragments)
        if received_size < self._source_batch_size:
            return None
        if received_size > self._source_batch_size:
            raise ValueError(
                f"Trajectory fragments exceed source batch size for {source.key}."
            )

        fragments.sort(key=lambda item: item.sources[0].offset)
        offsets = [item.sources[0].offset for item in fragments]
        sizes = [item.sources[0].size for item in fragments]
        if any(offset != sum(sizes[:index]) for index, offset in enumerate(offsets)):
            raise ValueError(f"Non-contiguous trajectory fragments: {offsets}.")

        del self._fragments[key]
        full_source = TrajectorySource(source.key, self._source_batch_size)
        if isinstance(event, PolicyStep):
            return PolicyStep(
                sources=[full_source],
                obs=merge_batch_values([item.obs for item in fragments]),
                rollout_result=EmbodiedRolloutResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item.rollout_result, name) for item in fragments]
                        )
                        for name in event.rollout_result.__dataclass_fields__
                    }
                ),
            )
        if isinstance(event, DummyPolicyStep):
            return DummyPolicyStep(
                sources=[full_source],
                obs=merge_batch_values([item.obs for item in fragments]),
                actions=merge_batch_values([item.actions for item in fragments]),
            )
        if isinstance(event, EnvStepResult):
            return EnvStepResult(
                sources=[full_source],
                result=EnvResult(
                    **{
                        name: merge_batch_values(
                            [getattr(item.result, name) for item in fragments]
                        )
                        for name in event.result.__dataclass_fields__
                        if name != "episode_data"
                    },
                    episode_data=(
                        merge_episode_data(
                            [item.result.episode_data for item in fragments]
                        )
                        if event.result.episode_data is not None
                        else None
                    ),
                ),
                next_obs=merge_batch_values([item.next_obs for item in fragments]),
                forward_inputs=merge_batch_values(
                    [item.forward_inputs for item in fragments]
                ),
                bootstrap_values=merge_batch_values(
                    [item.bootstrap_values for item in fragments]
                ),
                final_prev_values=merge_batch_values(
                    [item.final_prev_values for item in fragments]
                ),
            )
        raise TypeError(f"Unexpected trajectory event: {type(event)}")
