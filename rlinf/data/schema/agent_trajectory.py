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

"""Complete agent trajectories, buffered batches, and fixed token targets."""

import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from rlinf.algorithms.registry import get_adv_and_returns
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel


def retained_size_bytes(value: Any) -> int:
    """Estimate Python heap plus unique tensor storage, without serializing a copy.

    This is an admission signal, not a process RSS or a hard memory reservation.
    File-backed images count only their references; their bounded pixel cache has
    a separate budget. Shared objects and tensor views are counted once.
    """
    seen, storages = set(), set()

    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        total = sys.getsizeof(item)
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key not in storages:
                storages.add(key)
                total += storage.nbytes()
        elif isinstance(item, dict):
            total += sum(size(key) + size(val) for key, val in item.items())
        elif isinstance(item, (list, tuple, set, deque)):
            total += sum(size(val) for val in item)
        elif hasattr(item, "__dict__"):
            total += size(vars(item))
        return total

    return size(value)


def _action_positions(result: DynamicRolloutResult, row: int) -> torch.Tensor:
    """Sequence positions of the tokens the policy generated in one row."""
    start = result.prompt_lengths[row]
    if result.response_mask is None:
        return torch.arange(start, start + result.response_lengths[row])
    mask = torch.as_tensor(result.response_mask[row], dtype=torch.bool)
    return start + mask.nonzero(as_tuple=True)[0]


@dataclass
class BufferedTrajectory:
    """A completed rollout and its prompt, retained for stale-data retries."""

    sample_id: int
    prompt_ids: list[int]
    answer: Any
    result: DynamicRolloutResult
    metrics: dict[str, Any]

    @property
    def oldest_version(self) -> int:
        """Return the oldest weight version that generated an action token."""
        return self._oldest_version()

    def _oldest_version(self, published_version: int | None = None) -> int:
        versions = self.result.rollout_weight_versions
        if versions is None or len(versions) != self.result.num_sequence:
            raise ValueError(
                "Async trajectories require generated-token weight versions."
            )
        for length, turn in zip(self.result.response_lengths, versions, strict=True):
            if turn is None or len(turn) != length or not turn or min(turn) < 0:
                raise ValueError(
                    "Each generated token must have a nonnegative weight version."
                )
            if published_version is not None and max(turn) > published_version:
                raise ValueError(
                    "A rollout version cannot be newer than the published version."
                )
        return min(min(turn) for turn in versions)


@torch.no_grad()
def prepare_agent_critic_returns(
    results: list[DynamicRolloutResult], seq_length: int, gamma: float
) -> None:
    """Attach fixed action-only returns for complete, terminally rewarded episodes."""
    for result in results:
        result.returns = torch.zeros(
            result.num_sequence, seq_length, dtype=torch.float32
        )
        for trajectory in range(result.group_size):
            rows = [
                i for i, index in enumerate(result.idx_to_traj) if index == trajectory
            ]
            positions = [_action_positions(result, i) for i in rows]
            if not rows or any(len(p) == 0 for p in positions):
                raise ValueError(
                    "Each agent episode must contain nonempty generated turns."
                )
            remaining = sum(len(p) for p in positions)
            reward = float(result.rewards[rows[-1]])
            for row, actions in zip(rows, positions, strict=True):
                length = len(actions)
                discounts = gamma ** torch.arange(
                    remaining - 1, remaining - length - 1, -1, dtype=torch.float32
                )
                result.returns[row, actions] = reward * discounts
                remaining -= length


def construct_rollout_batch(
    input_channel: Channel,
    output_channel: Channel,
    retry_channel: Channel,
    batch_size: int,
    seq_length: int,
    gamma: float,
    published_version: int,
    max_weight_staleness: int | None = None,
    release_samples: Callable[[list[int]], None] | None = None,
) -> dict[str, Any]:
    """Consume completed trajectories, retry stale ones, and send critic targets."""
    entries: list[BufferedTrajectory] = []
    oldest_version = published_version
    stale_count = 0
    while len(entries) < batch_size:
        entry: BufferedTrajectory = input_channel.get()
        version = entry._oldest_version(published_version)
        lag = published_version - version
        if max_weight_staleness is not None and lag > max_weight_staleness:
            # Release the old admission charge before the same ID can be retried.
            if release_samples is not None:
                release_samples([entry.sample_id])
            retry_channel.put((entry.sample_id, entry.prompt_ids, entry.answer))
            stale_count += 1
            continue
        entries.append(entry)
        oldest_version = min(oldest_version, version)

    results = [entry.result for entry in entries]
    prepare_agent_critic_returns(results, seq_length, gamma)
    for result in results:
        output_channel.put(result)
    return {
        "sample_ids": [entry.sample_id for entry in entries],
        "oldest_version": oldest_version,
        "stale_dropped": stale_count,
        "agent_metrics": [entry.metrics for entry in entries],
    }


@torch.no_grad()
def prepare_agent_ppo_targets(
    input_channel: Channel,
    output_channel: Channel | list[Channel],
    batch_size: int,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    logprob_channel: Channel | None = None,
    adv_type: str = "actor_gae",
    gae_lambda_alpha: float = 1.5,
    expected_weight_version: int | None = None,
) -> None:
    """Read a complete batch, compute global GAE, and send fixed training targets.

    SearchR1 settles a final reward at both natural and budget episode endings;
    neither has a future value. Tool observations and padding do not advance GAE.
    A separate logprob channel joins predictions from parallel actor inference.
    Rows with a ``response_mask`` contribute only their generated tokens.
    """
    results = [input_channel.get() for _ in range(batch_size)]
    if expected_weight_version is not None and any(
        result.version_start is not None for result in results
    ):
        for result in results:
            if (
                result.version_start is None
                or result.version_end is None
                or any(
                    version != expected_weight_version
                    for version in result.version_start + result.version_end
                )
            ):
                raise ValueError(
                    f"Synchronous batch must contain only installed policy version {expected_weight_version}."
                )
    if logprob_channel is not None:
        predictions = {}
        for _ in range(batch_size):
            prediction = logprob_channel.get()
            if prediction.rollout_ids is None:
                raise ValueError("Parallel actor predictions require rollout IDs.")
            key = tuple(prediction.rollout_ids)
            if key in predictions:
                raise ValueError(f"Duplicate actor prediction for rollout IDs {key}.")
            predictions[key] = prediction
        for result in results:
            key = tuple(result.rollout_ids or ())
            prediction = predictions.pop(key, None)
            if prediction is None or prediction.input_ids != result.input_ids:
                raise ValueError(
                    "Expected actor and critic predictions for the same agent turns."
                )
            result.recomputed_logprobs = prediction.recomputed_logprobs

    trajectories = []
    for result in results:
        if result.values is None or result.rewards is None:
            raise ValueError("Agent PPO requires fixed old values and episode rewards.")
        result.advantages = torch.zeros_like(result.values)
        result.returns = torch.zeros_like(result.values)
        for trajectory in range(result.group_size):
            rows = [
                i for i, index in enumerate(result.idx_to_traj) if index == trajectory
            ]
            positions = [_action_positions(result, i) for i in rows]
            if not rows or any(len(p) == 0 for p in positions):
                raise ValueError(
                    "Each agent episode must contain nonempty generated turns."
                )
            values = torch.cat(
                [result.values[i, p] for i, p in zip(rows, positions)]
            ).float()  # [action tokens of the trajectory]
            trajectories.append(
                (
                    result,
                    list(zip(rows, positions)),
                    values,
                    float(result.rewards[rows[-1]]),
                )
            )

    if not trajectories:
        raise ValueError("Agent PPO requires at least one complete episode.")
    # max(length) = T, M = total traj(2048)
    lengths = [len(item[2]) for item in trajectories]
    values = torch.zeros(max(lengths), len(trajectories), dtype=torch.float32)  # [T, M]
    rewards = torch.zeros_like(values)  # [T, M]
    mask = torch.zeros_like(
        values, dtype=torch.bool
    )  # [T, M] 1 for valid steps, 0 for padding
    for index, (_, _, old_values, reward) in enumerate(trajectories):
        length = lengths[index]
        values[:length, index] = old_values
        mask[:length, index] = True
        rewards[length - 1, index] = reward

    advantages, returns = get_adv_and_returns(adv_type)(
        rewards=rewards,
        values=values,
        gamma=gamma,
        gae_lambda=gae_lambda,
        gae_lambda_alpha=gae_lambda_alpha,
        step_mask=mask,
        normalize_advantages=False,
    )
    if normalize_advantages:
        # under global batch
        valid = advantages[mask].double()
        mean, variance = valid.mean(), valid.var(unbiased=False)
        advantages = torch.where(
            mask,
            (advantages - mean.float()) / variance.sqrt().clamp_min(1e-8).float(),
            0.0,
        )

    for index, (result, rows, _, _) in enumerate(trajectories):
        offset = 0
        for row, positions in rows:
            length = len(positions)
            result.advantages[row, positions] = advantages[
                offset : offset + length, index
            ]
            result.returns[row, positions] = returns[offset : offset + length, index]
            offset += length

    channels = output_channel if isinstance(output_channel, list) else [output_channel]
    for result in results:
        for channel in channels:
            channel.put(result)
