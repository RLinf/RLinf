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

"""Fixed token targets for complete multi-turn agent episodes."""

from collections import defaultdict, deque

import torch

from rlinf.algorithms.registry import get_adv_and_returns
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel


def _action_positions(result: DynamicRolloutResult, row: int) -> torch.Tensor:
    """Sequence positions of the tokens the policy generated in one row."""
    start = result.prompt_lengths[row]
    if result.response_mask is None:
        return torch.arange(start, start + result.response_lengths[row])
    mask = torch.as_tensor(result.response_mask[row], dtype=torch.bool)
    return start + mask.nonzero(as_tuple=True)[0]


@torch.no_grad()
def prepare_agent_ppo_targets(
    input_channel: Channel,
    output_channel: Channel | list[Channel],
    batch_size: int,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    logprob_channel: Channel | None = None,
) -> None:
    """Read a complete batch, compute global GAE, and send fixed training targets.

    SearchR1 settles a final reward at both natural and budget episode endings;
    neither has a future value. Tool observations and padding do not advance GAE.
    A separate logprob channel joins predictions from parallel actor inference.
    Rows with a ``response_mask`` contribute only their generated tokens.
    """
    results = [input_channel.get() for _ in range(batch_size)]
    if logprob_channel is not None:
        logprobs = defaultdict(deque)
        for _ in range(batch_size):
            prediction = logprob_channel.get()
            key = tuple(map(tuple, prediction.input_ids))
            logprobs[key].append(prediction.recomputed_logprobs)
        for result in results:
            key = tuple(map(tuple, result.input_ids))
            if not logprobs[key]:
                raise ValueError(
                    "Expected actor and critic predictions for the same agent turns."
                )
            result.recomputed_logprobs = logprobs[key].popleft()

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

    advantages, returns = get_adv_and_returns("actor_gae")(
        rewards=rewards,
        values=values,
        gamma=gamma,
        gae_lambda=gae_lambda,
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
