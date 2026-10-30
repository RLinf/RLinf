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

"""One RPent episode run by an agent worker, and the turns its planner sampled."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from rlinf.agents.rpent.history import AppendOnlyHistory
from rlinf.workers.agent.agent_loop import AgentLoopOutput


class RPentEpisodeError(RuntimeError):
    """An episode has no task outcome because its process or a service failed.

    The agent loop retries such an episode with a fresh process.
    """

    def __init__(self, message: str, *, kind: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Episode:
    """One RPent episode and the turns its planner sampled from the policy."""

    episode_id: str
    task: dict
    history: AppendOnlyHistory
    started: float = field(default_factory=time.monotonic)
    turns: list[AgentLoopOutput] = field(default_factory=list)


def merge_prefix_turns(
    turns: list[AgentLoopOutput], max_scored_tokens: int
) -> list[AgentLoopOutput]:
    """Merge consecutive turns whose prompt extends the previous turn's tokens.

    A merged sample keeps the earlier turns' generated tokens and turns the
    tokens between turns (tool results, observations) into response tokens
    with mask 0 and logprob 0. A sample ends at a history rewrite, such as
    compaction, or before it would score more than ``max_scored_tokens``
    generated tokens, which bounds the logits one training row computes.
    Rewards and per-turn metadata come from the last merged turn;
    ``extra_fields["turn_ids"]`` lists the merged turn IDs.

    Unlike ``DynamicRolloutResult.pack_traj_batch``, which folds turns only for
    training, merged rows also shorten logprob and value inference. Train them
    with ``actor.pack_traj: false``.
    """
    merged: list[AgentLoopOutput] = []
    for turn in turns:
        mask = list(turn.response_mask or [1] * len(turn.response_ids))
        last = merged[-1] if merged else None
        history = last.prompt_ids + last.response_ids if last else None
        if (
            last is not None
            and turn.prompt_ids[: len(history)] == history
            and sum(last.response_mask) + sum(mask) <= max_scored_tokens
        ):
            gap = len(turn.prompt_ids) - len(history)
            last.response_ids = (turn.prompt_ids + turn.response_ids)[
                len(last.prompt_ids) :
            ]
            last.response_mask += [0] * gap + mask
            if last.response_logprobs is not None:
                last.response_logprobs += [0.0] * gap + list(turn.response_logprobs)
            last.response_text, last.is_end = turn.response_text, turn.is_end
            last.reward_score = turn.reward_score
            last.extra_fields = {
                **turn.extra_fields,
                "turn_ids": last.extra_fields["turn_ids"]
                + [turn.extra_fields.get("turn_id")],
            }
            continue
        merged.append(
            replace(
                turn,
                response_mask=mask,
                response_logprobs=None
                if turn.response_logprobs is None
                else list(turn.response_logprobs),
                extra_fields={
                    **turn.extra_fields,
                    "turn_ids": [turn.extra_fields.get("turn_id")],
                },
            )
        )
    return merged
