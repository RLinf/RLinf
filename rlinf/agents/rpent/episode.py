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

import asyncio
import json
import time
from dataclasses import dataclass, field, replace

from rlinf.agents.rpent.history import AppendOnlyHistory
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput


class RPentEpisodeError(RuntimeError):
    """An episode has no task outcome because its process or a service failed.

    The agent loop retries such an episode with a fresh process.
    """

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind

    def __reduce__(self):
        """Preserve the failure category across Ray's exception transport."""
        return type(self), (str(self), self.kind)


class GroupStartup:
    """Release policy requests only after every member starts its simulator."""

    def __init__(self, members: int):
        self.remaining = members
        self.admitted = False
        self.ready = asyncio.Event()

    def mark_ready(self) -> None:
        if not self.ready.is_set():
            self.remaining -= 1
            if self.remaining == 0:
                self.admitted = True
                self.ready.set()

    async def wait(self) -> None:
        await self.ready.wait()
        if not self.admitted:
            raise RPentEpisodeError("RPent group startup was cancelled", "env_setup")

    def abort(self) -> None:
        self.ready.set()


@dataclass
class Episode:
    """One RPent episode and the turns its planner sampled from the policy."""

    episode_id: str
    task: dict
    history: AppendOnlyHistory
    started: float = field(default_factory=time.monotonic)
    turns: list[AgentLoopOutput] = field(default_factory=list)
    weight_version: int | None = None
    startup: GroupStartup | None = None


def episode_task(answer: str | list[str]) -> dict:
    """Decode the single task carried by one reasoning dataset row."""
    if isinstance(answer, str):
        return json.loads(answer)
    if len(answer) != 1:
        raise ValueError(
            f"an RPent dataset row must specify exactly one task, got {len(answer)}"
        )
    return json.loads(answer[0])


def merge_prefix_turns(
    turns: list[AgentLoopOutput], max_scored_tokens: int
) -> list[AgentLoopOutput]:
    """Merge prefix-compatible turns within one image render chain.

    Tool/observation gaps receive mask 0 and logprob 0. Split at history
    rewrites or ``max_scored_tokens``. Keep the last turn's images, reward
    and metadata, and record all merged IDs in ``extra_fields["turn_ids"]``.
    Use ``actor.pack_traj=false`` because these rows are already merged.
    """
    merged: list[AgentLoopOutput] = []
    for turn in turns:
        mask = list(turn.response_mask or [1] * len(turn.response_ids))
        last = merged[-1] if merged else None
        history = last.prompt_ids + last.response_ids if last else None
        if (
            last is not None
            and turn.extra_fields.get("chain_id") == last.extra_fields.get("chain_id")
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
            if last.response_weight_versions is not None:
                if turn.response_weight_versions is None:
                    last.response_weight_versions = None
                else:
                    # Tool output between turns is not policy-generated; the
                    # preceding version keeps every entry nonnegative, and the
                    # zero mask excludes those tokens from staleness.
                    carried = last.response_weight_versions[-1]
                    last.response_weight_versions += [carried] * gap + list(
                        turn.response_weight_versions
                    )
            # The row now ends at this turn, whose prompt already expands every
            # image of the chain in order, so the row adopts its image set.
            last.response_images = turn.response_images
            last.version_end = turn.version_end
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
                response_weight_versions=None
                if turn.response_weight_versions is None
                else list(turn.response_weight_versions),
                extra_fields={
                    **turn.extra_fields,
                    "turn_ids": [turn.extra_fields.get("turn_id")],
                },
            )
        )
    return merged


def episode_output(
    turns: list[AgentLoopOutput],
    *,
    task: dict,
    episode_id: str,
    status: dict,
    elapsed_s: float,
    merge: bool,
    max_scored_tokens: int,
) -> MultiAgentLoopOutput:
    """Score sampled turns and retain their chronological episode identity."""
    for turn in turns:
        turn.reward_score = float(status["success"])
    output_turns = merge_prefix_turns(turns, max_scored_tokens) if merge else turns
    return MultiAgentLoopOutput(
        single_turn_outputs=output_turns,
        extra_fields={
            "success": status["success"],
            "init_state_id": status["init_state_id"],
            "truncated": status["truncated"],
            "task": task,
            "terminal": status["terminal"],
            "termination_reason": status["termination_reason"],
            "episode_id": episode_id,
            "elapsed_s": elapsed_s,
            "turns": len(turns),
        },
    )
