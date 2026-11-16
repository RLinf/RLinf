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

"""Attach Teacher targets to original reasoning and Agent trajectories."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.algorithms.distillation.config import build_distillation_plan
from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import tokenizer_fingerprint
from .client import TeacherClient

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def rollout_topk_count(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase) -> int:
    """Overfetch model-only IDs so rollout can retain K real tokenizer IDs."""
    plan = build_distillation_plan(cfg)
    if not plan.need_rollout_topk:
        return 0
    from transformers import AutoConfig

    model = cfg.rollout.model
    config = AutoConfig.from_pretrained(
        model.model_path, trust_remote_code=model.get("trust_remote_code", False)
    )
    return plan.topk + max(0, config.get_text_config().vocab_size - len(tokenizer))


def select_rollout_topk(
    rows: list | None, response_length: int, topk: int, vocab_size: int
) -> tuple[tuple[int, ...], ...]:
    """Keep K unique real IDs per response position from engine logprob rows."""
    if rows is None or len(rows) != response_length:
        raise ValueError("Student rollout top-k rows must match the response length")
    selected = []
    for row in rows:
        values = (
            [(int(i), float(value.logprob)) for i, value in row.items()]
            if isinstance(row, dict)
            else [(int(item[1]), float(item[0])) for item in row]
        )
        values = sorted(
            ((i, lp) for i, lp in values if 0 <= i < vocab_size),
            key=lambda item: (-item[1], item[0]),
        )
        ids = tuple(i for i, _ in values[:topk])
        if len(ids) != topk or len(set(ids)) != topk:
            raise ValueError(
                f"Student rollout must return {topk} unique real token IDs per position"
            )
        selected.append(ids)
    return tuple(selected)


class RolloutTeacherScorer:
    """One connection owner per rollout or Agent worker."""

    def __init__(self, cfg, endpoints, tokenizer):
        self.plan = build_distillation_plan(cfg)
        self.client = TeacherClient(endpoints)
        self.fingerprint = tokenizer_fingerprint(tokenizer)
        self.teacher_key = endpoints[0].teacher_key
        if self.fingerprint != endpoints[0].model_info.tokenizer_fingerprint:
            raise ValueError("Student and Teacher tokenizer mappings must be identical")
        self.student_policy_version = "0"

    def request(
        self, input_ids, response_mask, *, selected_token_ids=None, sample_id=None
    ):
        """Build a position-aligned request without decoding or re-tokenizing."""
        return TeacherScoreRequest(
            request_id=uuid4().hex,
            sample_id=sample_id or uuid4().hex,
            group_id=sample_id or "",
            teacher_key=self.teacher_key,
            input_ids=tuple(input_ids),
            response_mask=tuple(bool(v) for v in response_mask),
            topk=self.plan.topk,
            tokenizer_fingerprint=self.fingerprint,
            selected_token_ids=selected_token_ids,
            student_policy_version=self.student_policy_version
            if self.plan.need_rollout_topk
            else None,
        )

    async def score_agent_output(self, output):
        """Score trainable turns independently using their actual private context."""
        turns = getattr(output, "single_turn_outputs", [output])

        async def score_turn(turn):
            if turn.extra_fields.get("not_training", False):
                return
            mask = [False] * len(turn.prompt_ids) + list(
                turn.response_mask
                if turn.response_mask is not None
                else [True] * len(turn.response_ids)
            )
            selected = None
            if self.plan.need_rollout_topk:
                selected = self.align_selected_ids(
                    len(turn.prompt_ids),
                    len(turn.response_ids),
                    turn.response_topk_token_ids,
                )
            request = self.request(
                turn.prompt_ids + turn.response_ids, mask, selected_token_ids=selected
            )
            turn.teacher_targets = await self.client.score(request)

        await asyncio.gather(*(score_turn(turn) for turn in turns))
        return output

    def align_selected_ids(
        self,
        prompt_length: int,
        response_length: int,
        selected_ids: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        """Pad rollout support to the original context's target-token positions."""
        if selected_ids is None or len(selected_ids) != response_length:
            raise ValueError(
                "Student top-k IDs must come from the matching rollout response"
            )
        return ((0,) * self.plan.topk,) * prompt_length + tuple(
            tuple(row) for row in selected_ids
        )

    async def score_sequence(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        sample_id: str,
        selected_ids: Sequence[Sequence[int]] | None = None,
    ) -> TeacherTargets:
        """Score a completed reasoning sequence."""
        request = self.request(
            prompt_ids + response_ids,
            [False] * len(prompt_ids) + [True] * len(response_ids),
            sample_id=sample_id,
            selected_token_ids=self.align_selected_ids(
                len(prompt_ids), len(response_ids), selected_ids
            )
            if self.plan.need_rollout_topk
            else None,
        )
        return await self.client.score(request)

    async def close(self):
        """Release local client connections."""
        await self.client.close()
