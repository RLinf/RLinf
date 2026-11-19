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
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.algorithms.distillation.config import build_distillation_plan
from rlinf.data.schema.distillation import (
    SampleContext,
    TeacherScoreRequest,
    TeacherTargets,
)

from .backend import tokenizer_fingerprint
from .client import TeacherClient


def rollout_topk_count(cfg: DictConfig) -> int:
    """Return the number of model-output candidates collected during rollout."""
    plan = build_distillation_plan(cfg)
    if not plan.need_rollout_topk:
        return 0
    return plan.topk


def select_rollout_topk(
    rows: list | None, response_length: int, topk: int
) -> tuple[tuple[int, ...], ...]:
    """Keep K unique model-output IDs per position from engine logprob rows."""
    if rows is None or len(rows) != response_length:
        raise ValueError("Student rollout top-k rows must match the response length")
    selected = []
    for row in rows:
        values = (
            [(int(i), float(value.logprob)) for i, value in row.items()]
            if isinstance(row, dict)
            else [(int(item[1]), float(item[0])) for item in row]
        )
        values = sorted(values, key=lambda item: (-item[1], item[0]))
        ids = tuple(i for i, _ in values[:topk])
        if len(ids) != topk or len(set(ids)) != topk or any(i < 0 for i in ids):
            raise ValueError(
                f"Student rollout must return {topk} unique model-output token IDs per position"
            )
        selected.append(ids)
    return tuple(selected)


class RolloutTeacherScorer:
    """One connection owner per rollout or Agent worker."""

    def __init__(self, cfg, endpoints, tokenizer):
        self.plan = build_distillation_plan(cfg)
        self.client = TeacherClient(endpoints)
        self.fingerprint = tokenizer_fingerprint(tokenizer)
        self.teacher_key = (
            endpoints[0].teacher_key if len(self.client.endpoints) == 1 else None
        )
        if any(
            self.fingerprint != endpoint.model_info.tokenizer_fingerprint
            for endpoint in endpoints
        ):
            raise ValueError("Student and Teacher tokenizer mappings must be identical")
        if set(self.plan.teacher_keys) != self.client.endpoints.keys():
            raise ValueError("Expected a ready endpoint for every configured Teacher")
        if len(self.plan.teacher_keys) > 1:
            from transformers import AutoConfig

            model = cfg.rollout.model
            student_config = AutoConfig.from_pretrained(
                model.model_path,
                trust_remote_code=model.get("trust_remote_code", False),
            )
            vocab_size = student_config.get_text_config().vocab_size
            if any(
                endpoint.model_info.vocab_size != vocab_size for endpoint in endpoints
            ):
                raise ValueError(
                    f"Teacher output vocabulary sizes must match Student size {vocab_size}"
                )
        self.student_policy_version = "0"

    def request(
        self,
        input_ids,
        response_mask,
        *,
        selected_token_ids=None,
        sample_id=None,
        context: SampleContext | None = None,
        turn_index=0,
    ):
        """Build a position-aligned request without decoding or re-tokenizing."""
        teacher_key = context.teacher_key if context else self.teacher_key
        if teacher_key is None:
            raise ValueError("Multi-Teacher scoring requires a frozen sample context")
        return TeacherScoreRequest(
            request_id=f"{context.sample_id}/{turn_index}" if context else uuid4().hex,
            sample_id=context.sample_id if context else sample_id or uuid4().hex,
            group_id=context.group_id if context else sample_id or "",
            teacher_key=teacher_key,
            input_ids=tuple(input_ids),
            response_mask=tuple(bool(v) for v in response_mask),
            topk=self.plan.topk,
            tokenizer_fingerprint=self.fingerprint,
            selected_token_ids=selected_token_ids,
            student_policy_version=self.student_policy_version
            if self.plan.need_rollout_topk
            else None,
        )

    async def score_agent_output(self, output, context: SampleContext | None = None):
        """Score trainable turns independently using their actual private context."""
        turns = getattr(output, "single_turn_outputs", [output])

        async def score_turn(index, turn):
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
                turn.prompt_ids + turn.response_ids,
                mask,
                context=context,
                turn_index=index,
                selected_token_ids=selected,
            )
            turn.teacher_targets = await self.client.score(request)
            turn.teacher_targets.context = context

        await asyncio.gather(*(score_turn(i, turn) for i, turn in enumerate(turns)))
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
        *,
        context: SampleContext | None = None,
    ) -> TeacherTargets:
        """Score a completed reasoning sequence."""
        request = self.request(
            prompt_ids + response_ids,
            [False] * len(prompt_ids) + [True] * len(response_ids),
            sample_id=sample_id,
            context=context,
            selected_token_ids=self.align_selected_ids(
                len(prompt_ids), len(response_ids), selected_ids
            )
            if self.plan.need_rollout_topk
            else None,
        )
        target = await self.client.score(request)
        target.context = context
        return target

    async def close(self):
        """Release local client connections."""
        await self.client.close()
