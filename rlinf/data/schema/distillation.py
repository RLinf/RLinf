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

"""Teacher scoring contracts aligned to predicted token positions."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class SampleContext:
    """Student sample and frozen Teacher assignment shared by its turns."""

    group_id: str
    sample_id: str
    teacher_key: str
    teacher_index: int


def teacher_targets_to_batch(
    targets: list["TeacherTargets"],
    prompt_lengths: list[int],
    response_lengths: list[int],
    sequence_length: int,
    *,
    response_only: bool,
) -> dict[str, torch.Tensor]:
    """Pad CPU targets using the same sequence spans as the Student batch."""
    if (
        not targets
        or len(targets) != len(prompt_lengths)
        or len(targets) != len(response_lengths)
    ):
        raise ValueError("Expected one Teacher target per Student sequence")
    identity = {
        (
            t.token_ids is None,
            t.target_format,
            t.student_policy_version,
            t.logprobs.shape[-1],
        )
        for t in targets
    }
    if len(identity) != 1:
        raise ValueError(
            "Expected compatible Teacher target formats, widths and Student versions"
        )
    width = targets[0].logprobs.shape[-1]
    shape = (len(targets), sequence_length, width)
    result = {
        "teacher_logprobs": torch.zeros(shape, dtype=torch.float32),
        "teacher_valid_mask": torch.zeros(shape[:2], dtype=torch.bool),
    }
    identities = {}
    teacher_indices = {}
    for target in targets:
        if target.context and target.context.teacher_key != target.teacher_key:
            raise ValueError(
                f"Teacher target for {target.request_id} disagrees with sample route"
            )
        if (
            teacher_indices.setdefault(target.teacher_index, target.teacher_key)
            != target.teacher_key
        ):
            raise ValueError(
                "Expected a distinct index for each Teacher in a mixed batch"
            )
        identity = (target.model_fingerprint, target.teacher_index)
        if identities.setdefault(target.teacher_key, identity) != identity:
            raise ValueError(
                f"Teacher {target.teacher_key!r} changed identity within a batch"
            )
    result["teacher_index"] = torch.tensor([target.teacher_index for target in targets])
    if targets[0].token_ids is not None:
        result["teacher_token_ids"] = torch.zeros(shape, dtype=torch.long)
    for index, (target, prompt, response) in enumerate(
        zip(targets, prompt_lengths, response_lengths)
    ):
        start = prompt if response_only else 0
        end = prompt + response
        length = end - start
        if length > sequence_length or target.logprobs.shape[0] != end:
            raise ValueError(
                f"Teacher sequence {index}: expected original length {end} fitting batch width {sequence_length}, got {target.logprobs.shape[0]}"
            )
        result["teacher_logprobs"][index, :length] = target.logprobs[start:end]
        result["teacher_valid_mask"][index, :length] = target.valid_mask[start:end]
        if target.token_ids is not None:
            result["teacher_token_ids"][index, :length] = target.token_ids[start:end]
    return result


@dataclass(frozen=True)
class TeacherScoreRequest:
    """Unpadded original context and trainable target positions."""

    request_id: str
    sample_id: str
    group_id: str
    teacher_key: str
    input_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    topk: int | None
    tokenizer_fingerprint: str
    selected_token_ids: tuple[tuple[int, ...], ...] | None = None
    student_policy_version: str | None = None

    def validate(self, vocab_size: int) -> None:
        """Reject malformed requests before entering an inference backend."""
        prefix = f"Teacher request {self.request_id} ({self.teacher_key})"
        if not self.input_ids or len(self.input_ids) != len(self.response_mask):
            raise ValueError(f"{prefix}: expected equal nonempty token/mask lengths")
        if any(type(i) is not int or not 0 <= i < vocab_size for i in self.input_ids):
            raise ValueError(
                f"{prefix}: input IDs must belong to vocabulary of size {vocab_size}"
            )
        if self.topk is not None and (
            type(self.topk) is not int
            or self.topk != -1
            and not 1 <= self.topk <= vocab_size
        ):
            raise ValueError(
                f"{prefix}: expected K in [1, {vocab_size}], -1 or None; got {self.topk}"
            )
        if self.selected_token_ids is not None:
            if self.topk is None or self.topk == -1 or not self.student_policy_version:
                raise ValueError(
                    f"{prefix}: selected IDs require positive K and a Student version"
                )
            if len(self.selected_token_ids) != len(self.input_ids):
                raise ValueError(
                    f"{prefix}: selected IDs must match the sequence length"
                )
            for row, active in zip(self.selected_token_ids, self.response_mask):
                if len(row) != self.topk:
                    raise ValueError(f"{prefix}: expected {self.topk} IDs per position")
                if active and (
                    len(set(row)) != self.topk
                    or any(type(i) is not int or not 0 <= i < vocab_size for i in row)
                ):
                    raise ValueError(
                        f"{prefix}: selected IDs must be unique model output vocabulary IDs"
                    )


@dataclass
class TeacherTargets:
    """CPU targets; row t describes the probability of token t given its prefix."""

    request_id: str
    teacher_key: str
    model_fingerprint: str
    target_format: Literal["sampled", "sparse", "full"]
    student_policy_version: str | None
    token_ids: torch.Tensor | None
    logprobs: torch.Tensor
    valid_mask: torch.Tensor
    context: SampleContext | None = None

    @property
    def teacher_index(self) -> int:
        """Return the run-wide model index, defaulting to zero for legacy single-Teacher targets."""
        return self.context.teacher_index if self.context else 0

    def validate(self, request: TeacherScoreRequest, vocab_size: int) -> None:
        """Check response identity, support and positional alignment."""
        request.validate(vocab_size)
        prefix = f"Teacher response {request.request_id} ({request.teacher_key})"
        if (self.request_id, self.teacher_key, self.student_policy_version) != (
            request.request_id,
            request.teacher_key,
            request.student_policy_version,
        ):
            raise ValueError(f"{prefix}: response identity or Student version mismatch")
        expected_format = (
            "sampled"
            if request.topk is None
            else "full"
            if request.topk == -1
            else "sparse"
        )
        width = (
            1
            if request.topk is None
            else vocab_size
            if request.topk == -1
            else request.topk
        )
        if self.target_format != expected_format or self.logprobs.shape != (
            len(request.input_ids),
            width,
        ):
            raise ValueError(
                f"{prefix}: expected {expected_format} shape {(len(request.input_ids), width)}, got {self.target_format} {tuple(self.logprobs.shape)}"
            )
        if (
            self.logprobs.device.type != "cpu"
            or self.logprobs.dtype != torch.float32
            or self.logprobs.requires_grad
        ):
            raise ValueError(f"{prefix}: expected detached CPU float32 probabilities")
        if (
            self.valid_mask.device.type != "cpu"
            or self.valid_mask.dtype != torch.bool
            or self.valid_mask.shape != (len(request.input_ids),)
        ):
            raise ValueError(
                f"{prefix}: expected CPU bool validity mask matching sequence length"
            )
        expected_mask = torch.tensor(request.response_mask, dtype=torch.bool)
        expected_mask[0] = False
        if not torch.equal(self.valid_mask, expected_mask):
            raise ValueError(
                f"{prefix}: valid target positions differ from the request"
            )
        if not self.model_fingerprint or not torch.isfinite(self.logprobs).all():
            raise ValueError(
                f"{prefix}: expected model fingerprint and finite probabilities"
            )
        if expected_format == "full":
            if self.token_ids is not None:
                raise ValueError(f"{prefix}: full targets must omit repeated token IDs")
        else:
            ids = self.token_ids
            if (
                ids is None
                or ids.device.type != "cpu"
                or ids.dtype != torch.long
                or ids.shape != self.logprobs.shape
            ):
                raise ValueError(f"{prefix}: expected matching CPU int64 token IDs")
            if ((ids < 0) | (ids >= vocab_size)).any():
                raise ValueError(
                    f"{prefix}: target IDs must belong to the model output vocabulary"
                )
            rows = ids[self.valid_mask]
            ordered = rows.sort(-1).values
            if (ordered[:, 1:] == ordered[:, :-1]).any():
                raise ValueError(f"{prefix}: target support contains duplicate IDs")
            expected_ids = request.selected_token_ids
            if request.topk is None:
                expected_ids = tuple((i,) for i in request.input_ids)
            if expected_ids is not None and not torch.equal(
                rows, torch.tensor(expected_ids, dtype=torch.long)[self.valid_mask]
            ):
                raise ValueError(f"{prefix}: target IDs differ from requested IDs")
        masses = self.logprobs[self.valid_mask].exp().sum(-1)
        if (masses > 1 + 1e-5).any() or (
            expected_format == "full"
            and not torch.allclose(
                masses, torch.ones_like(masses), atol=1e-5, rtol=1e-5
            )
        ):
            raise ValueError(
                f"{prefix}: probabilities must retain full-vocabulary normalization"
            )
