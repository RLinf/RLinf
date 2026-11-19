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

"""Frozen inference backend and normalization of prompt-token scores."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets


@dataclass(frozen=True)
class TeacherModelInfo:
    """Immutable model identity and complete output-vocabulary size."""

    model_fingerprint: str
    tokenizer_fingerprint: str
    vocab_size: int
    max_context_length: int


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash token mappings and special/added token semantics."""
    special = dict(tokenizer.special_tokens_map)
    if not special.get("pad_token") and special.get("eos_token"):
        special["pad_token"] = special["eos_token"]
    payload = {
        "vocab": tokenizer.get_vocab(),
        "added": tokenizer.get_added_vocab(),
        "special": special,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


class TeacherScoringBackend(ABC):
    """Own an independent frozen engine without Student weight synchronization."""

    def __init__(self, model_config, teacher_key: str):
        self.config = model_config
        self.teacher_key = teacher_key
        self.engine = None
        self.info = None
        self.model_vocab_size = None

    def load_model_info(self) -> TeacherModelInfo:
        """Load tokenizer identity before allocating the inference engine."""
        from transformers import AutoConfig

        from rlinf.models.tokenization import hf_tokenizer

        path = self.config.model_path
        trust = self.config.inference.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(path, trust_remote_code=trust)
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust)
        text_config = config.get_text_config()
        self.model_vocab_size = text_config.vocab_size
        ids = set(tokenizer.get_vocab().values())
        if ids != set(range(len(ids))):
            raise ValueError("Teacher requires a contiguous real tokenizer vocabulary")
        vocab = len(ids)
        if self.model_vocab_size < vocab:
            raise ValueError(
                "Teacher model vocabulary does not cover all tokenizer IDs"
            )
        weights = hashlib.sha256()
        for shard in sorted(Path(path).glob("*")):
            if shard.suffix in {".safetensors", ".bin", ".pt"} and shard.is_file():
                weights.update(shard.name.encode())
                with shard.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        weights.update(chunk)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "path": path,
                    "weights": weights.hexdigest(),
                    "revision": getattr(config, "_commit_hash", None),
                    "config": config.to_dict(),
                    "tokenizer": tokenizer_fingerprint(tokenizer),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        self.info = TeacherModelInfo(
            fingerprint,
            tokenizer_fingerprint(tokenizer),
            self.model_vocab_size,
            self.config.inference.get(
                "max_model_len", getattr(text_config, "max_position_embeddings", 32768)
            ),
        )
        return self.info

    def validate_request(self, request: TeacherScoreRequest) -> None:
        """Check identity and context capacity without exposing prompt contents."""
        if self.info is None:
            raise RuntimeError("Teacher must be initialized before scoring")
        request.validate(self.info.vocab_size)
        width = self.info.vocab_size if request.topk == -1 else request.topk or 1
        estimated_bytes = (
            len(request.input_ids) * width * (4 if request.topk == -1 else 12)
        )
        # Account for engine rows and normalized dictionaries, not just tensors.
        retained_scores = len(request.input_ids) * (width + 1)
        transient_scores = (
            (self.model_vocab_size or self.info.vocab_size) + 1
            if request.selected_token_ids is not None
            else 0
        )
        estimated_bytes += 256 * (retained_scores + transient_scores)
        budget = self.config.inference.get("max_payload_bytes", 256 * 1024 * 1024)
        if estimated_bytes > budget:
            raise ValueError(
                f"Teacher request {request.request_id}: estimated scoring payload {estimated_bytes} exceeds budget {budget}"
            )
        if (
            request.teacher_key != self.teacher_key
            or request.tokenizer_fingerprint != self.info.tokenizer_fingerprint
        ):
            raise ValueError(
                f"Teacher request {request.request_id}: model key or tokenizer mismatch"
            )
        if len(request.input_ids) + 1 > self.info.max_context_length:
            raise ValueError(
                f"Teacher request {request.request_id}: expected context <= {self.info.max_context_length - 1}, got {len(request.input_ids)}"
            )

    async def score_response_tokens(
        self, request: TeacherScoreRequest
    ) -> TeacherTargets:
        """Score original prefixes, retaining only the requested distribution support."""
        self.validate_request(request)
        rows = [None] * len(request.input_ids)
        vocabulary = tuple(range(self.info.vocab_size))
        for position in range(1, len(request.input_ids)):
            if not request.response_mask[position]:
                continue
            selected = (
                request.selected_token_ids[position]
                if request.selected_token_ids is not None
                else vocabulary
            )
            row = await self.score_next(
                request.input_ids[:position],
                selected,
                f"{request.request_id}-{position}",
            )
            if request.selected_token_ids is None and request.topk != -1:
                selected = sorted(row, key=lambda i: (-row[i], i))[: request.topk]
            rows[position] = {i: row[i] for i in selected}
        return self.normalize(request, rows)

    @abstractmethod
    async def score_next(
        self, prefix: tuple[int, ...], selected_ids: tuple[int, ...], request_id: str
    ) -> dict[int, float]:
        """Return selected next-token scores with full-vocabulary normalization."""

    def normalize(
        self, request: TeacherScoreRequest, rows: list[dict[int, float] | None]
    ) -> TeacherTargets:
        """Align model-normalized scores with the requested target positions."""
        self.validate_request(request)
        size = len(request.input_ids)
        if len(rows) != size:
            raise ValueError(
                f"Teacher request {request.request_id}: expected {size} score rows, got {len(rows)}"
            )
        full = request.topk == -1
        width = self.info.vocab_size if full else request.topk or 1
        scores = torch.zeros((size, width), dtype=torch.float32)
        ids = None if full else torch.zeros((size, width), dtype=torch.long)
        mask = torch.tensor(request.response_mask, dtype=torch.bool)
        mask[0] = False
        for position in mask.nonzero().flatten().tolist():
            row = rows[position]
            if not row:
                raise ValueError(
                    f"Teacher request {request.request_id}: missing scores at position {position}"
                )
            if full:
                selected = list(range(self.info.vocab_size))
            elif request.selected_token_ids is not None:
                selected = request.selected_token_ids[position]
            elif request.topk is None:
                selected = [request.input_ids[position]]
            else:
                selected = sorted(
                    (i for i in row if 0 <= i < self.info.vocab_size),
                    key=lambda i: (-row[i], i),
                )[:width]
            if len(selected) != width or any(i not in row for i in selected):
                raise ValueError(
                    f"Teacher request {request.request_id}: expected {width} requested scores at position {position}"
                )
            scores[position] = torch.tensor(
                [row[i] for i in selected], dtype=torch.float32
            )
            if ids is not None:
                ids[position] = torch.tensor(selected, dtype=torch.long)
        result = TeacherTargets(
            request.request_id,
            request.teacher_key,
            self.info.model_fingerprint,
            "full" if full else "sampled" if request.topk is None else "sparse",
            request.student_policy_version,
            ids,
            scores,
            mask,
        )
        result.validate(request, self.info.vocab_size)
        return result

    @abstractmethod
    async def initialize(self) -> TeacherModelInfo:
        """Start the engine and return its immutable identity."""

    @abstractmethod
    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        """Score existing tokens under their original prefixes."""

    @abstractmethod
    async def close(self) -> None:
        """Idempotently release engine resources."""
