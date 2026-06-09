# Copyright 2025 The RLinf Authors.
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

"""Canonical RLT stage2 transition semantics.

This module is intentionally environment-agnostic. ManiSkill and realworld
rollouts may collect data differently, but both should enter stage2 training
through this source/replay vocabulary.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class TransitionSource(IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


def resolve_collection_phase_id(phase: str | int | None) -> int:
    if phase is None:
        return COLLECTION_PHASE_UNKNOWN
    if isinstance(phase, int):
        return int(phase)
    phase_name = str(phase).split(":", 1)[0].lower()
    if phase_name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if phase_name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def resolve_chunk_source(source_chunk: np.ndarray) -> int:
    values = {int(value) for value in np.asarray(source_chunk).reshape(-1)}
    if not values:
        return int(TransitionSource.RL)
    if int(TransitionSource.MIXED) in values or len(values) > 1:
        return int(TransitionSource.MIXED)
    return next(iter(values))


def human_source_mask(source_chunk) -> np.ndarray:
    source_array = np.asarray(source_chunk)
    return np.logical_or(
        source_array == int(TransitionSource.HUMAN),
        source_array == int(TransitionSource.MIXED),
    )
