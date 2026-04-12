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

"""Habitat GT-prefix metadata helpers."""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Iterable, Sequence

import torch

HABITAT_CURRENT_STEP_KEY = "habitat_current_step"
CMA_CURRENT_GT_ACTION_KEY = "gt_actions"
HABITAT_GT_CURRENT_ACTION_KEY = "habitat_gt_current_action"
HABITAT_GT_ACTION_VALID_KEY = "habitat_gt_action_valid"
HABITAT_GT_PREFIX_LENGTH_KEY = "habitat_gt_prefix_length"
HABITAT_GT_SEQUENCE_LENGTH_KEY = "habitat_gt_sequence_length"

HABITAT_GT_METADATA_KEYS = (
    HABITAT_CURRENT_STEP_KEY,
    CMA_CURRENT_GT_ACTION_KEY,
    HABITAT_GT_CURRENT_ACTION_KEY,
    HABITAT_GT_ACTION_VALID_KEY,
    HABITAT_GT_PREFIX_LENGTH_KEY,
    HABITAT_GT_SEQUENCE_LENGTH_KEY,
)


def encode_episode_state_id(episode_id: int | str) -> int:
    """Encode a Habitat episode ID into a deterministic int64-safe state ID.

    CMA reset detection only needs a stable tensor value that changes when the
    active episode changes. Real Habitat/VLN datasets may use string episode IDs,
    so this helper converts strings to a deterministic 63-bit integer.
    """
    if isinstance(episode_id, Integral):
        return int(episode_id)
    if isinstance(episode_id, str):
        digest = hashlib.blake2b(
            f"habitat_episode_id:{episode_id}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
    raise TypeError(
        "Habitat episode IDs must be integers or strings, "
        f"got {type(episode_id).__name__}."
    )


def encode_episode_state_ids(episode_ids: Sequence[int | str]) -> torch.Tensor:
    """Encode Habitat episode IDs into a tensor suitable for CMA state tracking."""
    return torch.tensor(
        [encode_episode_state_id(episode_id) for episode_id in episode_ids],
        dtype=torch.int64,
    )


def normalize_gt_action_sequence(
    gt_actions: Sequence[int] | None,
    *,
    episode_id: int | str,
    valid_action_ids: Iterable[int],
) -> tuple[int, ...] | None:
    """Validate and normalize a Habitat episode GT-action sequence."""
    if gt_actions is None:
        return None

    valid_action_ids = set(valid_action_ids)
    normalized_actions: list[int] = []
    for step_idx, action_id in enumerate(gt_actions):
        if not isinstance(action_id, Integral):
            raise ValueError(
                "Habitat GT action ids must be integers, "
                f"got {type(action_id).__name__} for episode {episode_id} at step {step_idx}."
            )

        action_id = int(action_id)
        if action_id not in valid_action_ids:
            raise ValueError(
                f"Invalid Habitat GT action id {action_id} for episode {episode_id} at step {step_idx}."
            )
        normalized_actions.append(action_id)

    return tuple(normalized_actions)


def build_gt_prefix_metadata(
    episode_ids: Sequence[int | str],
    elapsed_steps: Sequence[int],
    episode_gt_action_sequences: Sequence[Sequence[int] | None],
    *,
    valid_action_ids: Iterable[int],
) -> dict[str, torch.Tensor]:
    """Build Habitat-specific GT-prefix tensors for the current decision step.

    ``elapsed_steps`` counts already-executed actions. Rollout consumes the
    observation before the *next* action selection, so GT metadata must be keyed
    to decision step ``elapsed_steps + 1``.
    """
    if not (len(episode_ids) == len(elapsed_steps) == len(episode_gt_action_sequences)):
        raise ValueError(
            "Habitat GT metadata inputs must have the same length: "
            f"{len(episode_ids)=}, {len(elapsed_steps)=}, "
            f"{len(episode_gt_action_sequences)=}."
        )

    current_steps: list[int] = []
    current_gt_actions: list[int] = []
    current_gt_action_valids: list[bool] = []
    prefix_lengths: list[int] = []
    sequence_lengths: list[int] = []

    for episode_id, elapsed_step, gt_action_sequence in zip(
        episode_ids, elapsed_steps, episode_gt_action_sequences
    ):
        normalized_actions = normalize_gt_action_sequence(
            gt_action_sequence,
            episode_id=episode_id,
            valid_action_ids=valid_action_ids,
        )

        step = max(int(elapsed_step), 0) + 1
        seq_len = 0 if normalized_actions is None else len(normalized_actions)
        prefix_len = min(step, seq_len)
        has_current_gt_action = normalized_actions is not None and 0 < step <= seq_len
        current_gt_action = 0
        if has_current_gt_action and normalized_actions is not None:
            current_gt_action = normalized_actions[step - 1]

        current_steps.append(step)
        current_gt_actions.append(current_gt_action)
        current_gt_action_valids.append(has_current_gt_action)
        prefix_lengths.append(prefix_len)
        sequence_lengths.append(seq_len)

    return {
        HABITAT_CURRENT_STEP_KEY: torch.tensor(current_steps, dtype=torch.int64),
        CMA_CURRENT_GT_ACTION_KEY: torch.tensor(current_gt_actions, dtype=torch.int64),
        HABITAT_GT_CURRENT_ACTION_KEY: torch.tensor(
            current_gt_actions, dtype=torch.int64
        ),
        HABITAT_GT_ACTION_VALID_KEY: torch.tensor(
            current_gt_action_valids, dtype=torch.bool
        ),
        HABITAT_GT_PREFIX_LENGTH_KEY: torch.tensor(prefix_lengths, dtype=torch.int64),
        HABITAT_GT_SEQUENCE_LENGTH_KEY: torch.tensor(
            sequence_lengths, dtype=torch.int64
        ),
    }
