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

"""Psi0 SIMPLE observation and action layout helpers."""

from __future__ import annotations

import numpy as np

SIMPLE_ACTION_CHUNK_SIZE = 24
SIMPLE_ACTION_DIM = 36
SIMPLE_STATE_DIM = 32
SIMPLE_JOINT_QPOS_MIN_DIM = 43
SIMPLE_ACTION_CHUNK_SHAPE = (SIMPLE_ACTION_CHUNK_SIZE, SIMPLE_ACTION_DIM)

_STATE_SLICES = (
    (29, 32),
    (34, 36),
    (32, 34),
    (36, 43),
    (15, 22),
    (22, 29),
)


def extract_psi0_state(joint_qpos: np.ndarray, base_height: float) -> np.ndarray:
    """Build the official SIMPLE Teleop proprioceptive state."""
    joint_qpos = np.asarray(joint_qpos)
    if joint_qpos.ndim != 1 or joint_qpos.shape[0] < SIMPLE_JOINT_QPOS_MIN_DIM:
        raise ValueError(
            "SIMPLE joint_qpos must be one-dimensional with at least "
            f"{SIMPLE_JOINT_QPOS_MIN_DIM} joints."
        )
    waist_rpy = joint_qpos[[13, 14, 12]]
    state = np.concatenate(
        [joint_qpos[start:end] for start, end in _STATE_SLICES]
        + [waist_rpy, np.asarray([base_height])]
    )
    if state.shape != (SIMPLE_STATE_DIM,):
        raise RuntimeError(f"Unexpected Psi0 SIMPLE state shape: {state.shape}.")
    return state.astype(np.float32, copy=False)


def psi0_upper_joint_values(action: np.ndarray) -> np.ndarray:
    """Reorder Psi0's first 28 values into SIMPLE's upper-joint order."""
    action = np.asarray(action)
    if action.shape != (SIMPLE_ACTION_DIM,):
        raise ValueError(
            "Psi0 SIMPLE action must have shape "
            f"({SIMPLE_ACTION_DIM},), got {action.shape}."
        )
    return np.concatenate(
        (
            action[14:28],
            action[0:3],
            action[5:7],
            action[3:5],
            action[7:14],
        )
    ).astype(np.float32, copy=False)
