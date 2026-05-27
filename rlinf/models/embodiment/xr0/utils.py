# Copyright 2026 The RLinf Authors.
# Adapted from Xiaomi-Robotics-0 (xr0_src/xr0/mibot/utils/io.py).
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

"""XR0 action dimension constants and normalization utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_DIM = 32
STATE_DIM = 32
ACTION_EPS = 1e-6

ACTION_PARTS = (
    ("left_ee_pos", slice(0, 3)),
    ("left_ee_aa", slice(3, 6)),
    ("left_gripper", slice(6, 7)),
    ("left_joint", slice(7, 13)),
    ("right_ee_pos", slice(14, 17)),
    ("right_ee_aa", slice(17, 20)),
    ("right_gripper", slice(20, 21)),
    ("right_joint", slice(21, 27)),
)


def validate_stats(
    mean: Sequence[Sequence[float]],
    std: Sequence[Sequence[float]],
    action_length: int,
):
    """Validate that mean/std arrays have the expected shape."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (action_length, ACTION_DIM):
        raise ValueError(
            f"mean expected shape {(action_length, ACTION_DIM)}, got {mean.shape}"
        )
    if std.shape != (action_length, ACTION_DIM):
        raise ValueError(
            f"std expected shape {(action_length, ACTION_DIM)}, got {std.shape}"
        )
    return mean, std


def normalize_action(action, mean, std) -> np.ndarray:
    """Normalize actions using per-timestep mean/std."""
    return (action - mean) / (std + ACTION_EPS)


def denormalize_action(action, mean, std):
    """Denormalize actions using per-timestep mean/std."""
    return action * (std + ACTION_EPS) + mean
