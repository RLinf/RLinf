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

# This file contains code copied from the Bench2Dex project: https://github.com/Bench2Dex/Bench2Dex.
# The original code is licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any

StateMap = dict[str, Any]

PUZZLE_PIECES = tuple(f"obj_326_piece_{index}" for index in range(1, 6))
_PUZZLE_TARGETS = {
    "obj_326_piece_2": (-0.065, 0.300),
    "obj_326_piece_3": (-0.150, 0.270),
    "obj_326_piece_4": (-0.150, 0.190),
    "obj_326_piece_5": (-0.065, 0.215),
}
_PUZZLE_XY_TOL = 0.020
_PUZZLE_Z_TOL = 0.010


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(
        object_id in states and states[object_id].get("pose_world") is not None
        for object_id in object_ids
    )


def _pos(states: StateMap, object_id: str) -> tuple[float, float, float]:
    pose = states[object_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _speed(state: dict[str, Any], *, angular: bool = False) -> float:
    key = "ang_vel_world" if angular else "lin_vel_world"
    values = state.get(key, [0.0, 0.0, 0.0])
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _static(
    states: StateMap, *object_ids: str, linear: float = 0.06, angular: float = 0.5
) -> bool:
    return all(
        _speed(states[object_id]) < linear
        and _speed(states[object_id], angular=True) < angular
        for object_id in object_ids
    )


def _puzzle_assembled(states: StateMap) -> bool:
    if not _has(states, *PUZZLE_PIECES):
        return False
    for piece, (tx, ty) in _PUZZLE_TARGETS.items():
        if piece not in states:
            return False
        px, py, _ = _pos(states, piece)
        if math.hypot(px - tx, py - ty) > _PUZZLE_XY_TOL:
            return False
    ref_z = _pos(states, "obj_326_piece_1")[2]
    for piece in _PUZZLE_TARGETS:
        pz = _pos(states, piece)[2]
        if abs(pz - ref_z) > _PUZZLE_Z_TOL:
            return False
    return _static(states, *PUZZLE_PIECES)


def check_success(
    states: StateMap, ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    return _puzzle_assembled(states)
