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

from rlinf.envs.sim.bench2dex.success import evaluate_conditions

_PIECES = ("obj_326_piece_2", "obj_326_piece_3")

_TARGETS = {
    "obj_326_piece_2": (-0.065, 0.300),
    "obj_326_piece_3": (-0.150, 0.270),
}
_XY_TOL = 0.020
_Z_TOL = 0.010


def _pos(states: dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _static(states: dict[str, Any]) -> bool:
    # The 0.10 threshold tolerates interlock vibration after placement so the dwell
    # window stays contiguous.
    conds = [
        {
            "type": "object_static",
            "object": pid,
            "threshold": 0.10,
            "check_angular": False,
        }
        for pid in _PIECES
    ]
    return evaluate_conditions(conds, states, ctx={})


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    if not all(pid in states for pid in _PIECES):
        return False
    for pid, (tx, ty) in _TARGETS.items():
        px, py, _ = _pos(states, pid)
        if math.hypot(px - tx, py - ty) > _XY_TOL:
            return False
    ref_z = _pos(states, "obj_326_piece_1")[2]
    for pid in _PIECES:
        pz = _pos(states, pid)[2]
        if abs(pz - ref_z) > _Z_TOL:
            return False
    return _static(states)
