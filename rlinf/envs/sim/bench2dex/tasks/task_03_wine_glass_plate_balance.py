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

import itertools
import math
from typing import Any

from rlinf.envs.sim.bench2dex.success import evaluate_conditions, world_axis_from_pose

GLASS_IDS = ("obj_194_wineglass_1", "obj_194_wineglass_2", "obj_194_wineglass_3")

TARGETS: list[tuple[float, float]] = [
    (0.58, 0.25),
    (0.08, 0.25),
    (-0.42, 0.25),
]

TOLERANCE_M = 0.10
UPRIGHT_TOLERANCE_DEG = 30.0
# The glasses spawn with rpy_deg [90, 0, 0], which turns local y into world +z.
LOCAL_UPRIGHT_AXIS = "y"


def _pos_xy(states: dict[str, Any], obj_id: str) -> tuple[float, float]:
    pose = states[obj_id]["pose_world"]
    return (float(pose[0]), float(pose[1]))


def _dist_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _upright(states: dict[str, Any], obj_id: str) -> bool:
    state = states.get(obj_id)
    if state is None or state.get("pose_world") is None:
        return False
    axis = world_axis_from_pose(state, LOCAL_UPRIGHT_AXIS)
    return axis[2] > math.cos(math.radians(UPRIGHT_TOLERANCE_DEG))


def _count_placed(states: dict[str, Any]) -> int:
    if not all(
        obj_id in states and states[obj_id].get("pose_world") is not None
        for obj_id in GLASS_IDS
    ):
        return 0

    glass_positions = [_pos_xy(states, g) for g in GLASS_IDS]
    best = 0
    for perm in itertools.permutations(range(3)):
        count = sum(
            1
            for gi, ti in enumerate(perm)
            if _dist_xy(glass_positions[gi], TARGETS[ti]) <= TOLERANCE_M
        )
        best = max(best, count)
    return best


def _glasses_static(states: dict[str, Any]) -> bool:
    # A glass rolling through a target must not count as placed.
    conds = [
        {"type": "object_static", "object": g, "threshold": 0.05, "check_angular": True}
        for g in GLASS_IDS
    ]
    return evaluate_conditions(conds, states, ctx={})


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    return (
        _count_placed(states) >= 3
        and all(_upright(states, g) for g in GLASS_IDS)
        and _glasses_static(states)
    )
