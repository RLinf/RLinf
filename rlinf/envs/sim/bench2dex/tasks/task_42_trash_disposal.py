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

from typing import Any

from rlinf.envs.sim.bench2dex.success import evaluate_conditions

_TRASHCAN = "obj_122_trashcan_6"

_OBJECT_INSIDE = {
    "obj_341_banana_peel_1": {
        "type": "object_inside",
        "object": "obj_341_banana_peel_1",
        "container": _TRASHCAN,
        "tolerance": 0.25,
        "z_offset": 100.0,
    },
    "obj_349_paper_low_2": {
        "type": "object_inside",
        "object": "obj_349_paper_low_2",
        "container": _TRASHCAN,
        "tolerance": 0.25,
        "z_offset": 100.0,
    },
    "obj_080_bottle_3": {
        "type": "object_inside",
        "object": "obj_080_bottle_3",
        "container": _TRASHCAN,
        "tolerance": 0.25,
        "z_offset": 100.0,
    },
}

_STATIC = [
    {
        "type": "object_static",
        "object": "obj_341_banana_peel_1",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_349_paper_low_2",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_080_bottle_3",
        "threshold": 0.05,
        "check_angular": False,
    },
]

_TRASHCAN_UPRIGHT = [
    {"type": "object_upright", "object": _TRASHCAN, "tolerance_deg": 30},
]

_ALL_OBJ_IDS = list(_OBJECT_INSIDE.keys())


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    conditions: list[dict] = []
    for obj_id in _ALL_OBJ_IDS:
        conditions.append(_OBJECT_INSIDE[obj_id])
    conditions.extend(_STATIC)
    conditions.extend(_TRASHCAN_UPRIGHT)
    return evaluate_conditions(conditions, states, ctx)
