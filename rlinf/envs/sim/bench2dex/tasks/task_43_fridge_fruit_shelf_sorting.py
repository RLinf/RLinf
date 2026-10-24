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

_FRIDGE = "obj_111_refrigerator_3"

_OBJECT_IN_ZONE = {
    "obj_011_banana_2": {
        "type": "object_in_container_zone",
        "object": "obj_011_banana_2",
        "container": _FRIDGE,
        "object_center_offset": [0.011476, -0.0073435, 0.017963],
        "container_frame": True,
        "container_center_offset": [-0.0110169, 0.000875, -0.0118804],
        "zone_lo": [-0.1190998, -0.1026261, -0.3019416],
        "zone_hi": [0.1190998, 0.1026261, 0.3019416],
    },
    "obj_013_apple_1": {
        "type": "object_in_container_zone",
        "object": "obj_013_apple_1",
        "container": _FRIDGE,
        "object_center_offset": [0.000859, -0.0037845, 0.0355515],
        "container_frame": True,
        "container_center_offset": [-0.0110169, 0.000875, -0.0118804],
        "zone_lo": [-0.1190998, -0.1026261, -0.3019416],
        "zone_hi": [0.1190998, 0.1026261, 0.3019416],
    },
}

_STATIC = [
    {
        "type": "object_static",
        "object": "obj_011_banana_2",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_013_apple_1",
        "threshold": 0.05,
        "check_angular": False,
    },
]

_DOOR_CLOSED = [
    {
        "type": "joint_state",
        "object": _FRIDGE,
        "joint": "joint_0",
        "target": "closed",
        "tolerance": 0.1,
    },
    {
        "type": "joint_state",
        "object": _FRIDGE,
        "joint": "joint_1",
        "target": "closed",
        "tolerance": 0.1,
    },
]

_ALL_OBJ_IDS = list(_OBJECT_IN_ZONE.keys())


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    conditions: list[dict] = []
    for obj_id in _ALL_OBJ_IDS:
        conditions.append(_OBJECT_IN_ZONE[obj_id])
    conditions.extend(_STATIC)
    conditions.extend(_DOOR_CLOSED)
    return evaluate_conditions(conditions, states, ctx)
