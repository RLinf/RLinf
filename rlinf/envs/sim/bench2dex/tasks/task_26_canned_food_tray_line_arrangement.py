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

_OBJECT_ON = {
    "obj_002_master_chef_can_2": {
        "type": "object_on",
        "object": "obj_002_master_chef_can_2",
        "target": "obj_130_tray_1",
        "tolerance_xy": 0.18,
        "tolerance_z": 0.15,
        "target_center_offset": [1.21e-05, 0.0189337, 0.0],
        "object_center_offset": [-0.0136388, -0.0078084, 0.0629114],
    },
    "obj_171_msg_3": {
        "type": "object_on",
        "object": "obj_171_msg_3",
        "target": "obj_130_tray_1",
        "tolerance_xy": 0.18,
        "tolerance_z": 0.15,
        "target_center_offset": [1.21e-05, 0.0189337, 0.0],
        "object_center_offset": [-2e-06, 0.0475284, 2.2e-06],
    },
    "obj_151_milk_box_4": {
        "type": "object_on",
        "object": "obj_151_milk_box_4",
        "target": "obj_130_tray_1",
        "tolerance_xy": 0.18,
        "tolerance_z": 0.15,
        "target_center_offset": [1.21e-05, 0.0189337, 0.0],
        "object_center_offset": [0.0, 0.0, 0.0],
    },
    "obj_010_potted_meat_can_5": {
        "type": "object_on",
        "object": "obj_010_potted_meat_can_5",
        "target": "obj_130_tray_1",
        "tolerance_xy": 0.18,
        "tolerance_z": 0.15,
        "target_center_offset": [1.21e-05, 0.0189337, 0.0],
        "object_center_offset": [-0.0327845, -0.0265525, 0.0385835],
    },
}

_STATIC = [
    {
        "type": "object_static",
        "object": "obj_002_master_chef_can_2",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_171_msg_3",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_151_milk_box_4",
        "threshold": 0.05,
        "check_angular": False,
    },
    {
        "type": "object_static",
        "object": "obj_010_potted_meat_can_5",
        "threshold": 0.05,
        "check_angular": False,
    },
]

_TRAY_UPRIGHT = [
    {
        "type": "object_upright",
        "object": "obj_130_tray_1",
        "tolerance_deg": 30,
        "local_axis": "y",
    },
]

# The local axis that points up when each object spawns upright.
_OBJECT_UPRIGHT_AXIS = {
    "obj_002_master_chef_can_2": "z",
    "obj_171_msg_3": "y",
    "obj_151_milk_box_4": "y",
    "obj_010_potted_meat_can_5": "z",
}

_OBJECT_UPRIGHT_TOL_DEG = 25.0


def _upright_for(obj_id: str) -> list[dict]:
    axis = _OBJECT_UPRIGHT_AXIS.get(obj_id)
    if axis is None:
        return []
    return [
        {
            "type": "object_upright",
            "object": obj_id,
            "tolerance_deg": _OBJECT_UPRIGHT_TOL_DEG,
            "local_axis": axis,
        },
    ]


_ALL_OBJ_IDS = list(_OBJECT_ON.keys())


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    conditions: list[dict] = []
    for obj_id in _ALL_OBJ_IDS:
        conditions.append(_OBJECT_ON[obj_id])
        conditions.extend(_upright_for(obj_id))
    conditions.extend(_STATIC)
    conditions.extend(_TRAY_UPRIGHT)
    return evaluate_conditions(conditions, states, ctx)
