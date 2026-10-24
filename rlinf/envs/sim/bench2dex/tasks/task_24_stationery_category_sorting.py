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

_PEN_CUP = "obj_167_pencup_1"
_PLASTIC_BOX = "obj_169_plasticbox_2"

_CUP_UPRIGHT = {
    "type": "object_upright",
    "object": _PEN_CUP,
    "tolerance_deg": 30,
    "local_axis": "y",
}
_BOX_UPRIGHT = {
    "type": "object_upright",
    "object": _PLASTIC_BOX,
    "tolerance_deg": 30,
    "local_axis": "y",
}

_PEN_CUP_CENTER = [0.0, 0.12, 0.0]
_PEN_CUP_ZONE_LO = [-0.08, -0.12, -0.08]
_PEN_CUP_ZONE_HI = [0.08, 0.12, 0.08]

_BOX_CENTER = [0.0, 0.055, 0.0]
_BOX_ZONE_LO = [-0.12, -0.06, -0.08]
_BOX_ZONE_HI = [0.12, 0.07, 0.08]

_OBJECTS: dict[str, dict] = {
    "obj_221_pen_3": {
        "container": _PEN_CUP,
        "container_center": _PEN_CUP_CENTER,
        "zone_lo": _PEN_CUP_ZONE_LO,
        "zone_hi": _PEN_CUP_ZONE_HI,
        "container_upright": _CUP_UPRIGHT,
        # A pen standing in the cup points its long local y axis up.
        "upright": {
            "type": "object_upright",
            "object": "obj_221_pen_3",
            "tolerance_deg": 60,
            "local_axis": "y",
        },
    },
    "obj_166_markpen_4": {
        "container": _PEN_CUP,
        "container_center": _PEN_CUP_CENTER,
        "zone_lo": _PEN_CUP_ZONE_LO,
        "zone_hi": _PEN_CUP_ZONE_HI,
        "container_upright": _CUP_UPRIGHT,
        # A pen standing in the cup points its long local y axis up.
        "upright": {
            "type": "object_upright",
            "object": "obj_166_markpen_4",
            "tolerance_deg": 60,
            "local_axis": "y",
        },
    },
    "obj_199_glue_5": {
        "container": _PLASTIC_BOX,
        "container_center": _BOX_CENTER,
        "zone_lo": _BOX_ZONE_LO,
        "zone_hi": _BOX_ZONE_HI,
        "container_upright": _BOX_UPRIGHT,
        # Glue may lie on its side, so it has no upright check.
    },
    "obj_168_battery_6": {
        "container": _PLASTIC_BOX,
        "container_center": _BOX_CENTER,
        "zone_lo": _BOX_ZONE_LO,
        "zone_hi": _BOX_ZONE_HI,
        "container_upright": _BOX_UPRIGHT,
        # Battery orientation is unconstrained.
    },
}

_ALL_OBJ_IDS = list(_OBJECTS.keys())


def _in_zone_cond(obj_id: str) -> dict:
    spec = _OBJECTS[obj_id]
    return {
        "type": "object_in_container_zone",
        "object": obj_id,
        "container": spec["container"],
        "container_center_offset": spec["container_center"],
        "zone_lo": spec["zone_lo"],
        "zone_hi": spec["zone_hi"],
        "container_frame": True,
    }


def _static_cond(obj_id: str) -> dict:
    return {
        "type": "object_static",
        "object": obj_id,
        "threshold": 0.05,
        "check_angular": False,
    }


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    conditions: list[dict] = []
    for obj_id in _ALL_OBJ_IDS:
        spec = _OBJECTS[obj_id]
        conditions.append(spec["container_upright"])
        conditions.append(_in_zone_cond(obj_id))
        if "upright" in spec:
            conditions.append(spec["upright"])
        conditions.append(_static_cond(obj_id))
    return evaluate_conditions(conditions, states, ctx)
