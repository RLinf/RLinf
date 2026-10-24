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

BOX_ID = "obj_154_wooden_box_1"
SOY_ID = "obj_172_soy_sauce_2"
VINEGAR_ID = "obj_173_vinegar_3"
MSG_ID = "obj_171_msg_4"

_BOX_UPRIGHT = {
    "type": "object_upright",
    "object": BOX_ID,
    "tolerance_deg": 30,
    "local_axis": "y",
}


def _inside(obj_id: str) -> dict:
    return {
        "type": "object_inside",
        "object": obj_id,
        "container": BOX_ID,
        "tolerance": 0.18,
        "container_center_offset": [0.0, 0.075, 0.0],
        "object_center_offset": {
            SOY_ID: [0.0, 0.075, 0.0],
            VINEGAR_ID: [-8.6e-06, 0.1125, -2.6e-06],
            MSG_ID: [-2.0e-06, 0.0475284, 2.2e-06],
        }[obj_id],
    }


def _upright(obj_id: str) -> dict:
    # The bottles spawn with local y along world +z.
    return {
        "type": "object_upright",
        "object": obj_id,
        "tolerance_deg": 30,
        "local_axis": "y",
    }


def _static(obj_id: str) -> dict:
    return {
        "type": "object_static",
        "object": obj_id,
        "threshold": 0.05,
        "check_angular": False,
    }


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    return evaluate_conditions(
        [
            _BOX_UPRIGHT,
            _inside(SOY_ID),
            _upright(SOY_ID),
            _inside(VINEGAR_ID),
            _upright(VINEGAR_ID),
            _inside(MSG_ID),
            _static(SOY_ID),
            _static(VINEGAR_ID),
            _static(MSG_ID),
        ],
        states,
        ctx,
    )
