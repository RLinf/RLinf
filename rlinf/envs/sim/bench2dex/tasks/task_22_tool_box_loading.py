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

CONDITIONS = [
    {
        "type": "all",
        "conditions": [
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_043_phillips_screwdriver_2",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                        "container_center_offset": [0.0, 0.07, 0.0],
                        "object_center_offset": [-0.042698, 0.0033815, 0.0173985],
                    },
                ],
            },
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_044_flat_screwdriver_3",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                        "container_center_offset": [0.0, 0.07, 0.0],
                        "object_center_offset": [-0.0124925, 0.0116655, 0.0171135],
                    },
                ],
            },
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_042_adjustable_wrench_4",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                        "container_center_offset": [0.0, 0.07, 0.0],
                        "object_center_offset": [-0.0034875, -0.0123455, 0.008692],
                    },
                ],
            },
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_147_drill_5",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                        "container_center_offset": [0.0, 0.07, 0.0],
                        "object_center_offset": [0.0, 0.0900622, 0.0],
                    },
                ],
            },
        ],
    }
]


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
