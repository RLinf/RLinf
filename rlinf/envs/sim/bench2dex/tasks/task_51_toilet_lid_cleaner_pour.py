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

from rlinf.envs.sim.bench2dex.success import (
    get_joint_position,
    tilt_deg_from_pose,
    world_axis_from_pose,
)

# joint_0 is -1.0 with the lid fully closed and above 0.2 when it is substantially open.
LID_OPEN_MIN = 0.2
LID_CLOSED_TARGET = -1.0
LID_CLOSED_TOL = 0.10

CLEANER_ID = "obj_021_bleach_cleanser_2"
TOILET_ID = "obj_121_toilet_1"
POUR_DIST_THRESHOLD = 0.45
POUR_TILT_MIN_DEG = 50.0
POUR_LIFT_MIN_Z = 0.75
POUR_DIRECTION_COS_THRESHOLD = 0.0


def _lid_is_open(states: dict[str, Any]) -> bool:
    toilet = states.get(TOILET_ID)
    if toilet is None:
        return False
    pos = get_joint_position(toilet, "joint_0", 0.0)
    return pos > LID_OPEN_MIN


def _lid_is_closed(states: dict[str, Any]) -> bool:
    toilet = states.get(TOILET_ID)
    if toilet is None:
        return False
    pos = get_joint_position(toilet, "joint_0", 0.0)
    return abs(pos - LID_CLOSED_TARGET) < LID_CLOSED_TOL


def _cleaner_is_pouring(states: dict[str, Any]) -> bool:
    cleaner = states.get(CLEANER_ID)
    toilet = states.get(TOILET_ID)
    if cleaner is None or toilet is None:
        return False
    cp = cleaner["pose_world"]
    if float(cp[2]) < POUR_LIFT_MIN_Z:
        return False
    tilt = tilt_deg_from_pose(cleaner)
    if tilt < POUR_TILT_MIN_DEG:
        return False
    tp = toilet["pose_world"]

    dist = math.hypot(float(cp[0] - tp[0]), float(cp[1] - tp[1]))
    if dist >= POUR_DIST_THRESHOLD:
        return False

    # The XY projection of local z is the lean direction, which must point toward
    # the toilet.
    world_z = world_axis_from_pose(cleaner, "z")
    tilt_xy = (float(world_z[0]), float(world_z[1]))
    to_toilet_xy = (float(tp[0] - cp[0]), float(tp[1] - cp[1]))
    tilt_mag = math.hypot(*tilt_xy)
    toilet_mag = math.hypot(*to_toilet_xy)
    if tilt_mag < 1e-9 or toilet_mag < 1e-9:
        return False
    cos_angle = (tilt_xy[0] * to_toilet_xy[0] + tilt_xy[1] * to_toilet_xy[1]) / (
        tilt_mag * toilet_mag
    )
    return cos_angle > POUR_DIRECTION_COS_THRESHOLD


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    step = int(ctx.get("_t51_step", 0))

    if step == 0:
        if _lid_is_open(states):
            ctx["_t51_step"] = 1

    if step == 1:
        if _cleaner_is_pouring(states) and _lid_is_open(states):
            ctx["_t51_step"] = 2

    if step == 2:
        if _lid_is_closed(states):
            ctx["_t51_step"] = 3

    return int(ctx.get("_t51_step", 0)) >= 3
