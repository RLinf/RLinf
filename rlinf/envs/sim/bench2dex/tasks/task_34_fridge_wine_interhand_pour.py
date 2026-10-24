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
    evaluate_conditions,
    get_joint_position,
    tilt_deg_from_pose,
    world_axis_from_pose,
)

RECEIVER_DIST_THRESHOLD = 0.35
RECEIVER_ALIGN_TOL_DEG = 50.0
POUR_TILT_MIN_DEG = 50.0
LIFT_MIN_Z = 0.82
JOINT_OPEN_TOLERANCE = 1.22
JOINT_CLOSED_TOLERANCE = 0.25

# The returned bottle may lean but must not lie toppled.
BOTTLE_UPRIGHT_MAX_TILT_DEG = 60.0

FRIDGE_ID = "obj_111_refrigerator_1"
BOTTLE_ID = "obj_224_wine_bottle_1"
GLASS_IDS = ("obj_194_wineglass_2",)

BOTTLE_IN_FRIDGE_CONDITION = {
    "type": "object_in_container_zone",
    "object": BOTTLE_ID,
    "container": FRIDGE_ID,
    "container_frame": True,
    "container_center_offset": [-0.046, -0.019, -0.336],
    "zone_lo": [-0.10, -0.08, -0.04],
    "zone_hi": [0.10, 0.11, 0.05],
}


def _get_pos(states: dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _check_fridge_open(states: dict[str, Any]) -> bool:
    fridge = states.get(FRIDGE_ID)
    if fridge is None:
        return False
    pos = get_joint_position(fridge, "joint_0", 0.0)
    return abs(pos) > JOINT_OPEN_TOLERANCE


def _check_fridge_closed(states: dict[str, Any]) -> bool:
    fridge = states.get(FRIDGE_ID)
    if fridge is None:
        return False
    pos = get_joint_position(fridge, "joint_0", 0.0)
    return abs(pos) < JOINT_CLOSED_TOLERANCE


def _check_bottle_lifted(states: dict[str, Any]) -> bool:
    if BOTTLE_ID not in states:
        return False
    _ox, _oy, oz = _get_pos(states, BOTTLE_ID)
    return oz > LIFT_MIN_Z


def _check_bottle_in_fridge(states: dict[str, Any]) -> bool:
    if BOTTLE_ID not in states or FRIDGE_ID not in states:
        return False
    return evaluate_conditions([BOTTLE_IN_FRIDGE_CONDITION], states, {})


def _check_bottle_upright(states: dict[str, Any]) -> bool:
    if BOTTLE_ID not in states:
        return False
    tilt = tilt_deg_from_pose(states[BOTTLE_ID])
    return tilt <= BOTTLE_UPRIGHT_MAX_TILT_DEG


def _faces_target(
    source: dict[str, Any],
    target: dict[str, Any],
    tol_deg: float = RECEIVER_ALIGN_TOL_DEG,
) -> bool:
    sp = source["pose_world"]
    tp = target["pose_world"]
    dx = float(tp[0] - sp[0])
    dy = float(tp[1] - sp[1])
    dist_xy = math.hypot(dx, dy)
    if dist_xy <= 1.0e-6 or dist_xy > RECEIVER_DIST_THRESHOLD:
        return False

    # Bottle orientations differ, so either local x or z may face the glass.
    for axis_name in ("x", "z", "-x", "-z"):
        ax, ay, _az = world_axis_from_pose(source, axis_name.lstrip("-"))
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_xy_norm = math.hypot(ax, ay)
        if axis_xy_norm <= 1.0e-6:
            continue
        dot = sign * (ax * dx + ay * dy) / (axis_xy_norm * dist_xy)
        if dot > math.cos(math.radians(tol_deg)):
            return True
    return False


def _check_pour(states: dict[str, Any]) -> bool:
    bottle = states.get(BOTTLE_ID)
    if bottle is None:
        return False

    tilt = tilt_deg_from_pose(bottle)
    if tilt < POUR_TILT_MIN_DEG:
        return False

    for glass_id in GLASS_IDS:
        glass = states.get(glass_id)
        if glass is not None and _faces_target(bottle, glass):
            return True
    return False


def check_success(
    states: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    step = int(ctx.get("_step", 0))

    if step == 0 and _check_fridge_open(states):
        ctx["_step"] = 1

    if step == 1 and _check_bottle_lifted(states):
        ctx["_step"] = 2

    if step == 2:
        if _check_pour(states):
            ctx["_step"] = 3

    if step == 3 and _check_bottle_in_fridge(states) and _check_bottle_upright(states):
        ctx["_step"] = 4

    if step == 4 and _check_fridge_closed(states):
        ctx["_step"] = 5

    return int(ctx.get("_step", 0)) >= 5
