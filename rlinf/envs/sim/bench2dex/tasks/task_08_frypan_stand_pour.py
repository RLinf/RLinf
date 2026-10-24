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

from rlinf.envs.sim.bench2dex.success import quat_rotate_vector, world_axis_from_pose

FRYPAN_ID = "obj_223_chefmate_frypan_1"
STAND_ID = "obj_333_portable_stove_6"
BREAD_ID = "obj_182_bread_2"
SOY_ID = "obj_172_soy_sauce_1"
OIL_ID = "obj_146_olive_oil_1"

POUR_TILT_MIN_DEG = 50.0
UPRIGHT_MAX_TILT_DEG = 15.0

SOY_MOUTH_OFFSET = (0.0, 0.150, 0.0)
OIL_MOUTH_OFFSET = (0.0, 0.300, 0.0)
STATIC_LIN_THRESHOLD = 0.05
# Measured against the kinematic stove on the table, so a bottle still held aloft fails.
BOTTLE_ON_TABLE_DZ_MAX = 0.05

# The frypan origin is its handle; the pan centre lies 0.27 m along local z.
FRYPAN_CENTER_OFFSET_Z = 0.27
FRYPAN_RADIUS = 0.125


def _tilt_y(state: dict[str, Any]) -> float:
    # The bottles spawn with local y along world +z, so tilt is taken on y rather
    # than with tilt_deg_from_pose (local z).
    ax, ay, az = world_axis_from_pose(state, "y")
    return math.degrees(math.acos(max(-1.0, min(1.0, az))))


def _pos(states: dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _mouth_pos(states: dict[str, Any], obj_id: str, mouth_offset: tuple) -> tuple:
    return quat_rotate_vector(states[obj_id]["pose_world"], mouth_offset)


def _bottle_pouring(states: dict[str, Any], obj_id: str, mouth_offset: tuple) -> bool:
    if obj_id not in states or FRYPAN_ID not in states:
        return False
    if _tilt_y(states[obj_id]) < POUR_TILT_MIN_DEG:
        return False
    mx, my, _mz = _mouth_pos(states, obj_id, mouth_offset)
    cx, cy = _frypan_center(states)
    return math.hypot(mx - cx, my - cy) < FRYPAN_RADIUS


def _frypan_center(states: dict[str, Any]) -> tuple[float, float]:
    fx, fy, _fz = _pos(states, FRYPAN_ID)
    zx, zy, _zz = world_axis_from_pose(states[FRYPAN_ID], "z")
    return fx + zx * FRYPAN_CENTER_OFFSET_Z, fy + zy * FRYPAN_CENTER_OFFSET_Z


def _stove_center(states: dict[str, Any]) -> tuple[float, float]:
    ox, oy, _oz = _pos(states, STAND_ID)
    dx, dy, _dz = quat_rotate_vector(
        states[STAND_ID]["pose_world"], [-1.42e-05, 0.039681, 0.0]
    )
    return ox + dx, oy + dy


def _frypan_on_stove(states: dict[str, Any]) -> bool:
    if FRYPAN_ID not in states or STAND_ID not in states:
        return False
    fx, fy = _frypan_center(states)
    sx, sy = _stove_center(states)
    return math.hypot(fx - sx, fy - sy) < 0.10


def _bread_in_pan(states: dict[str, Any]) -> bool:
    if BREAD_ID not in states or FRYPAN_ID not in states:
        return False
    bx, by, _bz = _pos(states, BREAD_ID)
    cx, cy = _frypan_center(states)
    return math.hypot(bx - cx, by - cy) < FRYPAN_RADIUS


def _bottle_upright(states: dict[str, Any], obj_id: str) -> bool:
    return obj_id in states and _tilt_y(states[obj_id]) < UPRIGHT_MAX_TILT_DEG


def _bottle_static(states: dict[str, Any], obj_id: str) -> bool:
    state = states.get(obj_id)
    if state is None:
        return False
    lv = state.get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))
    return speed < STATIC_LIN_THRESHOLD


def _bottle_on_table(states: dict[str, Any], obj_id: str) -> bool:
    if obj_id not in states or STAND_ID not in states:
        return False
    _ox, _oy, oz = _pos(states, obj_id)
    _sx, _sy, sz = _pos(states, STAND_ID)
    return (oz - sz) < BOTTLE_ON_TABLE_DZ_MAX


def check_success(
    states: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    step = int(ctx.get("_t08_step", 0))

    if step == 0:
        if _frypan_on_stove(states):
            ctx["_t08_step"] = 1

    if step == 1:
        if _bottle_pouring(states, SOY_ID, SOY_MOUTH_OFFSET):
            ctx["_t08_step"] = 2

    if step == 2:
        if _bottle_pouring(states, OIL_ID, OIL_MOUTH_OFFSET):
            ctx["_t08_step"] = 3

    if step == 3:
        soy_ok = (
            _bottle_upright(states, SOY_ID)
            and _bottle_static(states, SOY_ID)
            and _bottle_on_table(states, SOY_ID)
        )
        oil_ok = (
            _bottle_upright(states, OIL_ID)
            and _bottle_static(states, OIL_ID)
            and _bottle_on_table(states, OIL_ID)
        )
        if soy_ok and oil_ok and _bread_in_pan(states):
            ctx["_t08_step"] = 4

    return int(ctx.get("_t08_step", 0)) >= 4
