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

from rlinf.envs.sim.bench2dex.success import get_joint_position, world_axis_from_pose


def _pos(states: dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _has(states: dict[str, Any], *obj_ids: str) -> bool:
    return all(
        oid in states and states[oid].get("pose_world") is not None for oid in obj_ids
    )


def _dist_3d(states: dict[str, Any], a: str, b: str) -> float:
    pa = _pos(states, a)
    pb = _pos(states, b)
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def _upright(state: dict[str, Any], local_axis: str, tolerance_deg: float) -> bool:
    axis = world_axis_from_pose(state, local_axis)
    return axis[2] > math.cos(math.radians(tolerance_deg))


MUG_FAUCET_TARGET_X = 0.01
MUG_FAUCET_TARGET_Y = -0.04
MUG_FAUCET_RADIUS = 0.063

FAUCET_OPEN_THRESHOLD = 1.0
FAUCET_CLOSE_THRESHOLD = 0.52

SPOON_MUG_DIST_THRESHOLD = 0.18
# The spoon's long axis is local x, which points roughly up when it stands in the mug.
SPOON_UPRIGHT_TOLERANCE_DEG = 65.0

MUG_TRAY_XY_TOLERANCE = 0.20
MUG_TRAY_Z_MIN = 0.0
MUG_TRAY_Z_MAX = 0.035
MUG_TRAY_STATIC_THRESHOLD = 0.05

SPOON = "obj_031_spoon_1"
MUG = "obj_025_mug_1"
FAUCET = "obj_094_faucet_1"
TRAY = "obj_130_tray_1"


def _spoon_in_mug(states: dict[str, Any]) -> bool:
    if not _has(states, SPOON, MUG):
        return False
    near = _dist_3d(states, SPOON, MUG) < SPOON_MUG_DIST_THRESHOLD
    upright = _upright(states[SPOON], "x", SPOON_UPRIGHT_TOLERANCE_DEG)
    return near and upright


def _mug_under_faucet(states: dict[str, Any]) -> bool:
    if not _has(states, MUG):
        return False
    mx, my, _ = _pos(states, MUG)
    return (
        math.hypot(mx - MUG_FAUCET_TARGET_X, my - MUG_FAUCET_TARGET_Y)
        < MUG_FAUCET_RADIUS
    )


def _faucet_cycle_done(states: dict[str, Any], ctx: dict[str, Any]) -> bool:
    # The tap must open with the mug already under the spout and close before the
    # mug leaves; ctx latches a completed cycle.
    if not _has(states, FAUCET, MUG):
        return False
    joint_0 = get_joint_position(states[FAUCET], "joint_0", 0.0)
    mug_under = _mug_under_faucet(states)

    now_open = joint_0 >= FAUCET_OPEN_THRESHOLD
    prev_open = bool(ctx.get("faucet_prev_open", False))

    state = int(ctx.get("faucet_state", 0))
    if state == 0:
        if now_open and not prev_open:
            state = 1
            ctx["faucet_mug_left"] = not mug_under
    elif state == 1:
        if not mug_under:
            ctx["faucet_mug_left"] = True
        if joint_0 < FAUCET_CLOSE_THRESHOLD:
            if ctx.get("faucet_mug_left", False):
                state = 0
            else:
                state = 2
            ctx["faucet_mug_left"] = False

    ctx["faucet_prev_open"] = now_open
    ctx["faucet_state"] = state
    return state >= 2


def _mug_on_tray(states: dict[str, Any]) -> bool:
    if not _has(states, MUG, TRAY):
        return False
    mx, my, mz = _pos(states, MUG)
    tx, ty, tz = _pos(states, TRAY)
    xy_ok = math.hypot(mx - tx, my - ty) < MUG_TRAY_XY_TOLERANCE
    z_ok = MUG_TRAY_Z_MIN < (mz - tz) < MUG_TRAY_Z_MAX
    lv = states[MUG].get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))
    static_ok = speed < MUG_TRAY_STATIC_THRESHOLD
    return xy_ok and z_ok and static_ok


def check_success(
    states: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    if not _has(states, SPOON, MUG, FAUCET, TRAY):
        return False

    spoon_ok = _spoon_in_mug(states)
    mug_faucet_ok = _mug_under_faucet(states)
    faucet_ok = _faucet_cycle_done(states, ctx)
    mug_tray_ok = _mug_on_tray(states)

    step = int(ctx.get("step", 0))
    if step == 0 and spoon_ok:
        step = 1
    if step == 1 and mug_faucet_ok:
        step = 2
    if step == 2 and faucet_ok:
        step = 3
    if step == 3 and mug_tray_ok and spoon_ok:
        step = 4
    ctx["step"] = step
    return step >= 4
