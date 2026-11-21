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

"""Absolute learned TCP targets to bounded LoopInfra local pose increments."""

import numpy as np

from .tianji_actions import _rotation_log
from .tianji_contract import rotation_from_6d, rotation_to_6d


def rotation_exp(vector):
    v = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _tool(observation, arm):
    return _tool_matrix(
        observation.raw["tianji_stage1"]["controller_tool_transforms"][arm]
    )


def _tool_matrix(value):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("invalid controller tool transform")
    if (
        not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )
        or np.linalg.det(transform[:3, :3]) < 0.999
    ):
        raise ValueError("invalid controller tool rotation")
    return transform


def target_deltas(physical, observation):
    """Recompute target error each step; never accumulate initial-state deltas."""
    result, scales = {}, {}
    translation_limit, rotation_limit = 0.0018, 0.035  # Older collectors.
    budget = observation.raw.get("robot_control", {}).get("policy_motion_limits")
    if budget is not None:
        if (
            not isinstance(budget, dict)
            or budget.get("protocol") != "loopinfra.policy_motion_limits.v1"
        ):
            raise ValueError("unsupported policy motion limits")
        translation_limit = budget.get("translation_lookahead_m")
        rotation_limit = budget.get("rotation_lookahead_rad")
        if (
            type(translation_limit) not in (int, float)
            or type(rotation_limit) not in (int, float)
            or not np.isfinite([translation_limit, rotation_limit]).all()
            or not 0 < translation_limit <= 0.0225
            or not 0 < rotation_limit <= np.pi
        ):
            raise ValueError("invalid policy motion limits")
    for arm, offset in (("A", 0), ("B", 10)):
        tool = _tool(observation, arm)
        flange_r = rotation_from_6d(physical[offset + 3 : offset + 9])
        flange_p = physical[offset : offset + 3] - flange_r @ [0, 0, 0.160]
        target_p = flange_p + flange_r @ tool[:3, 3]
        target_r = flange_r @ tool[:3, :3]
        before = observation.raw["sync"]["latest_arms"][arm]["eef"]
        r0 = np.asarray(before["rotation_matrix"], dtype=float)
        p0 = np.asarray(before["position_m"], dtype=float)
        dp = r0.T @ (target_p - p0)
        dr = _rotation_log(r0.T @ target_r)
        scale = min(
            1.0,
            translation_limit / max(np.linalg.norm(dp), 1e-12),
            rotation_limit / max(np.linalg.norm(dr), 1e-12),
        )
        g0 = observation.raw["collection_grippers"]["arms"][arm]["commanded_closure"]
        result[arm] = [*(dp * scale), *(dr * scale), float(physical[offset + 9] - g0)]
        scales[arm] = scale
    return result, scales


def controller_targets(physical, tools):
    """Absolute learned-TCP targets [N, 20] to the controller-TCP poses target_deltas tracks.

    rt fork: the LoopInfra server recomputes the bounded increment toward each
    pose from its own newest measured capture at that row's time.
    """
    physical = np.asarray(physical, dtype=float).reshape(-1, 20)
    result = [{} for _ in physical]
    for arm, offset in (("A", 0), ("B", 10)):
        tool = _tool_matrix(tools[arm])
        flange_r = rotation_from_6d(physical[:, offset + 3 : offset + 9])
        flange_p = physical[:, offset : offset + 3] - flange_r @ [0, 0, 0.160]
        positions = (flange_p + flange_r @ tool[:3, 3]).tolist()
        rotations = (flange_r @ tool[:3, :3]).tolist()
        for target, p, r, g in zip(result, positions, rotations, physical[:, offset + 9]):
            target[arm] = {"position_m": p, "rotation_matrix": r, "closure": float(g)}
    return result


def executed_absolute(deltas, observation):
    """Recover actual learned-TCP pose from the matched SDK terminal receipt."""
    result = []
    for arm in ("A", "B"):
        d = np.asarray(deltas[arm], dtype=float)
        if d.shape != (7,) or not np.isfinite(d).all():
            raise ValueError("invalid executed delta")
        tool = _tool(observation, arm)
        before = observation.raw["sync"]["latest_arms"][arm]["eef"]
        r0 = np.asarray(before["rotation_matrix"])
        controller_p = np.asarray(before["position_m"]) + r0 @ d[:3]
        controller_r = r0 @ rotation_exp(d[3:6])
        flange_r = controller_r @ tool[:3, :3].T
        learned_p = controller_p - flange_r @ tool[:3, 3] + flange_r @ [0, 0, 0.160]
        g = (
            observation.raw["collection_grippers"]["arms"][arm]["commanded_closure"]
            + d[6]
        )
        result.extend([*learned_p, *rotation_to_6d(flange_r), g])
    return np.asarray(result, dtype=np.float32)
