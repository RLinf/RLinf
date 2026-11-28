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

"""Track actual stair passage from terrain geometry and terminal state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MethodType

import numpy as np
import torch

VERSION = "square_obstacle_v1"


@dataclass(frozen=True)
class Obstacle:
    task: str
    row: int
    col: int
    inner_radius: float
    outer_radius: float
    tile_radius: float
    landing_height: float
    rise: float
    stair_rises: int
    profile: tuple

    def to_dict(self):
        def native(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {k: native(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [native(v) for v in value]
            return value

        return native(asdict(self))


def _profile(mesh, axis, sign):
    """Intersect upward horizontal triangles with a centre-line half ray."""
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.face_normals)
    intervals = []
    other = 1 - axis
    for tri in triangles[normals[:, 2] > 1 - 1e-6]:
        if np.ptp(tri[:, 2]) > 1e-6:
            continue
        points = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if abs(a[other]) < 1e-7:
                points.append(sign * a[axis])
            if a[other] * b[other] < 0:
                alpha = -a[other] / (b[other] - a[other])
                points.append(sign * (a[axis] + alpha * (b[axis] - a[axis])))
        if len(points) >= 2:
            lo, hi = max(0.0, min(points)), max(points)
            if hi - lo > 1e-6:
                intervals.append((lo, hi, float(tri[0, 2])))
    # Resolve overlaps by the uppermost surface, then merge adjacent coplanar pieces.
    edges = sorted({round(v, 6) for lo, hi, _ in intervals for v in (lo, hi)})
    bands = []
    for lo, hi in zip(edges, edges[1:]):
        mid = (lo + hi) / 2
        heights = [
            z for left, right, z in intervals if left - 1e-7 <= mid <= right + 1e-7
        ]
        if not heights:
            continue
        z = max(heights)
        if bands and abs(bands[-1][1] - lo) < 1e-5 and abs(bands[-1][2] - z) < 1e-5:
            bands[-1] = (bands[-1][0], hi, z)
        else:
            bands.append((lo, hi, z))
    return bands


def obstacle_from_mesh(mesh, origin, task, row, col, generator_function):
    """Read supported square rings from actual local mesh vertices, not difficulty guesses."""
    expected = {"B2": "inverted_pyramid_stairs_terrain", "C2": "gap_terrain"}
    if generator_function != expected.get(task):
        raise ValueError(f"Unsupported {task} geometry: {generator_function}")
    profiles = [
        _profile(mesh, axis, sign) for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1))
    ]
    bands = profiles[0]
    if not bands or bands[0][0] > 1e-5:
        raise ValueError("No central landing surface in generated mesh")
    for profile in profiles[1:]:
        if np.shape(profile) != np.shape(bands) or not np.allclose(
            profile, bands, atol=1e-5, rtol=0
        ):
            raise ValueError("Terrain is not the supported symmetric square obstacle")
    tile_radius = float(bands[-1][1])
    if task == "B2":
        if len(bands) < 2 or any(
            b[2] <= a[2] + 1e-5 or abs(a[1] - b[0]) > 1e-5
            for a, b in zip(bands, bands[1:])
        ):
            raise ValueError(
                "B2 mesh does not contain a continuous ascending staircase"
            )
        inner, outer = bands[0][1], bands[-1][0]
    else:
        if len(bands) == 1:
            # A genuinely zero-width gap is a measured no-obstacle case, never a success.
            inner = outer = bands[0][1]
        elif (
            len(bands) == 2
            and bands[1][0] > bands[0][1] + 1e-6
            and abs(bands[0][2] - bands[1][2]) < 1e-5
        ):
            inner, outer = bands[0][1], bands[1][0]
        else:
            raise ValueError(
                "C2 mesh does not contain one central platform and one gap ring"
            )
    return Obstacle(
        task,
        row,
        col,
        inner,
        outer,
        tile_radius,
        bands[-1][2] - float(origin[2]),
        bands[-1][2] - bands[0][2],
        len(bands) - 1 if task == "B2" else 0,
        tuple(tuple(float(v) for v in band) for band in bands),
    )


class PassageTracker:
    """Vectorized traversal evidence, matched to upstream episode ordering at finish()."""

    def __init__(self, obstacles, num_envs, step_dt, device="cpu"):
        self.obstacles = obstacles
        self.device = torch.device(device)
        self.n = num_envs
        self.step_dt = float(step_dt)
        self.hold_steps = math.ceil(0.2 / self.step_dt)
        self.keys = sorted(obstacles)
        self.lookup = {key: i for i, key in enumerate(self.keys)}
        self.table = torch.tensor(
            [
                [o.inner_radius, o.outer_radius, o.tile_radius, o.landing_height]
                for o in (obstacles[k] for k in self.keys)
            ],
            device=device,
        )
        self.geometry_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.origin = torch.zeros((num_envs, 3), device=device)
        self.start = torch.zeros_like(self.origin)
        self.last = torch.zeros_like(self.origin)
        self.end = torch.zeros_like(self.origin)
        self.pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.eligible = torch.zeros_like(self.pending)
        self.nonzero_obstacle = torch.zeros_like(self.pending)
        self.invalid_jump = torch.zeros_like(self.pending)
        self.left_tile = torch.zeros_like(self.pending)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.furthest = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros((num_envs, 4), dtype=torch.long, device=device)
        self.contacts_seen = torch.zeros(
            (num_envs, 4, 4), dtype=torch.bool, device=device
        )
        self.passed_at = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.exit_side = torch.full_like(self.passed_at, -1)
        self.episode_index = torch.zeros_like(self.passed_at)

    def begin(self, root, feet, origins, levels, types):
        ids = self.pending.nonzero().flatten()
        if not ids.numel():
            return
        indices = [
            self.lookup[(int(row), int(col))]
            for row, col in zip(levels[ids].tolist(), types[ids].tolist())
        ]
        self.geometry_index[ids] = torch.tensor(indices, device=self.device)
        self.origin[ids] = origins[ids]
        self.start[ids] = root[ids] - origins[ids]
        self.last[ids] = self.start[ids]
        inner, outer = self.table[self.geometry_index[ids], :2].unbind(-1)
        self.nonzero_obstacle[ids] = outer > inner + 1e-5
        foot_radius = (
            (feet[ids] - origins[ids, None, :])[:, :, :2].abs().amax(dim=(1, 2))
        )
        self.eligible[ids] = (foot_radius < inner - 0.02) & (
            self.start[ids, :2].abs().amax(-1) < inner - 0.02
        )
        self.pending[ids] = False

    def observe(self, root, feet, contact_force, upright):
        if self.pending.any():
            raise RuntimeError("Episode start missing before passage observation")
        root, feet = root - self.origin, feet - self.origin[:, None, :]
        self.steps += 1
        self.end.copy_(root)
        self.invalid_jump |= torch.linalg.vector_norm(root - self.last, dim=-1) > max(
            0.5, 10 * self.step_dt
        )
        self.last.copy_(root)
        _, outer, limit, height = self.table[self.geometry_index].unbind(-1)
        radius = root[:, :2].abs().amax(-1)
        self.furthest = torch.maximum(self.furthest, radius)
        # Leaving the source tile before verified landing cannot count as crossing it.
        self.left_tile |= (self.passed_at < 0) & (radius >= limit - 0.02)
        foot_xy = feet[:, :, :2]
        projections = torch.stack(
            (foot_xy[:, :, 0], -foot_xy[:, :, 0], foot_xy[:, :, 1], -foot_xy[:, :, 1]),
            dim=1,
        )
        base_projection = torch.stack(
            (root[:, 0], -root[:, 0], root[:, 1], -root[:, 1]), dim=1
        )
        wholly_beyond = (projections > outer[:, None, None] + 0.02).all(-1) & (
            base_projection > outer[:, None] + 0.02
        )
        in_tile = (foot_xy.abs() < limit[:, None, None] - 0.02).all(dim=(1, 2))
        near_landing = ((feet[:, :, 2] - height[:, None]).abs() < 0.12).all(-1)
        contact = contact_force > 1.0
        standing = (
            (upright > 0.7)
            & (root[:, 2] > height + 0.15)
            & near_landing
            & (contact.sum(-1) >= 2)
        )
        valid = (
            self.eligible
            & self.nonzero_obstacle
            & ~self.invalid_jump
            & ~self.left_tile
            & in_tile
            & standing
        )
        good = wholly_beyond & valid[:, None]
        # B2's command and reset yaw point uphill along world +x. C2 can exit any side.
        if self.obstacles[self.keys[0]].task == "B2":
            good[:, 1:] = False
        self.hold = torch.where(good, self.hold + 1, 0)
        self.contacts_seen = torch.where(
            good[:, :, None], self.contacts_seen | contact[:, None, :], False
        )
        landed = (self.hold >= self.hold_steps) & self.contacts_seen.all(-1)
        newly = (self.passed_at < 0) & landed.any(-1)
        self.passed_at[newly] = self.steps[newly]
        self.exit_side[newly] = landed[newly].long().argmax(-1)

    def finish(self, done_ids, benchmark_rows):
        ids = done_ids.tolist()
        if len(ids) != len(benchmark_rows):
            raise RuntimeError("Tracking/passage episode alignment mismatch")
        for idx, row in zip(ids, benchmark_rows):
            obstacle = self.obstacles[self.keys[int(self.geometry_index[idx])]]
            nonzero = bool(self.nonzero_obstacle[idx])
            passed = (
                bool(
                    self.passed_at[idx] >= 0
                    and not self.invalid_jump[idx]
                    and row["termination_reason"] != "failure"
                )
                if nonzero
                else None
            )
            row.update(
                env_id=idx,
                episode_index=int(self.episode_index[idx]),
                physical_obstacle_passage=passed,
                joint_tracking_and_passage=(
                    bool(row["episode_success"] and passed) if nonzero else None
                ),
                passage_evidence={
                    "metric_version": VERSION,
                    "geometry_key": f"{obstacle.row},{obstacle.col}",
                    "eligible_start": bool(self.eligible[idx]),
                    "nonzero_obstacle": nonzero,
                    "unexpected_position_jump": bool(self.invalid_jump[idx]),
                    "left_tile_before_landing": bool(self.left_tile[idx]),
                    "start_root_relative_m": self.start[idx].tolist(),
                    "end_root_relative_m": self.end[idx].tolist(),
                    "maximum_radius_m": float(self.furthest[idx]),
                    "landing_step": int(self.passed_at[idx]),
                    "exit_side": ("+x", "-x", "+y", "-y")[int(self.exit_side[idx])]
                    if self.exit_side[idx] >= 0
                    else None,
                    "required_outer_radius_m": obstacle.outer_radius,
                    "required_rise_m": obstacle.rise,
                    "stair_rises": obstacle.stair_rises,
                    "sustained_landing_s": self.hold_steps * self.step_dt,
                },
            )
            self.episode_index[idx] += 1
        self.pending[done_ids] = True
        for buffer in (
            self.steps,
            self.furthest,
            self.hold,
            self.contacts_seen,
            self.invalid_jump,
            self.left_tile,
        ):
            buffer[done_ids] = 0
        self.passed_at[done_ids] = -1
        self.exit_side[done_ids] = -1
        return benchmark_rows


def summarize_passage(episodes):
    keys = set()
    for row in episodes:
        if "physical_obstacle_passage" not in row:
            continue
        evidence = row["passage_evidence"]
        key = (row["env_id"], row["episode_index"])
        if key in keys or evidence["metric_version"] != VERSION:
            raise ValueError(
                "Duplicate episode identity or unknown passage metric version"
            )
        keys.add(key)
        passed = row["physical_obstacle_passage"]
        if passed is None:
            if evidence["nonzero_obstacle"]:
                raise ValueError("Missing physical result for a nonzero obstacle")
            continue
        if type(passed) is not bool or row["joint_tracking_and_passage"] != (
            row["episode_success"] and passed
        ):
            raise ValueError("Inconsistent physical/joint episode success")
        if passed and (
            row["termination_reason"] == "failure"
            or not evidence["eligible_start"]
            or evidence["unexpected_position_jump"]
            or evidence["left_tile_before_landing"]
            or evidence["landing_step"] < 0
        ):
            raise ValueError("Passage success lacks valid landing evidence")
    measured = [e for e in episodes if e.get("physical_obstacle_passage") is not None]
    if not episodes or len(measured) != len(episodes):
        return {
            "physical_obstacle_passage_rate": None,
            "joint_tracking_and_passage_success_rate": None,
            "physical_evaluation_status": "incomplete_no_obstacle_or_missing_evidence",
            "physical_measured_episodes": len(measured),
            "required_obstacle_metrics_complete": False,
        }
    n = len(episodes)
    eligible = sum(e["passage_evidence"]["eligible_start"] for e in episodes)
    successes = sum(e["physical_obstacle_passage"] for e in episodes)
    return {
        "physical_obstacle_passage_rate": sum(
            e["physical_obstacle_passage"] for e in episodes
        )
        / n,
        "joint_tracking_and_passage_success_rate": sum(
            e["joint_tracking_and_passage"] for e in episodes
        )
        / n,
        "physical_evaluation_status": "measured",
        "physical_measured_episodes": n,
        "physical_eligible_start_episodes": eligible,
        "physical_passage_rate_given_eligible_start": successes / eligible
        if eligible
        else None,
        "physical_passage_successes": successes,
        "metric_version": VERSION,
        "required_obstacle_metrics_complete": True,
    }


def capture_geometry(cfg, task, captured):
    """Observe meshes before their placement transform; do not regenerate or reseed."""
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
        raise RuntimeError("Passage evaluation requires the original generated terrain")
    base = generator.class_type

    class CapturedGenerator(base):
        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            captured[(row, col)] = obstacle_from_mesh(
                mesh, origin, task, row, col, sub_terrain_cfg.function.__name__
            )
            return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    generator.class_type = CapturedGenerator


class PassageRuntime:
    """Read terminal kinematics before Isaac Lab's automatic reset, without extra sensing."""

    def __init__(self, env, captured):
        import torch

        self.env, self.terminal, self.in_step = env, None, False
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.foot_ids, self.foot_names = self.robot.find_bodies(
            ".*_foot", preserve_order=True
        )
        self.contact_ids, names = self.sensor.find_bodies(
            self.foot_names, preserve_order=True
        )
        if len(self.foot_ids) != 4 or names != self.foot_names:
            raise RuntimeError(
                f"Cannot align four robot feet with contact sensor: {self.foot_names}, {names}"
            )
        self.tracker = PassageTracker(captured, env.num_envs, env.step_dt, env.device)
        self.old_step, self.old_reset = env.step, env._reset_idx

        def reset(inner, ids):
            if self.in_step and len(ids):
                if self.terminal is not None:
                    raise RuntimeError(
                        "Unexpected multiple autoresets in one control step"
                    )
                self.terminal = (
                    torch.as_tensor(ids, device=env.device).clone(),
                    self.snapshot(),
                )
            return self.old_reset(ids)

        def step(inner, action):
            before = self.snapshot()
            terrain = env.scene.terrain
            self.tracker.begin(
                before["root"],
                before["feet"],
                terrain.env_origins,
                terrain.terrain_levels,
                terrain.terrain_types,
            )
            self.terminal = None
            self.in_step = True
            try:
                result = self.old_step(action)
            finally:
                self.in_step = False
            after = self.snapshot()
            done = (result[2] | result[3]).nonzero().flatten()
            if done.numel():
                if self.terminal is None or not torch.equal(done, self.terminal[0]):
                    raise RuntimeError("Terminal capture and original done IDs differ")
                for name in after:
                    after[name][done] = self.terminal[1][name][done]
            self.tracker.observe(
                after["root"], after["feet"], after["force"], after["upright"]
            )
            return result

        env._reset_idx = MethodType(reset, env)
        env.step = MethodType(step, env)

    def snapshot(self):
        import torch

        return {
            "root": self.robot.data.root_pos_w.clone(),
            "feet": self.robot.data.body_pos_w[:, self.foot_ids].clone(),
            "force": torch.linalg.vector_norm(
                self.sensor.data.net_forces_w[:, self.contact_ids], dim=-1
            ),
            "upright": -self.robot.data.projected_gravity_b[:, 2].clone(),
        }

    def close(self):
        self.env.step, self.env._reset_idx = self.old_step, self.old_reset
