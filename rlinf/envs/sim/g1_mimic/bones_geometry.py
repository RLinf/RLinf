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

"""Geom-to-floor distances for the Appendix E feasibility gates.

Poses the mjlab G1 at each reference frame and measures, with MuJoCo's
``mj_geomDistance``, the signed distance from every collision geom to a
ground plane at z = 0. Negative values are penetration. ``zmin`` is the
minimum over all collision geoms; ``foot_clearance`` is the minimum over
each ankle-roll link's geoms. Visual meshes carry no collision flags and
are not measured.
"""

from __future__ import annotations

import numpy as np
import torch

from rlinf.envs.sim.g1_mimic.constants import JOINT_POS, ROOT_QUAT

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Distances beyond this are reported as this value. Large enough for any
# pose a G1 clip reaches; elevated support needs only > 0.20 m.
DIST_MAX = 5.0


class G1FloorDistance:
    """One compiled G1 plus a floor plane, reused across frames and clips."""

    def __init__(self):
        import mujoco
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

        self._mujoco = mujoco
        spec = get_spec()
        spec.worldbody.add_geom(
            name="supermimic_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.1],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.floor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "supermimic_floor"
        )
        collides = (self.model.geom_contype | self.model.geom_conaffinity) != 0
        robot = self.model.geom_bodyid != 0
        self.geoms = np.nonzero(collides & robot)[0]
        if self.geoms.size == 0:
            raise RuntimeError("the G1 model has no collision geoms to measure")
        foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ]
        self.foot_geoms = [
            [int(g) for g in self.geoms if self.model.geom_bodyid[g] == body]
            for body in foot_ids
        ]
        if not all(self.foot_geoms):
            counts = dict(zip(FOOT_BODIES, map(len, self.foot_geoms)))
            raise RuntimeError(
                f"expected collision geoms on both ankle-roll links, found {counts}"
            )
        self._fromto = np.zeros(6)

    def _distance(self, geom: int) -> float:
        return float(
            self._mujoco.mj_geomDistance(
                self.model, self.data, self.floor, int(geom), DIST_MAX, self._fromto
            )
        )

    def frame(self, state: np.ndarray) -> tuple[float, float, float]:
        """``(zmin, left_clearance, right_clearance)`` for one 71-d state."""
        qpos = self.data.qpos
        qpos[0:3] = state[0:3]
        qpos[3:7] = state[ROOT_QUAT]
        qpos[7:36] = state[JOINT_POS]
        self._mujoco.mj_kinematics(self.model, self.data)
        distances = {int(g): self._distance(g) for g in self.geoms}
        zmin = min(distances.values())
        feet = [min(distances[g] for g in group) for group in self.foot_geoms]
        return zmin, feet[0], feet[1]

    def clip(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``zmin [T]`` and ``foot_clearance [T, 2]`` for a ``[T, 71]`` clip."""
        frames = state.detach().cpu().double().numpy()
        out = np.array([self.frame(row) for row in frames], dtype=np.float32)
        return torch.from_numpy(out[:, 0].copy()), torch.from_numpy(out[:, 1:].copy())
