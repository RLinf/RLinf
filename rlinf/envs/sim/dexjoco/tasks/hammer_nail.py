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

# This file contains code copied from the DexJoCo project: https://github.com/brave-eai/dexjoco.
# The original code is licensed under the MIT License.

import random
from typing import Any

import mujoco
import numpy as np

from rlinf.envs.sim.dexjoco.tasks.base import DexJoCoTaskBase

_MAX_EPISODE_STEPS = 1000
_HAMMER_SAMPLING_BOUNDS = np.asarray([[-0.25, -0.35], [-0.4, -0.5]])
_NAIL_SAMPLING_BOUNDS = np.asarray([[-0.1, 0.0], [0.0, 0.1]])
_HAMMER_YAW_PERTURB_BOUNDS = np.array([-10, 10])


class HammerNailTask(DexJoCoTaskBase):
    """Hammer the nail down to the required depth."""

    xml_name = "arena_arm_hand_hammer_nail.xml"
    task_description = "Use the hammer to drive the nail into the wooden board."
    # This scene names its dark-wood material dark_wood_mat.
    _TABLE_TEXTURES = (
        "table_bamboo",
        "table_blue-wood",
        "table_brass-ambra",
        "table_ceramic",
        "table_cream-plaster",
        "table_dark_wood_planks_2",
        "dark_wood_mat",
        "table_gray-plaster",
        "table_gray_wood_planks",
        "table_light-wood",
        "table_metal",
        "table_pink-plaster",
        "table_red-wood",
        "table_legs_metal",
        "table_steel-scratched",
        "table_walnut_wood_grain",
        "table_warm_wood_grain_2",
        "table_white-plaster",
        "table_wood_grain_1",
        "table_yellow-plaster",
    )

    def _initialize_task(
        self,
        success_depth: float = 0.04,
        impact_insert_step: float = 0.008,
        impact_vel_threshold: float = 0.02,
        max_insert_depth: float = 0.0726,
    ) -> None:
        self._success_depth = float(success_depth)
        self._impact_insert_step = float(impact_insert_step)
        self._impact_vel_threshold = float(impact_vel_threshold)
        self._nail_max_depth = float(max_insert_depth)
        nail_body_id = self._model.body("nail").id
        self._nail_mocap_id = int(self._model.body("nail").mocapid[0])
        if self._nail_mocap_id < 0:
            raise RuntimeError(
                f"Expected body 'nail' in {self.xml_name} to be mocap-enabled "
                f"(mocap='true'), got mocapid {self._nail_mocap_id}."
            )
        self._nail_init_pos = self._model.body_pos[nail_body_id].copy()
        self._nail_init_quat = self._model.body_quat[nail_body_id].copy()
        self._nail_depth = 0.0
        self._hammer_geom_ids = [
            self._model.geom(name).id for name in ("face", "head", "neck", "claw")
        ]
        self._nail_geom_ids = [
            self._model.geom(name).id for name in ("nail_head", "nail_shaft")
        ]
        self._face_gid = self._model.geom("face").id
        self._prev_face_z = None
        self._vz_buf: list[float] = []
        self._table_body_id = self._model.body("table").id
        self._table_body_z0 = float(self._model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1])
            for gid in self._table_leg_geom_ids
        }
        self._wood_body_z0 = float(self._model.body("wood").pos[2])
        self._hammer_body_z0 = float(self._model.body("hammer_body").pos[2])
        self._nail_body_z0 = float(self._model.body("nail").pos[2])
        self._hammer_body_id = self._model.body("hammer_body").id
        self._hammer_mass0 = float(self._model.body_mass[self._hammer_body_id])
        self._hammer_mass_mul = (0.75, 1.25)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        delta_h = np.float64(np.random.uniform(0.0, 0.05))
        self._model.body_pos[self._table_body_id, 2] = self._table_body_z0 + delta_h
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + delta_h
        self._model.body_pos[self._model.body("wood").id, 2] = (
            self._wood_body_z0 + delta_h
        )
        self._model.body_pos[self._model.body("hammer_body").id, 2] = (
            self._hammer_body_z0 + delta_h
        )
        self._data.mocap_pos[self._nail_mocap_id][2] = self._nail_body_z0 + delta_h
        self._nail_init_pos[2] = self._nail_body_z0 + delta_h
        hammer_xy = np.random.uniform(*_HAMMER_SAMPLING_BOUNDS)
        hammer_orig = np.array(
            self._data.jnt("hammer_joint").qpos[3:7], dtype=np.float64
        )
        hammer_yaw = np.deg2rad(np.random.uniform(*_HAMMER_YAW_PERTURB_BOUNDS))
        hqw, hqz = np.cos(hammer_yaw / 2), np.sin(hammer_yaw / 2)
        hw1, hx1, hy1, hz1 = hqw, 0, 0, hqz
        hw2, hx2, hy2, hz2 = hammer_orig
        hammer_q_new = np.array(
            [
                hw1 * hw2 - hx1 * hx2 - hy1 * hy2 - hz1 * hz2,
                hw1 * hx2 + hx1 * hw2 + hy1 * hz2 - hz1 * hy2,
                hw1 * hy2 - hx1 * hz2 + hy1 * hw2 + hz1 * hx2,
                hw1 * hz2 + hx1 * hy2 - hy1 * hx2 + hz1 * hw2,
            ]
        )
        hammer_q_new /= np.linalg.norm(hammer_q_new)
        hammer_ori_pose = np.concatenate(
            [hammer_xy, [self._hammer_body_z0 + delta_h], hammer_q_new]
        ).astype(np.float64)
        self._data.jnt("hammer_joint").qpos = hammer_ori_pose
        nail_xy = np.random.uniform(*_NAIL_SAMPLING_BOUNDS)
        self._nail_init_pos[:2] = nail_xy
        self._data.mocap_pos[self._nail_mocap_id][:2] = nail_xy
        self._data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = self._ALLEGRO_HOME
        self._nail_depth = 0.0
        self._data.mocap_pos[self._nail_mocap_id] = self._nail_init_pos
        self._data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat
        mujoco.mj_forward(self._model, self._data)
        tcp_pos = self._data.sensor("franka/flange_pos").data
        self._data.mocap_pos[self._mocap_id] = tcp_pos
        self.env_step = 0
        self._prev_face_z = float(self._data.geom_xpos[self._face_gid][2])
        self._vz_buf.clear()
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            mass = float(
                np.random.uniform(self._hammer_mass_mul[0], self._hammer_mass_mul[1])
            )
            self._model.body_mass[self._hammer_body_id] = self._hammer_mass0 * mass
        mujoco.mj_forward(self._model, self._data)
        if self.randomize:
            self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"success": False, "nail_depth": float(self._nail_depth)}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        hit = self._apply_hammer_nail_interaction()
        obs = self._compute_observation()
        self.env_step += 1
        success = self._compute_success()
        rew = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return (
            obs,
            rew,
            success,
            truncated,
            {
                "success": success,
                "nail_depth": float(self._nail_depth),
                "hammer_hit": hit,
            },
        )

    def _compute_success(self):
        return float(self._nail_depth) >= self._success_depth

    def _apply_hammer_nail_interaction(self) -> bool:
        hammer_set = set(self._hammer_geom_ids)
        nail_set = set(self._nail_geom_ids)
        hit = False
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in hammer_set and g2 in nail_set) or (
                g2 in hammer_set and g1 in nail_set
            ):
                hit = True
                break
        if not hit:
            return False
        preimpact_vz = min(self._vz_buf) if self._vz_buf else 0.0
        if preimpact_vz >= -self._impact_vel_threshold:
            return True
        depth = float(self._nail_depth)
        max_depth = float(self._nail_max_depth)
        scale = min(3.0, abs(preimpact_vz) / max(self._impact_vel_threshold, 1e-6))
        delta = self._impact_insert_step * scale
        new_depth = min(max_depth, depth + delta)
        if new_depth > depth:
            self._nail_depth = new_depth
            new_pos = self._nail_init_pos.copy()
            new_pos[2] = self._nail_init_pos[2] - new_depth
            self._data.mocap_pos[self._nail_mocap_id] = new_pos
            self._data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat
            mujoco.mj_forward(self._model, self._data)
        return True

    def _compute_observation(self):
        obs = super()._compute_observation()
        # The official hammer task reads hand joints from qpos, which leads the joint
        # sensors used by the base observation by one physics step.
        obs["state"][7:23] = self._data.qpos[self._allegro_dof_ids].astype(np.float32)
        return obs

    def _after_substep(self):
        face_z = float(self._data.geom_xpos[self._face_gid][2])
        # The official task divides per-substep motion by control_dt, and
        # impact_vel_threshold is tuned to that scale.
        self._vz_buf.append((face_z - self._prev_face_z) / self.control_dt)
        self._prev_face_z = face_z
        if len(self._vz_buf) > 12:
            self._vz_buf.pop(0)
