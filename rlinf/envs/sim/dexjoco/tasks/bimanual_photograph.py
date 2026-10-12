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
_TABLE_HEIGHT_BOUNDS = np.asarray([0.0, 0.05], dtype=np.float64)
_LOGO_SAMPLING_BOUNDS_YZ = np.asarray([[-0.1, 1.22], [0.1, 1.38]], dtype=np.float64)
_CAMERA_SAMPLING_BOUNDS = np.asarray([[-0.3, 0.1], [-0.2, 0.2]], dtype=np.float64)
_TARGET_REGION_OFFSET = np.asarray([-0.4, 0.0, 0.0], dtype=np.float64)
_TARGET_REGION_RADIUS = 0.16
_TARGET_REGION_HALF_HEIGHT = 0.1
_TARGET_ANGLE_DEG = 10.0
_WORLD_POS_X = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


class BimanualPhotographTask(DexJoCoTaskBase):
    """Aim the camera at the target region and press the shutter."""

    xml_name = "arena_arm_hand_bimanual_photograph.xml"
    dual_arm = True
    task_description = "Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand."

    def _initialize_task(self) -> None:
        self._table_body_id = self._model.body("table").id
        self._table_z = float(self._model.body("table").pos[2])
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
        self._logo_geom_id = int(self._model.geom("photograph_logo_disk").id)
        self._target_region_geom_id = int(
            self._model.geom("photograph_target_region_visual").id
        )
        self._camera_axis_z_geom_id = int(self._model.geom("camera_axis_z").id)
        self._view_site_id = int(self._model.site("view_direction_site").id)
        self._camera_body_id = int(self._model.body("link_19").id)
        self._shutter_geom_id = int(self._model.geom("camera_body_top_cap").id)
        self._camera_geom_ids = self._collect_object_geom_ids(self._camera_body_id)
        self._logo_local_pos0 = self._model.geom_pos[self._logo_geom_id].copy()
        self._logo_local_quat0 = self._model.geom_quat[self._logo_geom_id].copy()
        self._logo_size0 = self._model.geom_size[self._logo_geom_id].copy()
        self._target_region_size0 = self._model.geom_size[
            self._target_region_geom_id
        ].copy()
        self._camera_body_pos0 = self._model.body_pos[self._camera_body_id].copy()
        self._camera_body_quat0 = self._model.body_quat[self._camera_body_id].copy()
        self._camera_body_z0 = float(self._camera_body_pos0[2])
        self._camera_mass0 = float(self._model.body_mass[self._camera_body_id])
        self._camera_mass_mul = (0.75, 1.25)
        self._hide_camera_screen_geom()

    def _collect_object_geom_ids(self, root_body_id: int) -> set:
        body_ids = {root_body_id}
        changed = True
        while changed:
            changed = False
            for bid in range(self._model.nbody):
                pid = int(self._model.body_parentid[bid])
                if pid in body_ids and bid not in body_ids:
                    body_ids.add(bid)
                    changed = True
        return {
            gid
            for gid in range(self._model.ngeom)
            if int(self._model.geom_bodyid[gid]) in body_ids
        }

    def _hide_camera_screen_geom(self):
        gid = int(self._model.geom("camera_live_screen").id)
        self._model.geom_matid[gid] = -1
        self._model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._model.geom_size[gid] = np.array([1e-9, 1e-9, 1e-9], dtype=np.float64)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        delta_h = np.float64(np.random.uniform(*_TABLE_HEIGHT_BOUNDS))
        table_pos = self._model.body("table").pos
        table_pos[2] = delta_h + self._table_z
        self._model.body("table").pos = table_pos
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + delta_h
        logo_yz = np.random.uniform(
            _LOGO_SAMPLING_BOUNDS_YZ[0], _LOGO_SAMPLING_BOUNDS_YZ[1]
        )
        logo_local_pos = self._logo_local_pos0.copy()
        logo_local_pos[1:] = logo_yz
        logo_local_quat = self._logo_local_quat0.copy()
        logo_ori_pose = np.concatenate([logo_local_pos, logo_local_quat]).astype(
            np.float64
        )
        self._model.geom_pos[self._logo_geom_id] = logo_ori_pose[0:3]
        self._model.geom_quat[self._logo_geom_id] = logo_ori_pose[3:7]
        self._model.geom_size[self._logo_geom_id] = self._logo_size0
        target_region_local_pos = logo_local_pos + _TARGET_REGION_OFFSET
        self._model.geom_pos[self._target_region_geom_id] = target_region_local_pos
        self._model.geom_size[self._target_region_geom_id, 0] = _TARGET_REGION_RADIUS
        self._model.geom_size[self._target_region_geom_id, 1] = (
            _TARGET_REGION_HALF_HEIGHT
        )
        self._model.geom_size[self._target_region_geom_id, 2] = (
            self._target_region_size0[2]
        )
        camera_xy = np.random.uniform(
            _CAMERA_SAMPLING_BOUNDS[0], _CAMERA_SAMPLING_BOUNDS[1]
        )
        camera_body_pos = self._camera_body_pos0.copy()
        camera_body_pos[:2] = camera_xy
        camera_body_pos[2] = self._camera_body_z0 + delta_h + 0.002
        camera_ori_pose = np.concatenate(
            [camera_body_pos, self._camera_body_quat0]
        ).astype(np.float64)
        self._data.jnt("camera_root").qpos = camera_ori_pose
        self._data.qpos[self._panda_right_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = self._ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = self._ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        self._data.mocap_quat[self._mocap_right_id] = tcp_quat_right
        self._data.mocap_quat[self._mocap_left_id] = tcp_quat_left
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            mass_mul = float(
                np.random.uniform(self._camera_mass_mul[0], self._camera_mass_mul[1])
            )
            self._model.body_mass[self._camera_body_id] = self._camera_mass0 * mass_mul
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {
            "success": False,
            "region_pass": False,
            "angle_pass": False,
            "shutter_pressed": False,
            "view_angle_deg": float(180.0),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        success, metrics = self._compute_success_metrics()
        obs = self._compute_observation()
        self.env_step += 1
        reward = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        info = {"success": bool(success)}
        info.update(metrics)
        return obs, reward, success, truncated, info

    def _compute_success_metrics(self) -> tuple[bool, dict[str, Any]]:
        region_center = self._data.geom_xpos[self._target_region_geom_id].copy()
        region_rot = self._data.geom_xmat[self._target_region_geom_id].reshape(3, 3)
        view_pos = self._data.site_xpos[self._view_site_id].copy()
        delta_local = region_rot.T @ (view_pos - region_center)
        radial_dist = float(np.linalg.norm(delta_local[:2]))
        region_height_dist = float(abs(delta_local[2]))
        region_pass = (
            radial_dist <= _TARGET_REGION_RADIUS
            and region_height_dist <= _TARGET_REGION_HALF_HEIGHT
        )
        axis_z_world = (
            self._data.geom_xmat[self._camera_axis_z_geom_id].reshape(3, 3)[:, 2].copy()
        )
        axis_z_norm = float(np.linalg.norm(axis_z_world))
        if axis_z_norm < 1e-8:
            axis_z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_z_world = axis_z_world / axis_z_norm
        cos_val = float(np.clip(np.dot(axis_z_world, _WORLD_POS_X), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        angle_pass = angle_deg <= _TARGET_ANGLE_DEG
        shutter_pressed = False
        shutter_contact_count = 0
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 == self._shutter_geom_id:
                if g2 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1
            elif g2 == self._shutter_geom_id:
                if g1 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1
        success = region_pass and angle_pass and shutter_pressed
        return bool(success), {
            "region_pass": bool(region_pass),
            "angle_pass": bool(angle_pass),
            "shutter_pressed": bool(shutter_pressed),
            "radial_dist": radial_dist,
            "region_height_dist": region_height_dist,
            "view_angle_deg": angle_deg,
            "shutter_contact_count": int(shutter_contact_count),
        }
