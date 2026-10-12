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

_MAX_EPISODE_STEPS = 1200
_MOUSE_XY_BOUNDS = np.asarray([[-0.2, 0.0], [-0.25, 0.05]])
_MOUSEPAD_OFFSET = np.asarray([0.0, -0.25, 0.0])
_MONITOR_XY_BOUNDS = np.asarray([[0.12, 0.3], [0.12, 0.3]])
_MOUSEPAD_Z_OFFSET = 0.002
_MOUSEPAD_Z_TOL = 0.05
_CLICK_THRESHOLD = 0.001
_YAW_PERTURB_BOUNDS = np.array([-10, 10])


class ClickMouseTask(DexJoCoTaskBase):
    """Click the mouse on its pad to turn the display blue."""

    xml_name = "arena_arm_hand_monitor_mouse.xml"
    task_description = (
        "Move the mouse to the purple mouse pad and click the left mouse button."
    )

    def _initialize_task(self) -> None:
        self._table_site_id = self._model.site("table_top").id
        self._allegro_default_pose = np.array(
            [
                0.47,
                0.3333,
                -0.0015,
                -0.227,
                0.47,
                0.3093,
                -0.0046,
                -0.227,
                0.1221,
                0.2868,
                -0.0147,
                -0.227,
                1.0577,
                -0.105,
                0.3653,
                -0.162,
            ],
            dtype=np.float64,
        )
        self._display_material_id = self._model.material("display_new-0_material-0").id
        self._display_rgba_default = self._model.mat_rgba[
            self._display_material_id
        ].copy()
        self._mousepad_geom_id = self._model.geom("mousepad").id
        self._mousepad_radius = float(self._model.geom_size[self._mousepad_geom_id][0])
        self._display_blue = False
        self._mouse_joint0_init = 0.0
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
        self._mousepad_body_id = self._model.body("mousepad").id
        self._display_root_z0 = float(self._data.jnt("display_root").qpos[2])
        self._success_trigger_target = 10
        self._success_trigger_count = 0
        self._mouse_body_id = int(self._model.body("link_3").id)
        self._mouse_mass0 = float(self._model.body_mass[self._mouse_body_id])
        self._mouse_mass_mul = (0.75, 1.25)

    def _set_display_blue(self):
        self._model.mat_rgba[self._display_material_id] = np.array(
            [0.0, 0.0, 1.0, 1.0], dtype=np.float32
        )
        self._display_blue = True

    def _restore_display_color(self):
        self._model.mat_rgba[self._display_material_id] = (
            self._display_rgba_default.copy()
        )
        self._display_blue = False

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
        self._data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = self._ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)
        tcp_pos = self._data.sensor("franka/flange_pos").data
        self._data.mocap_pos[self._mocap_id] = tcp_pos
        table_z = float(self._data.site_xpos[self._table_site_id][2])
        mouse_z = table_z + _MOUSEPAD_Z_OFFSET
        mouse_xy = np.float64(np.random.uniform(*_MOUSE_XY_BOUNDS))
        orig_quat = np.array(self._data.jnt("mouse_root").qpos[3:7], dtype=np.float64)
        yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        qw, qz = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        w1, x1, y1, z1 = qw, 0.0, 0.0, qz
        w2, x2, y2, z2 = orig_quat
        q_new = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
        q_new /= np.linalg.norm(q_new)
        mouse_ori_pose = np.concatenate([mouse_xy, [mouse_z], q_new]).astype(np.float64)
        self._data.jnt("mouse_root").qpos = mouse_ori_pose
        monitor_xy = np.random.uniform(*_MONITOR_XY_BOUNDS)
        display_qpos = self._data.jnt("display_root").qpos
        display_qpos[:3] = (
            monitor_xy[0],
            monitor_xy[1],
            self._display_root_z0 + delta_h,
        )
        display_qpos[3:7] = np.array([0.5, 0.5, -0.5, -0.5])
        self._model.body_pos[self._mousepad_body_id][:3] = (
            np.array([mouse_xy[0], mouse_xy[1], table_z]) + _MOUSEPAD_OFFSET
        )
        self._data.qpos[self._allegro_dof_ids] = self._allegro_default_pose
        mujoco.mj_forward(self._model, self._data)
        self._data.ctrl[self._allegro_ctrl_ids] = self._allegro_default_pose
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            mul_mass = float(
                np.random.uniform(self._mouse_mass_mul[0], self._mouse_mass_mul[1])
            )
            self._model.body_mass[self._mouse_body_id] = self._mouse_mass0 * mul_mass
        self._restore_display_color()
        self._mouse_joint0_init = float(self._data.sensor("mouse_joint0_pos").data[0])
        self._success_trigger_count = 0
        self.env_step = 0
        obs = self._compute_observation()
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        if (
            self._mouse_in_mousepad()
            and self._detect_click()
            and not self._display_blue
        ):
            self._set_display_blue()
        obs = self._compute_observation()
        self.env_step += 1
        success = self._compute_success()
        reward = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return obs, reward, success, truncated, {"success": success}

    def _detect_click(self) -> bool:
        joint0 = float(self._data.sensor("mouse_joint0_pos").data[0])
        return joint0 - self._mouse_joint0_init > _CLICK_THRESHOLD

    def _mouse_in_mousepad(self) -> bool:
        mouse_pos = self._data.sensor("mouse_pos").data
        pad_center = self._data.geom_xpos[self._mousepad_geom_id]
        dx, dy, dz = mouse_pos - pad_center
        in_xy = dx * dx + dy * dy <= self._mousepad_radius * self._mousepad_radius
        in_z = abs(dz) <= _MOUSEPAD_Z_TOL
        return in_xy and in_z

    def _compute_success(self) -> bool:
        trigger_active = self._mouse_in_mousepad() and self._display_blue
        if trigger_active:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0
        return self._success_trigger_count >= self._success_trigger_target
