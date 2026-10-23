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
_STAND_SAMPLING_BOUNDS = np.array([[-0.35, 0.05], [-0.3, 0.1]])


class BimanualUnlockIpadTask(DexJoCoTaskBase):
    """Unlock the tablet by entering the password in order."""

    xml_name = "arena_arm_hand_ipad.xml"
    dual_arm = True
    task_description = "Grasp the iPad and enter the password 123 to unlock the device."

    def _initialize_task(self, password: list[int] | None = None) -> None:
        self._table_body_id = self._model.body("table").id
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
        self._button_geom_ids = []
        self._button_geom_to_digit = {}
        self._button_visual_geom_ids = []
        self._button_visual_orig_rgba = {}
        self._digit_geom_ids_by_digit = {}
        for gid in range(self._model.ngeom):
            gname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and gname.startswith("btn") and gname.endswith("_cyl"):
                self._button_geom_ids.append(gid)
                num = int(gname.replace("btn", "").replace("_cyl", ""))
                self._button_geom_to_digit[gid] = num
            if gname and gname.startswith("btn") and gname.endswith("_digit"):
                num = int(gname.replace("btn", "").replace("_digit", ""))
                self._digit_geom_ids_by_digit[num] = gid
            matid = int(self._model.geom_matid[gid])
            mname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
            if mname in ("mat_digit",) or (mname and mname.startswith("mat_digit_")):
                self._button_visual_geom_ids.append(gid)
                self._button_visual_orig_rgba[gid] = self._model.geom_rgba[gid].copy()
        self._mat_digit_ids = {}
        self._mat_digit_light_ids = {}
        for d in range(10):
            self._mat_digit_ids[d] = int(self._model.material(f"mat_digit_{d}").id)
            self._mat_digit_light_ids[d] = int(
                self._model.material(f"mat_digit_{d}_light").id
            )
        if password is None:
            self._unlock_sequence = [1, 2, 3]
        else:
            assert all(0 <= p and p <= 9 for p in password), (
                f"Expected password digits in [0, 9], got {password}."
            )
            self._unlock_sequence = list(password)
        self._unlock_index = 0
        self._unlock_failed = False
        self._pressed_last = set()
        self._buttons_hidden = False
        self._screen_unlocked = False
        self._screen_geom_id = int(self._model.geom("ipad_screen").id)
        self._mat_screen_locked_id = int(self._model.material("Material_black").id)
        self._mat_screen_unlocked_id = int(
            self._model.material("mat_screen_unlocked").id
        )
        self._success_trigger_count = 0
        self._success_trigger_target = 10
        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._table_z = self._model.body("table").pos[2].copy()
        self._ipad_body_z0 = self._model.body("ipad_body").pos[2].copy()
        self._stand_body_z0 = self._model.body("ipad_stand").pos[2].copy()
        self._finger_geom_ids = {
            self._model.geom("fingertip_ff_right").id,
            self._model.geom("fingertip_mf_right").id,
            self._model.geom("fingertip_rf_right").id,
            self._model.geom("thumbtip_right").id,
        }
        self._ipad_body_id = self._model.body("ipad_body").id
        self._ipad_mass0 = float(self._model.body_mass[self._ipad_body_id])
        self._ipad_mass_mul = (0.75, 1.25)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        delta_h = np.float64(np.random.uniform(0.0, 0.05))
        table_ori_pos = self._model.body("table").pos
        table_ori_pos[2] = delta_h + self._table_z
        self._model.body("table").pos = table_ori_pos
        dh = float(delta_h)
        for lgid in self._table_leg_geom_ids:
            base_center_z = float(self._model_geom_pos0[lgid, 2])
            base_half_len = self._table_leg_half_len0[lgid]
            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh
            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len
        stand_xy = np.random.uniform(*_STAND_SAMPLING_BOUNDS)
        stand_body_id = self._model.body("ipad_stand").id
        stand_pos = self._model.body("ipad_stand").pos
        stand_quat = self._model.body("ipad_stand").quat
        ori_stand_xy = np.array(self._model.body_pos[stand_body_id][:2])
        delta_xy = stand_xy - ori_stand_xy
        ipad_body_id = self._model.body("ipad_body").id
        ipad_pos = self._model.body_pos[ipad_body_id]
        ipad_quat = self._model.body("ipad_body").quat
        ipad_pos[:2] += delta_xy
        ipad_pos[2] = self._ipad_body_z0 + delta_h
        stand_pos[:2] = stand_xy
        stand_pos[2] = self._stand_body_z0 + delta_h
        ipad_ori_pose = np.array(list(ipad_pos) + list(ipad_quat), dtype=np.float64)
        stand_ori_pose = np.array(list(stand_pos) + list(stand_quat), dtype=np.float64)
        self._model.body_pos[stand_body_id] = stand_ori_pose[:3]
        self._data.jnt("ipad_freejoint").qpos[:3] = ipad_ori_pose[:3]
        self._data.qpos[self._panda_right_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = self._ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = self._ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            mass = float(
                np.random.uniform(self._ipad_mass_mul[0], self._ipad_mass_mul[1])
            )
            self._model.body_mass[self._ipad_body_id] = self._ipad_mass0 * mass
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._set_button_materials(set())
        self._unlock_index = 0
        self._unlock_failed = False
        self._success_trigger_count = 0
        self._pressed_last = set()
        self._buttons_hidden = False
        self._set_screen_locked()
        self._show_buttons()
        self._screen_unlocked = False
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        pressed_digits = self._update_button_visuals()
        obs = self._compute_observation()
        self.env_step += 1
        success = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = success or self._unlock_failed
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return (
            obs,
            rew,
            terminated,
            truncated,
            {"success": success, "pressed_digits": pressed_digits},
        )

    def _set_button_materials(self, pressed_ids: set):
        for gid in self._button_geom_ids:
            digit = self._button_geom_to_digit[gid]
            digit_gid = self._digit_geom_ids_by_digit.get(digit, None)
            if gid in pressed_ids:
                mat_light = self._mat_digit_light_ids[digit]
                self._model.geom_matid[gid] = mat_light
                if digit_gid is not None:
                    self._model.geom_matid[digit_gid] = mat_light
            else:
                mat_off = self._mat_digit_ids[digit]
                self._model.geom_matid[gid] = mat_off
                if digit_gid is not None:
                    self._model.geom_matid[digit_gid] = mat_off

    def _update_button_visuals(self):
        if self._buttons_hidden:
            return
        button_set = set(self._button_geom_ids)
        pressed = set()
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 in button_set and g2 in self._finger_geom_ids:
                pressed.add(g1)
            elif g2 in button_set and g1 in self._finger_geom_ids:
                pressed.add(g2)
        self._set_button_materials(pressed)
        pressed_digits = self._update_unlock_sequence(pressed)
        return pressed_digits

    def _update_unlock_sequence(self, pressed: set):
        newly_pressed = pressed - self._pressed_last
        pressed_digits = []
        for gid in sorted(newly_pressed):
            digit = self._button_geom_to_digit[gid]
            pressed_digits.append(digit)
            expected = self._unlock_sequence[self._unlock_index]
            if digit == expected:
                self._unlock_index += 1
                if self._unlock_index >= len(self._unlock_sequence):
                    self._set_screen_unlocked()
                    self._hide_buttons()
                    self._screen_unlocked = True
            else:
                self._unlock_failed = True
                self._unlock_index = 0
                self._set_screen_locked()
                self._screen_unlocked = False
        self._pressed_last = pressed
        return pressed_digits

    def _set_screen_locked(self):
        self._model.geom_matid[self._screen_geom_id] = self._mat_screen_locked_id

    def _set_screen_unlocked(self):
        self._model.geom_matid[self._screen_geom_id] = self._mat_screen_unlocked_id
        self._screen_unlocked = True

    def _hide_buttons(self):
        for gid in self._button_visual_geom_ids:
            rgba = self._model.geom_rgba[gid].copy()
            rgba[3] = 0.0
            self._model.geom_rgba[gid] = rgba
        self._buttons_hidden = True

    def _show_buttons(self):
        for gid in self._button_visual_geom_ids:
            self._model.geom_rgba[gid] = self._button_visual_orig_rgba[gid]

    def _compute_success(self):
        if self._unlock_failed:
            return False
        if self._screen_unlocked:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0
        return self._success_trigger_count >= self._success_trigger_target
