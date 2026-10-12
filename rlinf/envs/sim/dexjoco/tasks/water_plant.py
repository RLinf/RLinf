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

_SPRAY_SAMPLE_LOW = np.array([-0.35, -0.25], dtype=np.float64)
_SPRAY_SAMPLE_HIGH = np.array([-0.3, -0.2], dtype=np.float64)
_PLANT_SAMPLE_LOW = np.array([-0.1, 0.15], dtype=np.float64)
_PLANT_SAMPLE_HIGH = np.array([-0.05, 0.2], dtype=np.float64)
_MAX_EPISODE_STEPS = 1000
_CONE_VISIBLE_STEPS = 30
_TRIGGER_RELEASE_THRESHOLD = 0.25
_TRIGGER_PULL_THRESHOLD = 0.34
_SUCCESS_STEPS_REQUIRED = 30
_TABLE_LEG_NAMES = ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")


class WaterPlantTask(DexJoCoTaskBase):
    """Spray water into the plant's target region."""

    xml_name = "arena_arm_hand_plant.xml"
    task_description = "Grasp the watering can and apply water to the plant."

    def _initialize_task(self) -> None:
        self._cone_alpha_counter = 0
        self._trigger_pulled = False
        self._success_counter = 0
        self._table_body_z0 = float(self._model.body("table").pos[2])
        self._table_leg_half_len0 = {
            name: float(self._model.geom(name).size[1]) for name in _TABLE_LEG_NAMES
        }
        self._plant_body_z0 = float(self._model.body("plant").pos[2])
        self._spray_body_z0 = float(self._model.body("link_2").pos[2])
        self._init_spray_dynamics_cache()

    def _init_spray_dynamics_cache(self) -> None:
        self._spray_joint_id = int(self._model.joint("joint_0").id)
        self._spray_dof_id = int(self._model.jnt_dofadr[self._spray_joint_id])
        self._spray_body_id = int(self._model.body("link_2").id)
        self._spray_joint_stiffness0 = float(
            self._model.jnt_stiffness[self._spray_joint_id]
        )
        self._spray_body_mass0 = float(self._model.body_mass[self._spray_body_id])
        self._spray_dyn_friction_range = (0.0, 0.05)
        self._spray_dyn_stiffness_mul = (0.75, 1.25)
        self._spray_dyn_mass_mul = (0.75, 1.25)

    def _randomize_spray_dynamics(self) -> None:
        def _mul(rng):
            return float(np.random.uniform(rng[0], rng[1]))

        frictionloss = float(
            np.random.uniform(
                self._spray_dyn_friction_range[0], self._spray_dyn_friction_range[1]
            )
        )
        stiffness = self._spray_joint_stiffness0 * _mul(self._spray_dyn_stiffness_mul)
        mass = self._spray_body_mass0 * _mul(self._spray_dyn_mass_mul)
        self._model.dof_frictionloss[self._spray_dof_id] = frictionloss
        self._model.jnt_stiffness[self._spray_joint_id] = stiffness
        self._model.body_mass[self._spray_body_id] = mass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        delta_h = np.random.uniform(0.0, 0.05)
        table_body_id = self._model.body("table").id
        self._model.body_pos[table_body_id, 2] = self._table_body_z0 + delta_h
        for lname in _TABLE_LEG_NAMES:
            lgid = self._model.geom(lname).id
            self._model.geom_size[lgid, 1] = self._table_leg_half_len0[lname] + delta_h
        geom_id = self._model.geom("cone_visual").id
        self._model.geom_rgba[geom_id][3] = 0
        spray_xy = np.random.uniform(_SPRAY_SAMPLE_LOW, _SPRAY_SAMPLE_HIGH)
        spray_ori_pos = self._model.body("link_2").pos
        spray_ori_pos[:2] = spray_xy
        spray_ori_pos[2] = self._spray_body_z0 + delta_h
        self._data.jnt("spray_root").qpos[:3] = spray_ori_pos
        body_id = self._model.body("plant").id
        plant_xy = np.random.uniform(_PLANT_SAMPLE_LOW, _PLANT_SAMPLE_HIGH)
        plant_ori_pos = self._model.body("plant").pos
        plant_ori_pos[:2] = plant_xy
        plant_ori_pos[2] = self._plant_body_z0 + delta_h
        self._model.body_pos[body_id] = plant_ori_pos
        self._data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_ids] = self._ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)
        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        self._data.mocap_pos[self._mocap_id] = tcp_pos
        self._data.mocap_quat[self._mocap_id] = tcp_quat
        self.env_step = 0
        self._trigger_pulled = False
        self._cone_alpha_counter = 0
        self._success_counter = 0
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            self._randomize_spray_dynamics()
        mujoco.mj_forward(self._model, self._data)
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        watering_joint_0_pos = float(self._data.sensor("spray_joint_0_pos").data[0])
        was_trigger_pulled = self._trigger_pulled
        if watering_joint_0_pos < _TRIGGER_RELEASE_THRESHOLD:
            self._trigger_pulled = False
        elif watering_joint_0_pos > _TRIGGER_PULL_THRESHOLD:
            self._trigger_pulled = True
        if self._trigger_pulled and not was_trigger_pulled:
            self._cone_alpha_counter = _CONE_VISIBLE_STEPS
        if self._cone_alpha_counter > 0:
            alpha = 0.5
            self._cone_alpha_counter -= 1
        else:
            alpha = 0.0
        geom_id = self._model.geom("cone_visual").id
        rgba = self._model.geom_rgba[geom_id].copy()
        rgba[3] = alpha
        self._model.geom_rgba[geom_id] = rgba
        obs = self._compute_observation()
        self.env_step += 1
        success, failed_trigger_outside = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = success or failed_trigger_outside
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return obs, rew, terminated, truncated, {"success": success}

    def _compute_success(self):
        site_id = self._model.site("ref_point").id
        p_world = self._data.site_xpos[site_id]
        obj_world_pos = self._data.body("plant").xpos
        R = 0.2
        HALF_H = 0.2
        dx, dy, dz = p_world - obj_world_pos
        inside = dx * dx + dy * dy <= R * R and -HALF_H <= dz <= HALF_H
        failed_trigger_outside = not inside and self._trigger_pulled
        if inside and self._trigger_pulled:
            self._success_counter += 1
        else:
            self._success_counter = 0
        success = self._success_counter >= _SUCCESS_STEPS_REQUIRED
        return success, failed_trigger_outside
