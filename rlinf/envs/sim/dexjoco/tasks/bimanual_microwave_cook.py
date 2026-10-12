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
from scipy.spatial.transform import Rotation as R

from rlinf.envs.sim.dexjoco.tasks.base import DexJoCoTaskBase

_MAX_EPISODE_STEPS = 1100
_SAMPLING_BOUNDS = np.asarray([[-0.35, -0.3], [-0.25, -0.4]])
_HOT_DOG_YAW_BOUNDS = (-20.0, 20.0)


class BimanualMicrowaveCookTask(DexJoCoTaskBase):
    """Put the hot dog in the microwave, close the door, and press start."""

    xml_name = "arena_arm_hand_microwave_cook.xml"
    dual_arm = True
    task_description = "Open the microwave door, place the food inside the microwave, close the door, and press the start button."
    # Float64, unlike the other tasks, as in the official environment.
    _ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0))

    def _initialize_task(self) -> None:
        self._table_z = self._model.body("table").pos[2].copy()
        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._model_geom_size0 = self._model.geom_size.copy()
        self._hot_dog_body_z0 = self._model.body("hot_dog").pos[2].copy()
        self._microwave_body_z0 = self._model.body("microwave_object").pos[2].copy()
        self._micro_joint_id = self._model.joint("microjoint").id
        self._micro_dof_id = int(self._model.jnt_dofadr[self._micro_joint_id])
        self._micro_friction0 = float(self._model.dof_frictionloss[self._micro_dof_id])
        self._micro_friction_mul = (0.75, 1.25)
        self._microwave_body_id = self._model.body("microwave_object").id
        self._hot_dog_body_id = self._model.body("hot_dog").id
        self._microwave_mass0 = float(self._model.body_mass[self._microwave_body_id])
        self._hot_dog_mass0 = float(self._model.body_mass[self._hot_dog_body_id])
        self._mass_mul = (0.75, 1.25)

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
        leg_names = ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
        dh = delta_h
        for lname in leg_names:
            lgid = self._model.geom(lname).id
            base_center_z = float(self._model_geom_pos0[lgid, 2])
            base_half_len = float(self._model_geom_size0[lgid, 1])
            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh
            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len
        hot_dog_base_quat = self._model.body("hot_dog").quat
        hot_dog_base_quat = np.array(
            [
                hot_dog_base_quat[1],
                hot_dog_base_quat[2],
                hot_dog_base_quat[3],
                hot_dog_base_quat[0],
            ]
        )
        r_base = R.from_quat(hot_dog_base_quat)
        yaw_deg = np.random.uniform(*_HOT_DOG_YAW_BOUNDS)
        r_yaw = R.from_euler("z", yaw_deg, degrees=True)
        r_new = r_yaw * r_base
        hot_dog_ori_quat = r_new.as_quat()
        hot_dog_ori_quat = np.array(
            [
                hot_dog_ori_quat[3],
                hot_dog_ori_quat[0],
                hot_dog_ori_quat[1],
                hot_dog_ori_quat[2],
            ]
        )
        hot_dog_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        hot_dog_z = self._hot_dog_body_z0 + delta_h
        hot_dog_ori_pos = (*hot_dog_xy, hot_dog_z)
        hot_dog_ori_pose = np.concatenate([hot_dog_ori_pos, hot_dog_ori_quat]).astype(
            np.float64
        )
        self._data.jnt("hot_dog_free").qpos = hot_dog_ori_pose
        microwave_ori_pos = self._model.body("microwave_object").pos
        microwave_ori_quat = self._model.body("microwave_object").quat
        microwave_ori_pos[2] = self._microwave_body_z0 + delta_h
        microwave_ori_pose = np.concatenate(
            [microwave_ori_pos, microwave_ori_quat]
        ).astype(np.float64)
        self._model.body("microwave_object").pos = microwave_ori_pose[:3]
        self._model.body("microwave_object").quat = microwave_ori_pose[3:]
        self._data.qpos[self._panda_right_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._panda_left_dof_ids] = self._PANDA_HOME
        self._data.qpos[self._allegro_dof_right_ids] = self._ALLEGRO_HOME
        self._data.qpos[self._allegro_dof_left_ids] = self._ALLEGRO_HOME
        mujoco.mj_forward(self._model, self._data)
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        mujoco.mj_forward(self._model, self._data)
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            frictionloss = self._micro_friction0 * float(
                np.random.uniform(
                    self._micro_friction_mul[0], self._micro_friction_mul[1]
                )
            )
            self._model.dof_frictionloss[self._micro_dof_id] = frictionloss
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._microwave_body_id] = (
                self._microwave_mass0 * mass_mul
            )
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._hot_dog_body_id] = (
                self._hot_dog_mass0 * mass_mul
            )
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"success": False}

    def _geom_in_contact(
        self, target_geom_name: str, dist_threshold: float = 0.0
    ) -> bool:
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1))
            g2 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))
            dist = float(c.dist)
            if (
                g1 == target_geom_name or g2 == target_geom_name
            ) and dist <= dist_threshold:
                return True
        return False

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        obs = self._compute_observation()
        self.env_step += 1
        success = self._compute_success()
        rew = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return obs, rew, success, truncated, {"success": success}

    def _compute_success(self):
        micro_qpos = float(self._data.joint("microjoint").qpos[0])
        microwave_closed = abs(micro_qpos) < 1e-2
        inside = self._hot_dog_inside_microwave(margin=0.01)
        contact = self._geom_in_contact("start_button")
        return inside and microwave_closed and contact

    def _world_to_body(self, body_name: str, p_world: np.ndarray) -> np.ndarray:
        body_pos = self._data.body(body_name).xpos
        body_mat = self._data.body(body_name).xmat.reshape(3, 3)
        return body_mat.T @ (p_world - body_pos)

    def _hot_dog_inside_microwave(self, margin: float = 0.0) -> bool:
        p_world = self._data.sensor("hot_dog_pos").data
        p_local = self._world_to_body("microwave_object", p_world)
        p0 = self._model.site("int_p0").pos
        px = self._model.site("int_px").pos
        py = self._model.site("int_py").pos
        pz = self._model.site("int_pz").pos
        inside_x = p0[0] - margin <= p_local[0] <= px[0] + margin
        inside_y = p0[1] - margin <= p_local[1] <= py[1] + margin
        inside_z = p0[2] - margin <= p_local[2] <= pz[2] + margin
        return inside_x and inside_y and inside_z
