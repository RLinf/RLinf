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

_MAX_EPISODE_STEPS = 1500
_PEG_SAMPLING_BOUNDS = np.asarray([[-0.3, -0.25], [-0.25, -0.2]], dtype=np.float64)
_SOCKET_SAMPLING_BOUNDS = np.asarray([[-0.3, 0.15], [-0.2, 0.25]], dtype=np.float64)
_PEG_YAW_BOUNDS_DEG = np.asarray([-10.0, 10.0], dtype=np.float64)
_SOCKET_YAW_BOUNDS_DEG = np.asarray([-20.0, 20.0], dtype=np.float64)
_SUCCESS_STEPS_REQUIRED = 30


class BimanualAssemblyTask(DexJoCoTaskBase):
    """Insert the peg into the socket and maintain bottom contact."""

    xml_name = "arena_arm_hand_bimanual_assembly.xml"
    dual_arm = True
    task_description = "Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."

    def _initialize_task(self) -> None:
        self._success_counter = 0
        self._peg_joint_id = self._model.joint("industreal_round_peg_8mm_joint").id
        self._socket_joint_id = self._model.joint(
            "industreal_tray_insert_round_peg_8mm_joint"
        ).id
        self._peg_qpos_adr = int(self._model.jnt_qposadr[self._peg_joint_id])
        self._peg_qvel_adr = int(self._model.jnt_dofadr[self._peg_joint_id])
        self._socket_qpos_adr = int(self._model.jnt_qposadr[self._socket_joint_id])
        self._socket_qvel_adr = int(self._model.jnt_dofadr[self._socket_joint_id])
        self._peg_body_id = self._model.body("industreal_round_peg_8mm").id
        self._socket_body_id = self._model.body(
            "industreal_tray_insert_round_peg_8mm"
        ).id
        self._peg_init_pos = self._model.body_pos[self._peg_body_id].copy()
        self._peg_init_quat = self._model.body_quat[self._peg_body_id].copy()
        self._socket_init_pos = self._model.body_pos[self._socket_body_id].copy()
        self._socket_init_quat = self._model.body_quat[self._socket_body_id].copy()
        self._peg_body_z0 = float(self._peg_init_pos[2])
        self._socket_body_z0 = float(self._socket_init_pos[2])
        self._peg_geom_id = self._model.geom("industreal_round_peg_8mm_collision").id
        self._socket_bottom_geom_id = self._model.geom(
            "industreal_tray_insert_round_peg_8mm_bottom_contact"
        ).id
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
        self._peg_mass0 = float(self._model.body_mass[self._peg_body_id])
        self._socket_mass0 = float(self._model.body_mass[self._socket_body_id])
        self._mass_mul = (0.75, 1.25)

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        out = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
        out /= max(np.linalg.norm(out), 1e-8)
        return out

    def _set_free_joint_pose(
        self, qpos_adr: int, qvel_adr: int, pos: np.ndarray, quat: np.ndarray
    ) -> None:
        self._data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
            quat, dtype=np.float64
        )
        self._data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        delta_h = np.float64(np.random.uniform(0.0, 0.05))
        table_pos = self._model.body("table").pos
        table_pos[2] = delta_h + self._table_z
        self._model.body("table").pos = table_pos
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + delta_h
        peg_xy = np.random.uniform(_PEG_SAMPLING_BOUNDS[0], _PEG_SAMPLING_BOUNDS[1])
        peg_yaw = np.deg2rad(np.random.uniform(*_PEG_YAW_BOUNDS_DEG))
        peg_yaw_quat = np.array(
            [np.cos(peg_yaw / 2.0), 0.0, 0.0, np.sin(peg_yaw / 2.0)], dtype=np.float64
        )
        peg_quat = self._quat_mul(peg_yaw_quat, self._peg_init_quat)
        peg_pos = np.array(
            [peg_xy[0], peg_xy[1], self._peg_body_z0 + delta_h], dtype=np.float64
        )
        self._set_free_joint_pose(
            self._peg_qpos_adr, self._peg_qvel_adr, peg_pos, peg_quat
        )
        socket_xy = np.random.uniform(
            _SOCKET_SAMPLING_BOUNDS[0], _SOCKET_SAMPLING_BOUNDS[1]
        )
        socket_yaw = np.deg2rad(np.random.uniform(*_SOCKET_YAW_BOUNDS_DEG))
        socket_yaw_quat = np.array(
            [np.cos(socket_yaw / 2.0), 0.0, 0.0, np.sin(socket_yaw / 2.0)],
            dtype=np.float64,
        )
        socket_quat = self._quat_mul(socket_yaw_quat, self._socket_init_quat)
        socket_pos = np.array(
            [socket_xy[0], socket_xy[1], self._socket_body_z0 + delta_h],
            dtype=np.float64,
        )
        self._set_free_joint_pose(
            self._socket_qpos_adr, self._socket_qvel_adr, socket_pos, socket_quat
        )
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
            mass = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._peg_body_id] = self._peg_mass0 * mass
            mass = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._socket_body_id] = self._socket_mass0 * mass
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._success_counter = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        success = self._compute_success_metrics()
        obs = self._compute_observation()
        self.env_step += 1
        reward = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return obs, reward, success, truncated, {"success": success}

    def _compute_success_metrics(self) -> bool:
        contact_count = 0
        for i in range(int(self._data.ncon)):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 == self._peg_geom_id and g2 == self._socket_bottom_geom_id) or (
                g2 == self._peg_geom_id and g1 == self._socket_bottom_geom_id
            ):
                contact_count += 1
        bottom_contact = contact_count > 0
        triggered = bottom_contact
        if triggered:
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= _SUCCESS_STEPS_REQUIRED
