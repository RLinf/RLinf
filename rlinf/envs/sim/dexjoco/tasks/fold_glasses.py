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
_SAMPLING_BOUNDS = np.asarray([[-0.4, -0.225], [-0.35, -0.175]])
_BOX_SAMPLING_BOUNDS = np.array([[-0.275, 0.25], [-0.225, 0.3]])
_YAW_PERTURB_BOUNDS = np.array([-10, 10])


class FoldGlassesTask(DexJoCoTaskBase):
    """Fold both glasses arms and place the glasses in the box."""

    xml_name = "arena_arm_hand_glass_v2.xml"
    task_description = "Fold the glasses and place them into the case."

    def _initialize_task(self) -> None:
        self._success_trigger_count = 0
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
        self._open_box_body_z0 = float(self._model.body("open_box").pos[2])
        self._open_box_body_quat0 = self._model.body("open_box").quat.copy()
        self._glass_body_z0 = float(self._model.body("glass").pos[2])
        self._success_trigger_target = 50
        self._init_dynamics_cache()

    def _init_dynamics_cache(self) -> None:
        self._glasses_joint_0_id = int(self._model.joint("glass_joint_0").id)
        self._glasses_joint_1_id = int(self._model.joint("glass_joint_1").id)
        self._glasses_body_id = int(self._model.body("glass").id)
        self._glasses_joint_0_stiffness0 = float(
            self._model.jnt_stiffness[self._glasses_joint_0_id]
        )
        self._glasses_body_mass0 = float(self._model.body_mass[self._glasses_body_id])
        self._glasses_dyn_stiffness_mul = (1, 1.5)
        self._glasses_dyn_mass_mul = (0.75, 1.25)

    def _randomize_glasses_dynamics(self) -> None:
        frictionloss = float(np.random.uniform(0.0, 0.05))
        # Indexed by joint id rather than jnt_dofadr, as in the official task.
        self._model.dof_frictionloss[self._glasses_joint_0_id] = frictionloss
        self._model.dof_frictionloss[self._glasses_joint_1_id] = frictionloss
        stiffness = self._glasses_joint_0_stiffness0 * float(
            np.random.uniform(*self._glasses_dyn_stiffness_mul)
        )
        self._model.jnt_stiffness[self._glasses_joint_0_id] = stiffness
        self._model.jnt_stiffness[self._glasses_joint_1_id] = stiffness
        mass = self._glasses_body_mass0 * float(
            np.random.uniform(*self._glasses_dyn_mass_mul)
        )
        self._model.body_mass[self._glasses_body_id] = mass

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
        box_xy = np.random.uniform(*_BOX_SAMPLING_BOUNDS)
        box_body_id = self._model.body("open_box").id
        box_orig = np.array(self._open_box_body_quat0, dtype=np.float64)
        box_yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        bqw, bqz = np.cos(box_yaw / 2), np.sin(box_yaw / 2)
        bw1, bx1, by1, bz1 = bqw, 0, 0, bqz
        bw2, bx2, by2, bz2 = box_orig
        box_q_new = np.array(
            [
                bw1 * bw2 - bx1 * bx2 - by1 * by2 - bz1 * bz2,
                bw1 * bx2 + bx1 * bw2 + by1 * bz2 - bz1 * by2,
                bw1 * by2 - bx1 * bz2 + by1 * bw2 + bz1 * bx2,
                bw1 * bz2 + bx1 * by2 - by1 * bx2 + bz1 * bw2,
            ]
        )
        box_q_new /= np.linalg.norm(box_q_new)
        open_box_ori_pose = np.concatenate(
            [box_xy, [self._open_box_body_z0 + delta_h], box_q_new]
        ).astype(np.float64)
        self._model.body_pos[box_body_id] = open_box_ori_pose[:3]
        self._model.body_quat[box_body_id] = open_box_ori_pose[3:7]
        glass_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("glass_root").qpos[:2] = glass_xy
        self._data.jnt("glass_root").qpos[2] = self._glass_body_z0 + delta_h
        orig = np.array(self._data.jnt("glass_root").qpos[3:7], dtype=np.float64)
        yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
        w1, x1, y1, z1 = qw, 0, 0, qz
        w2, x2, y2, z2 = orig
        q_new = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        q_new /= np.linalg.norm(q_new)
        glass_ori_pose = np.concatenate(
            [glass_xy, [self._glass_body_z0 + delta_h], q_new]
        ).astype(np.float64)
        self._data.jnt("glass_root").qpos = glass_ori_pose
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._success_trigger_count = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        if self.randomize_dynamics:
            self._randomize_glasses_dynamics()
        return obs, {"success": False}

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
        glass_pos_world = np.array(
            self._data.sensor("glass_pos").data, dtype=np.float64
        )
        body_id = self._model.body("open_box").id
        box_pos_world = np.array(self._model.body_pos[body_id], dtype=np.float64)
        glass_local = glass_pos_world - box_pos_world
        x_half = 0.145
        y_half = 0.145
        z_min = -0.045
        z_max = 0.0475
        inside_x = -x_half <= glass_local[0] <= x_half
        inside_y = -y_half <= glass_local[1] <= y_half
        inside_z = z_min <= glass_local[2] <= z_max - 0.02
        glass_joint_0_pos = self._data.sensor("glass_joint_0_pos").data
        glass_joint_1_pos = self._data.sensor("glass_joint_1_pos").data
        j0 = float(glass_joint_0_pos[0])
        j1 = float(glass_joint_1_pos[0])
        trigger_active = j0 > 1.1 and j1 > 1.1 and inside_x and inside_y and inside_z
        if trigger_active:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0
        return self._success_trigger_count >= self._success_trigger_target
