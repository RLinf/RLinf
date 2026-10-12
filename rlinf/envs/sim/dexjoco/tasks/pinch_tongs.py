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
_TONGS_XY_BOUNDS = np.asarray([[-0.35, -0.25], [-0.3, -0.2]])
_MATCHED_Z = 0.955
_LIFT_HEIGHT = 0.1
_TONGS_CLOSE_THRESHOLD = -0.07
_TONGS_OPEN_THRESHOLD = 0.1
_REQUIRED_PINCHES = 3
_SUCCESS_STEPS_REQUIRED = 30


class PinchTongsTask(DexJoCoTaskBase):
    """Lift the tongs and complete three pinches."""

    xml_name = "arena_arm_hand_table_tongs.xml"
    task_description = (
        "Grasp the tongs and perform three consecutive open-close motions."
    )

    def _initialize_task(self) -> None:
        self._pinch_count = 0
        self._pinch_in_progress = True
        self._success_counter = 0
        self._table_site_id = self._model.site("table_top").id
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
        self._tongs_z0 = _MATCHED_Z
        self._tongs_body_id = self._model.body("link_1").id
        self._tongs_mass0 = float(self._model.body_mass[self._tongs_body_id])
        self._tongs_mass_mul = (0.75, 1.25)
        self._tongs_joint_id = self._model.joint("joint_0").id
        self._tongs_dof_id = int(self._model.jnt_dofadr[self._tongs_joint_id])
        self._tongs_friction_range = (0.0, 0.05)
        self._tongs_stiffness0 = float(self._model.jnt_stiffness[self._tongs_joint_id])
        self._tongs_stiffness_mul = (0.75, 1.25)

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
        self._lift_z = table_z + _LIFT_HEIGHT
        tongs_z = self._tongs_z0 + delta_h
        tongs_xy = np.random.uniform(*_TONGS_XY_BOUNDS)
        tongs_ori_quat = self._data.jnt("tongs_root").qpos[3:7]
        tongs_ori_pose = np.concatenate([tongs_xy, [tongs_z], tongs_ori_quat]).astype(
            np.float64
        )
        self._data.jnt("tongs_root").qpos = tongs_ori_pose
        self._data.qpos[self._model.jnt_qposadr[self._tongs_joint_id]] = 0.3
        self._pinch_count = 0
        self._pinch_in_progress = False
        self._success_counter = 0
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            frictionloss = float(
                np.random.uniform(
                    self._tongs_friction_range[0], self._tongs_friction_range[1]
                )
            )
            stiffness = self._tongs_stiffness0 * float(
                np.random.uniform(
                    self._tongs_stiffness_mul[0], self._tongs_stiffness_mul[1]
                )
            )
            self._model.dof_frictionloss[self._tongs_dof_id] = frictionloss
            self._model.jnt_stiffness[self._tongs_joint_id] = stiffness
            mass_mul = float(
                np.random.uniform(self._tongs_mass_mul[0], self._tongs_mass_mul[1])
            )
            self._model.body_mass[self._tongs_body_id] = self._tongs_mass0 * mass_mul
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        obs = self._compute_observation()
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        self._update_pinch_count()
        obs = self._compute_observation()
        self.env_step += 1
        success = self._compute_success()
        reward = 1.0 if success else 0.0
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        return (
            obs,
            reward,
            success,
            truncated,
            {"success": success, "pinch_count": self._pinch_count},
        )

    def _update_pinch_count(self) -> None:
        joint_pos = float(self._data.sensor("tongs_joint_0_pos").data[0])
        if joint_pos <= _TONGS_CLOSE_THRESHOLD and self._pinch_in_progress:
            self._pinch_in_progress = False
            self._pinch_count += 1
        elif joint_pos >= _TONGS_OPEN_THRESHOLD and not self._pinch_in_progress:
            self._pinch_in_progress = True

    def _compute_success(self) -> bool:
        tongs_pos = self._data.sensor("tongs_pos").data
        lifted = tongs_pos[2] >= self._lift_z
        triggered = lifted and self._pinch_count >= _REQUIRED_PINCHES
        if triggered:
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= _SUCCESS_STEPS_REQUIRED
