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
_SAMPLING_BOUNDS = np.asarray([[-0.2, -0.2], [-0.15, -0.25]])
_FOOD_SAMPLING_BOUNDS = np.array([[-0.35, 0.15], [-0.3, 0.2]])
_YAW_PERTURB_BOUNDS = np.array([-10, 10])


class PickBucketTask(DexJoCoTaskBase):
    """Lift the bucket with the boxed food inside."""

    xml_name = "arena_arm_hand_bucket_pick.xml"
    task_description = "Place the boxed food into the bucket and then lift the bucket."

    def _initialize_task(self) -> None:
        self._table_body_z0 = float(self._model.body("table").pos[2])
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
        self._bucket_body_z0 = float(self._model.body("bucket").pos[2])
        self._boxed_food_body_z0 = float(self._model.body("boxed_food_0").pos[2])
        self._bucket_bottom_site_ids = [
            self._model.site(f"bucket_ref_{i}").id for i in (0, 2, 4, 6)
        ]
        self._bucket_bottom_z0 = None
        self._bucket_body_id = self._model.body("bucket").id
        self._boxed_food_body_id = self._model.body("boxed_food_0").id
        self._bucket_mass0 = float(self._model.body_mass[self._bucket_body_id])
        self._boxed_food_mass0 = float(self._model.body_mass[self._boxed_food_body_id])
        self._mass_mul = (0.75, 1.25)
        self._bucket_joint_id = self._model.joint("bucket_joint_0").id
        self._bucket_dof_id = int(self._model.jnt_dofadr[self._bucket_joint_id])
        self._bucket_friction_mul = (0.75, 1.25)

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
        bucket_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        bucket_z = self._bucket_body_z0 + delta_h
        bucket_orig = np.array(
            self._data.jnt("bucket_root").qpos[3:7], dtype=np.float64
        )
        bucket_yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        bqw, bqz = np.cos(bucket_yaw / 2), np.sin(bucket_yaw / 2)
        bw1, bx1, by1, bz1 = bqw, 0, 0, bqz
        bw2, bx2, by2, bz2 = bucket_orig
        bucket_q_new = np.array(
            [
                bw1 * bw2 - bx1 * bx2 - by1 * by2 - bz1 * bz2,
                bw1 * bx2 + bx1 * bw2 + by1 * bz2 - bz1 * by2,
                bw1 * by2 - bx1 * bz2 + by1 * bw2 + bz1 * bx2,
                bw1 * bz2 + bx1 * by2 - by1 * bx2 + bz1 * bw2,
            ]
        )
        bucket_q_new /= np.linalg.norm(bucket_q_new)
        bucket_ori_pose = np.concatenate([bucket_xy, [bucket_z], bucket_q_new]).astype(
            np.float64
        )
        self._data.jnt("bucket_root").qpos = bucket_ori_pose
        box_food_xy = np.random.uniform(*_FOOD_SAMPLING_BOUNDS)
        orig = np.array(
            self._data.jnt("boxed_food_0_freejoint").qpos[3:7], dtype=np.float64
        )
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
        box_food_ori_pose = np.concatenate(
            [box_food_xy, [self._boxed_food_body_z0 + delta_h], q_new]
        ).astype(np.float64)
        self._data.jnt("boxed_food_0_freejoint").qpos = box_food_ori_pose
        if self.randomize:
            self._randomize_lighting()
            self._randomize_camera()
            self._randomize_desktop_texture()
        if self.randomize_dynamics:
            # Scales the current value, so friction compounds across resets as in the
            # official task.
            frictionloss = float(
                self._model.dof_frictionloss[self._bucket_dof_id]
            ) * float(
                np.random.uniform(
                    self._bucket_friction_mul[0], self._bucket_friction_mul[1]
                )
            )
            self._model.dof_frictionloss[self._bucket_dof_id] = frictionloss
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._bucket_body_id] = self._bucket_mass0 * mass_mul
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._boxed_food_body_id] = (
                self._boxed_food_mass0 * mass_mul
            )
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._bucket_bottom_z0 = self._data.site_xpos[
            self._bucket_bottom_site_ids, 2
        ].copy()
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
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
        box_pos = np.array(self._data.sensor("boxed_food_0_pos").data, dtype=float)
        site_ids = [self._model.site(f"bucket_ref_{i}").id for i in range(8)]
        corners = np.array(self._data.site_xpos[site_ids], dtype=float)
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)
        inside = np.all(box_pos >= mins) and np.all(box_pos <= maxs)
        bottom_z = self._data.site_xpos[self._bucket_bottom_site_ids, 2]
        lifted = np.all(bottom_z - self._bucket_bottom_z0 >= 0.15)
        success = inside and lifted
        return success
