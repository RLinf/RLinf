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
_HANOI_BASE_SAMPLING_BOUNDS = np.asarray([[-0.25, 0], [-0.2, 0]], dtype=np.float64)
_HANOI_RESET_STATE = {
    "A": ["hanoi_disk_small"],
    "B": ["hanoi_disk_medium"],
    "C": ["hanoi_disk_large"],
}


class BimanualHanoiTask(DexJoCoTaskBase):
    """Stack all three disks on post C from largest to smallest."""

    xml_name = "arena_arm_hand_bimanual_hanoi.xml"
    dual_arm = True
    task_description = "Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand."

    def _initialize_task(
        self,
        post_assignment_tol: float = 0.04,
        placement_height_tol: float = 0.012,
        static_qvel_tol: float = 0.35,
        success_trigger_target: int = 20,
        illegal_trigger_target: int = 10,
    ) -> None:
        self._post_assignment_tol = float(post_assignment_tol)
        self._placement_height_tol = float(placement_height_tol)
        self._static_qvel_tol = float(static_qvel_tol)
        self._success_trigger_target = int(success_trigger_target)
        self._illegal_trigger_target = int(illegal_trigger_target)
        self._success_trigger_count = 0
        self._illegal_trigger_count = 0
        self._illegal_trigger_reason = ""
        self._base_body_id = self._model.body("hanoi_base").id
        self._base_init_pos = self._model.body_pos[self._base_body_id].copy()
        self._base_body_z0 = float(self._base_init_pos[2])
        self._disk_names = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        self._disk_size_rank = {
            "hanoi_disk_large": 3,
            "hanoi_disk_medium": 2,
            "hanoi_disk_small": 1,
        }
        self._disk_qpos_adr = {}
        self._disk_qvel_adr = {}
        self._disk_body_id = {}
        self._disk_init_pos = {}
        self._disk_init_quat = {}
        self._disk_half_height = {}
        self._disk_mass0 = {}
        self._disk_mass_mul = (0.75, 1.25)
        for disk_name in self._disk_names:
            joint_name = f"{disk_name}_joint"
            joint_id = self._model.joint(joint_name).id
            body_id = self._model.body(disk_name).id
            self._disk_qpos_adr[disk_name] = int(self._model.jnt_qposadr[joint_id])
            self._disk_qvel_adr[disk_name] = int(self._model.jnt_dofadr[joint_id])
            self._disk_body_id[disk_name] = body_id
            self._disk_init_pos[disk_name] = self._model.body_pos[body_id].copy()
            self._disk_init_quat[disk_name] = self._model.body_quat[body_id].copy()
            self._disk_mass0[disk_name] = float(self._model.body_mass[body_id])
            top_site = self._model.site(f"{disk_name}_top_site").id
            bottom_site = self._model.site(f"{disk_name}_bottom_site").id
            half_height = 0.5 * (
                float(self._model.site_pos[top_site][2])
                - float(self._model.site_pos[bottom_site][2])
            )
            self._disk_half_height[disk_name] = half_height
        self._post_site_names = {
            "A": "hanoi_post_a_site",
            "B": "hanoi_post_b_site",
            "C": "hanoi_post_c_site",
        }
        self._post_site_ids = {
            name: self._model.site(site_name).id
            for name, site_name in self._post_site_names.items()
        }
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
        self._base_upper_geom_id = self._model.geom("hanoi_base_upper_collision").id
        self._base_upper_top_local_z = float(
            self._model.geom_pos[self._base_upper_geom_id, 2]
            + self._model.geom_size[self._base_upper_geom_id, 2]
        )

    def _set_free_joint_pose(
        self, qpos_adr: int, qvel_adr: int, pos: np.ndarray, quat: np.ndarray
    ) -> None:
        self._data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
            quat, dtype=np.float64
        )
        self._data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _apply_reset_tower_state(
        self,
        tower_state: dict[str, list[str]],
        base_delta_xy: np.ndarray,
        delta_h: float,
    ) -> None:
        base_top_z = self._base_body_z0 + delta_h + self._base_upper_top_local_z
        post_xy = {
            post_name: self._base_init_pos[:2] + self._model.site_pos[site_id][:2]
            for post_name, site_id in self._post_site_ids.items()
        }
        for post_name, disk_names in tower_state.items():
            running_height = base_top_z
            for disk_name in disk_names:
                disk_pos = self._disk_init_pos[disk_name].copy()
                disk_pos[:2] = post_xy[post_name] + base_delta_xy
                disk_pos[2] = running_height + self._disk_half_height[disk_name]
                running_height += 2.0 * self._disk_half_height[disk_name]
                self._set_free_joint_pose(
                    self._disk_qpos_adr[disk_name],
                    self._disk_qvel_adr[disk_name],
                    disk_pos,
                    self._disk_init_quat[disk_name],
                )

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
        base_xy = np.random.uniform(
            _HANOI_BASE_SAMPLING_BOUNDS[0], _HANOI_BASE_SAMPLING_BOUNDS[1]
        )
        base_pos = np.array(
            [base_xy[0], base_xy[1], self._base_body_z0 + delta_h], dtype=np.float64
        )
        base_delta_xy = base_pos[:2] - self._base_init_pos[:2]
        self._model.body_pos[self._base_body_id] = base_pos
        self._apply_reset_tower_state(_HANOI_RESET_STATE, base_delta_xy, delta_h)
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
        mujoco.mj_forward(self._model, self._data)
        self.env_step = 0
        self._prime_rgb_array_renderer()
        self._success_trigger_count = 0
        self._illegal_trigger_count = 0
        self._illegal_trigger_reason = ""
        obs = self._compute_observation()
        if self.randomize_dynamics:
            for disk_name in self._disk_names:
                body_id = self._disk_body_id[disk_name]
                mass_mul = float(
                    np.random.uniform(self._disk_mass_mul[0], self._disk_mass_mul[1])
                )
                self._model.body_mass[body_id] = self._disk_mass0[disk_name] * mass_mul
        return obs, {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        success, metrics = self._compute_success_metrics()
        obs = self._compute_observation()
        self.env_step += 1
        reward = 1.0 if success else 0.0
        terminated = success or metrics["illegal_state"]
        truncated = self.env_step >= _MAX_EPISODE_STEPS
        success_stable_count = int(self._success_trigger_count)
        illegal_stable_count = int(self._illegal_trigger_count)
        info = {
            "success": success,
            "tower_state": metrics["tower_state"],
            "disk_assignment": metrics["disk_assignment"],
            "all_disks_static": metrics["all_disks_static"],
            "illegal_state": metrics["illegal_state"],
            "illegal_reason": metrics["illegal_reason"],
            "xy_error_max": float(metrics["xy_error_max"]),
            "height_error_max": float(metrics["height_error_max"]),
            "success_stable_count": success_stable_count,
            "illegal_stable_count": illegal_stable_count,
        }
        return obs, reward, terminated, truncated, info

    def _disk_is_static(self, disk_name: str) -> bool:
        qvel_adr = self._disk_qvel_adr[disk_name]
        return (
            float(np.linalg.norm(self._data.qvel[qvel_adr : qvel_adr + 6]))
            <= self._static_qvel_tol
        )

    def _get_post_world_positions(self) -> dict[str, np.ndarray]:
        return {
            post_name: np.array(self._data.site_xpos[site_id], dtype=np.float64)
            for post_name, site_id in self._post_site_ids.items()
        }

    def _base_top_world_z(self) -> float:
        base_body = self._data.body("hanoi_base")
        return float(base_body.xpos[2] + self._base_upper_top_local_z)

    def _reconstruct_tower_state(self) -> dict[str, Any]:
        post_positions = self._get_post_world_positions()
        base_top_z = self._base_top_world_z()
        disk_assignment = {}
        disk_positions = {}
        disk_static = {}
        towers = {"A": [], "B": [], "C": []}
        for disk_name in self._disk_names:
            pos = np.array(self._data.body(disk_name).xpos, dtype=np.float64)
            disk_positions[disk_name] = pos
            disk_static[disk_name] = self._disk_is_static(disk_name)
            closest_post = min(
                post_positions.keys(),
                key=lambda k: np.linalg.norm(pos[:2] - post_positions[k][:2]),
            )
            closest_dist = float(
                np.linalg.norm(pos[:2] - post_positions[closest_post][:2])
            )
            if closest_dist <= self._post_assignment_tol:
                disk_assignment[disk_name] = closest_post
                towers[closest_post].append(disk_name)
            elif pos[2] <= base_top_z + 0.06:
                disk_assignment[disk_name] = "table"
            else:
                disk_assignment[disk_name] = "floating"
        for post_name in towers:
            towers[post_name].sort(key=lambda name: disk_positions[name][2])
        return {
            "post_positions": post_positions,
            "base_top_z": float(base_top_z),
            "disk_assignment": disk_assignment,
            "disk_positions": disk_positions,
            "disk_static": disk_static,
            "towers": towers,
        }

    def _tower_state_repr(self, towers: dict[str, list[str]]) -> dict[str, list[str]]:
        return {post: list(names) for post, names in towers.items()}

    def _tower_geometry_errors(
        self,
        disks: list[str],
        post_name: str,
        disk_positions: dict[str, np.ndarray],
        post_positions: dict[str, np.ndarray],
        base_top_z: float,
    ) -> tuple[float, float]:
        running_height = base_top_z
        xy_error_max = 0.0
        height_error_max = 0.0
        for disk_name in disks:
            expected_center_z = running_height + self._disk_half_height[disk_name]
            running_height += 2.0 * self._disk_half_height[disk_name]
            pos = disk_positions[disk_name]
            xy_error = float(np.linalg.norm(pos[:2] - post_positions[post_name][:2]))
            z_error = float(abs(pos[2] - expected_center_z))
            xy_error_max = max(xy_error_max, xy_error)
            height_error_max = max(height_error_max, z_error)
        return xy_error_max, height_error_max

    def _compute_success_metrics(self) -> tuple[bool, dict[str, Any]]:
        state = self._reconstruct_tower_state()
        towers = state["towers"]
        disk_assignment = state["disk_assignment"]
        disk_positions = state["disk_positions"]
        disk_static = state["disk_static"]
        post_positions = state["post_positions"]
        base_top_z = state["base_top_z"]
        tower_state = self._tower_state_repr(towers)
        all_disks_static = all(bool(v) for v in disk_static.values())
        illegal_candidate = False
        illegal_reason = ""
        for post_name, disks in towers.items():
            ranks = [self._disk_size_rank[disk] for disk in disks]
            if any(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)):
                if all(disk_static[disk] for disk in disks):
                    xy_error, height_error = self._tower_geometry_errors(
                        disks, post_name, disk_positions, post_positions, base_top_z
                    )
                    if (
                        xy_error <= self._post_assignment_tol
                        and height_error <= self._placement_height_tol
                    ):
                        illegal_candidate = True
                        illegal_reason = f"illegal_stack_{post_name}"
                        break
        if not illegal_candidate and all_disks_static:
            on_table = [
                disk for disk, assign in disk_assignment.items() if assign == "table"
            ]
            if on_table:
                illegal_candidate = True
                illegal_reason = "disk_on_table"
        if illegal_candidate:
            if illegal_reason == self._illegal_trigger_reason:
                self._illegal_trigger_count += 1
            else:
                self._illegal_trigger_reason = illegal_reason
                self._illegal_trigger_count = 1
        else:
            self._illegal_trigger_reason = ""
            self._illegal_trigger_count = 0
        illegal_state = self._illegal_trigger_count >= self._illegal_trigger_target
        xy_error_max = 0.0
        height_error_max = 0.0
        target_stack = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        if (
            towers["C"] == target_stack
            and len(towers["A"]) == 0
            and len(towers["B"]) == 0
        ):
            xy_error_max, height_error_max = self._tower_geometry_errors(
                target_stack, "C", disk_positions, post_positions, base_top_z
            )
            geom_success = (
                xy_error_max <= self._post_assignment_tol
                and height_error_max <= self._placement_height_tol
                and all_disks_static
            )
        else:
            geom_success = False
        if geom_success and not illegal_state:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0
        success = self._success_trigger_count >= self._success_trigger_target
        return bool(success), {
            "tower_state": tower_state,
            "disk_assignment": dict(disk_assignment),
            "all_disks_static": bool(all_disks_static),
            "illegal_state": bool(illegal_state),
            "illegal_reason": illegal_reason,
            "xy_error_max": float(xy_error_max),
            "height_error_max": float(height_error_max),
        }
