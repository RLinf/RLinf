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
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from scipy.spatial.transform import Rotation as R

_CAMERAS = {
    "front": "front",
    "wrist": "handcam_rgb",
    "ego_left": "left",
    "ego_right": "right",
    "ego": "back",
    "wrist_left": "handcam_rgb_left",
    "wrist_right": "handcam_rgb_right",
}
_SCENE_CENTER = np.array([0.0, 0.0, 1.0])


class DexJoCoTaskBase(gym.Env):
    """MuJoCo scene, Cartesian control, and RGB observations for DexJoCo tasks."""

    xml_name: str
    dual_arm = False
    _PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
    _ALLEGRO_HOME = np.asarray(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32
    )
    _TABLE_TEXTURES = (
        "table_bamboo",
        "table_blue-wood",
        "table_brass-ambra",
        "table_ceramic",
        "table_cream-plaster",
        "table_dark_wood_planks_2",
        "table_dark-wood",
        "table_gray-plaster",
        "table_gray_wood_planks",
        "table_light-wood",
        "table_metal",
        "table_pink-plaster",
        "table_red-wood",
        "table_legs_metal",
        "table_steel-scratched",
        "table_walnut_wood_grain",
        "table_warm_wood_grain_2",
        "table_white-plaster",
        "table_wood_grain_1",
        "table_yellow-plaster",
    )

    def __init__(
        self,
        asset_dir: str,
        camera_names: list[str],
        camera_heights: int = 640,
        camera_widths: int = 640,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        **task_options,
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(
            str(Path(asset_dir) / "xmls" / self.xml_name)
        )
        self._model.vis.global_.offwidth = camera_widths
        self._model.vis.global_.offheight = camera_heights
        self._model.opt.timestep = physics_dt
        self._data = mujoco.MjData(self._model)
        self.control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.env_step = 0
        self._arms = []
        hand_controls = []
        for side in ["right", "left"] if self.dual_arm else [""]:
            suffix = f"_{side}" if side else ""
            arm_dofs = np.array(
                [self._model.joint(f"joint{i}{suffix}").id for i in range(1, 8)]
            )
            arm_controls = np.array(
                [self._model.actuator(f"actuator{i}{suffix}").id for i in range(1, 8)]
            )
            fingers = (
                ["rf", "mf", "ff", "th"] if side == "left" else ["ff", "mf", "rf", "th"]
            )
            joints = [f"{finger}j{i}{suffix}" for finger in fingers for i in range(4)]
            hand_dofs = np.array(
                [int(self._model.joint(name).qposadr[0]) for name in joints]
            )
            hand_controls.extend(
                self._model.actuator(f"{finger}a{i}{suffix}").id
                for finger in fingers
                for i in range(4)
            )
            site = self._model.site(f"attachment_site{suffix}").id
            mocap = int(self._model.body(f"target{suffix}").mocapid[0])
            self._arms.append((arm_dofs, arm_controls, site, mocap))
            if side == "right":
                self._panda_right_dof_ids = arm_dofs
                self._allegro_dof_right_ids = hand_dofs
                self._mocap_right_id = mocap
            elif side == "left":
                self._panda_left_dof_ids = arm_dofs
                self._allegro_dof_left_ids = hand_dofs
                self._mocap_left_id = mocap
            else:
                self._allegro_dof_ids = hand_dofs
                self._mocap_id = mocap
        self._panda_dof_ids = np.concatenate([arm[0] for arm in self._arms])
        self._allegro_ctrl_ids = np.array(hand_controls)
        self._viewer = MujocoRenderer(
            self._model, self._data, width=camera_widths, height=camera_heights
        )
        self._viewer.render("rgb_array")
        self._front_camera_id = self._model.camera(
            "back" if self.dual_arm else "front"
        ).id
        self._cameras = {
            name: self._model.camera(_CAMERAS[name]).id for name in camera_names
        }
        self._camera_params = np.load(
            Path(asset_dir)
            / ("replay_cameras_2.npy" if self.dual_arm else "replay_cameras.npy")
        )
        self._num_preset_cameras = len(self._camera_params)
        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()
        self._table_geom_id = self._model.geom("table_visual").id
        self._initialize_task(**task_options)

    def _initialize_task(self, **task_options) -> None:
        raise NotImplementedError

    def _apply_action(self, action: np.ndarray) -> None:
        for index, (_, _, _, mocap) in enumerate(self._arms):
            pose = action[7 * index : 7 * (index + 1)]
            self._data.mocap_pos[mocap] = pose[:3]
            self._data.mocap_quat[mocap] = pose[3:7]
        hand_action = action[14 if self.dual_arm else 7 :]
        for _ in range(self._n_substeps):
            for dofs, controls, site, mocap in self._arms:
                self._data.ctrl[controls] = self._compute_torque(dofs, site, mocap)
            self._data.ctrl[self._allegro_ctrl_ids] = hand_action
            mujoco.mj_step(self._model, self._data)
            self._after_substep()

    def _after_substep(self):
        pass

    def _compute_torque(self, dofs: np.ndarray, site: int, mocap: int) -> np.ndarray:
        jac_pos = np.zeros((3, self._model.nv))
        jac_ori = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jac_pos, jac_ori, site)
        jac_pos, jac_ori = jac_pos[:, dofs], jac_ori[:, dofs]
        jac = np.concatenate([jac_pos, jac_ori])
        velocity = self._data.qvel[dofs]
        position_error = self._data.mocap_pos[mocap] - self._data.site_xpos[site]
        orientation = R.from_matrix(self._data.site_xmat[site].reshape(3, 3))
        target = R.from_quat(self._data.mocap_quat[mocap], scalar_first=True)
        orientation_error = (orientation * target.inv()).as_rotvec()
        acceleration = np.concatenate(
            [
                400 * position_error - 160 * (jac_pos @ velocity),
                -200 * orientation_error - 8 * np.sqrt(200) * (jac_ori @ velocity),
            ]
        )
        mass = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, mass, self._data.qM)
        mass_inverse = np.linalg.inv(mass[np.ix_(dofs, dofs)])
        task_inverse = jac @ mass_inverse @ jac.T
        task_mass = (
            np.linalg.inv(task_inverse)
            if abs(np.linalg.det(task_inverse)) >= 1e-2
            else np.linalg.pinv(task_inverse, rcond=1e-2)
        )
        torque = jac.T @ task_mass @ acceleration
        nullspace = mass_inverse @ jac.T @ task_mass
        joint_acceleration = (
            0.5 * (self._PANDA_HOME - self._data.qpos[dofs])
            - 8 * np.sqrt(0.5) * velocity
        )
        torque += (np.eye(len(dofs)) - jac.T @ nullspace.T) @ joint_acceleration
        return torque + self._data.qfrc_bias[dofs]

    def _render_camera(self, camera: int) -> np.ndarray:
        self._viewer.camera_id = camera
        return self._viewer.render("rgb_array").copy()

    def _prime_rgb_array_renderer(self):
        for camera in self._cameras.values():
            self._render_camera(camera)

    def _compute_observation(self):
        poses, hands = [], []
        for side in ["right", "left"] if self.dual_arm else [""]:
            suffix = f"_{side}" if side else ""
            poses.extend(
                [
                    self._data.sensor(f"franka/flange_pos{suffix}").data.copy(),
                    self._data.sensor(f"franka/flange_quat{suffix}").data.copy(),
                ]
            )
            fingers = (
                ["rf", "mf", "ff", "th"] if side == "left" else ["ff", "mf", "rf", "th"]
            )
            hands.append(
                np.array(
                    [
                        self._data.sensor(
                            f"allegro_{side or 'right'}/{finger}j{i}_pos"
                        ).data[0]
                        for finger in fingers
                        for i in range(4)
                    ],
                    dtype=np.float32,
                )
            )
        return {
            "state": np.concatenate(poses + hands),
            **{
                name: self._render_camera(camera)
                for name, camera in self._cameras.items()
            },
        }

    def close(self) -> None:
        """Release all offscreen rendering contexts."""
        self._viewer.close()

    def _randomize_lighting(self):
        model = self._model
        model.light_pos[:] = self._orig_light_pos
        model.light_dir[:] = self._orig_light_dir
        for i in range(model.nlight):
            model.light_pos[i, 0] += random.uniform(-0.3, 0.3)
            model.light_pos[i, 1] += random.uniform(-0.3, 0.3)
            model.light_dir[i, 0] += random.uniform(-0.4, 0.4)
            model.light_dir[i, 1] += random.uniform(-0.4, 0.4)
            model.light_diffuse[i] = [random.uniform(0.3, 0.8) for _ in range(3)]
        model.vis.headlight.ambient[:] = [random.uniform(0.3, 0.7) for _ in range(3)]
        model.vis.headlight.diffuse[:] = [random.uniform(0.2, 0.6) for _ in range(3)]

    def _randomize_desktop_texture(self):
        chosen_texture = random.choice(self._TABLE_TEXTURES)
        mat_id = self._model.material(chosen_texture).id
        self._model.geom_matid[self._table_geom_id] = mat_id

    def _randomize_camera(self):
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _apply_random_camera_to_front(self, camera_idx: int) -> None:
        camera = self._camera_params[camera_idx]
        azimuth = float(camera[0])
        # Preset elevations use MuJoCo's free-camera sign, where negative looks down.
        elevation = float(-camera[1])
        distance = float(camera[2])
        azim_rad = np.deg2rad(azimuth)
        elev_rad = np.deg2rad(elevation)
        cam_offset = np.array(
            [
                -distance * np.cos(elev_rad) * np.cos(azim_rad),
                distance * np.cos(elev_rad) * np.sin(azim_rad),
                distance * np.sin(elev_rad),
            ],
            dtype=np.float64,
        )
        cam_pos = _SCENE_CENTER + cam_offset
        forward = -cam_offset
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # MuJoCo cameras look along their -Z axis with +Y up.
        rot_matrix = np.column_stack([right, up, -forward])
        cam_quat_wxyz = R.from_matrix(rot_matrix).as_quat(scalar_first=True)
        self._model.cam_pos[self._front_camera_id] = cam_pos
        self._model.cam_quat[self._front_camera_id] = cam_quat_wxyz
