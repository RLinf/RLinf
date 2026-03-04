# Copyright 2025 The RLinf Authors.
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

"""CubePick task for Genesis.

A Franka Panda arm must pick up a cube from a table.  The cube spawns at
a randomized (x, y) position and the episode succeeds when the cube's
z-coordinate exceeds a configurable height threshold.
"""

from __future__ import annotations

from typing import Any

import genesis as gs
import numpy as np
import torch

from rlinf.envs.genesis.tasks import register_task
from rlinf.envs.genesis.tasks.base import GenesisTaskBase
from rlinf.envs.genesis.utils import camera_render_rgb, extract_robot_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
_NUM_MOTOR_DOFS = 7
_NUM_FINGER_DOFS = 2
_HOME_QPOS = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04]

_CUBE_SIZE = (0.04, 0.04, 0.04)
_CUBE_DEFAULT_POS = (0.65, 0.0, 0.02)

_DEFAULT_SUCCESS_HEIGHT = 0.10

_CAMERA_POS = (3.5, 0.0, 2.5)
_CAMERA_LOOKAT = (0.0, 0.0, 0.5)
_CAMERA_FOV = 30


class CubePickTask(GenesisTaskBase):
    """Pick up a cube with a Franka Panda arm.

    Config fields consumed from ``cfg.init_params`` (all optional):

    * ``robot_file`` (str): Path to robot MJCF file.
      Default ``"xml/franka_emika_panda/panda.xml"``.
    * ``cube_size`` (list[float]): Cube half-extents ``[x, y, z]``.
    * ``cube_x_range`` (list[float]): Uniform sample range for cube x.
    * ``cube_y_range`` (list[float]): Uniform sample range for cube y.
    * ``success_height`` (float): z threshold for success.
    * ``camera_height`` (int): Camera resolution height.
    * ``camera_width`` (int): Camera resolution width.
    * ``dt`` (float): Simulation time-step.
    """

    task_description: str = "Pick up the cube from the table."

    def __init__(self) -> None:
        super().__init__()
        self.cube: Any = None
        self._rng: np.random.Generator | None = None
        self._success_height: float = _DEFAULT_SUCCESS_HEIGHT
        self._cube_x_range: tuple[float, float] = (0.45, 0.80)
        self._cube_y_range: tuple[float, float] = (-0.25, 0.25)

    # ------------------------------------------------------------------
    # GenesisTaskBase interface
    # ------------------------------------------------------------------

    def build_scene(self, scene, cfg) -> None:
        """Add Franka, cube, plane and camera to the scene."""
        init_params = cfg.get("init_params", {})

        # Ground plane
        scene.add_entity(gs.morphs.Plane())

        # Robot
        robot_file = init_params.get("robot_file", _FRANKA_MJCF)
        self.robot = scene.add_entity(gs.morphs.MJCF(file=robot_file))

        # Cube
        cube_size = tuple(init_params.get("cube_size", _CUBE_SIZE))
        self.cube = scene.add_entity(
            gs.morphs.Box(size=cube_size, pos=_CUBE_DEFAULT_POS),
        )

        # Camera
        cam_h = int(init_params.get("camera_height", 480))
        cam_w = int(init_params.get("camera_width", 640))
        cam_pos = tuple(init_params.get("camera_pos", _CAMERA_POS))
        cam_lookat = tuple(init_params.get("camera_lookat", _CAMERA_LOOKAT))
        cam_fov = float(init_params.get("camera_fov", _CAMERA_FOV))
        self.camera = scene.add_camera(
            res=(cam_w, cam_h),
            pos=cam_pos,
            lookat=cam_lookat,
            fov=cam_fov,
            GUI=False,
        )
        # Store camera base pose for per-env rendering in batched mode.
        self._camera_base_pos = cam_pos
        self._camera_base_lookat = cam_lookat

        # DOF indices
        self.motor_dofs = np.arange(_NUM_MOTOR_DOFS)
        self.finger_dofs = np.arange(_NUM_MOTOR_DOFS, _NUM_MOTOR_DOFS + _NUM_FINGER_DOFS)

        # Task-specific parameters
        self._success_height = float(init_params.get("success_height", _DEFAULT_SUCCESS_HEIGHT))
        self._cube_x_range = tuple(init_params.get("cube_x_range", self._cube_x_range))
        self._cube_y_range = tuple(init_params.get("cube_y_range", self._cube_y_range))

        # EEF link (resolved after scene.build via post_build)
        self._eef_link_name = init_params.get("eef_link_name", "hand")

    def post_build(self) -> None:
        """Called right after ``scene.build()`` to resolve link references."""
        self.eef_link = self.robot.get_link(self._eef_link_name)

        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def reset(
        self,
        scene,
        num_envs: int,
        envs_idx: torch.Tensor | None = None,
    ) -> None:
        """Randomize cube position and reset robot to home pose."""
        if self._rng is None:
            self._rng = np.random.default_rng()

        B = num_envs if envs_idx is None else len(envs_idx)

        # Randomize cube position
        x = self._rng.uniform(*self._cube_x_range, size=(B,))
        y = self._rng.uniform(*self._cube_y_range, size=(B,))
        z = np.full((B,), _CUBE_SIZE[2] / 2.0)
        cube_pos = torch.tensor(
            np.stack([x, y, z], axis=1), dtype=torch.float32, device=gs.device
        )
        cube_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * B, dtype=torch.float32, device=gs.device
        )

        # Robot home qpos
        qpos = torch.tensor(
            _HOME_QPOS, dtype=torch.float32, device=gs.device
        ).unsqueeze(0).repeat(B, 1)

        if envs_idx is not None:
            self.cube.set_pos(cube_pos, envs_idx=envs_idx)
            self.cube.set_quat(cube_quat, envs_idx=envs_idx)
            self.robot.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True)
            self.robot.control_dofs_position(qpos[:, :_NUM_MOTOR_DOFS], self.motor_dofs, envs_idx=envs_idx)
            self.robot.control_dofs_position(qpos[:, _NUM_MOTOR_DOFS:], self.finger_dofs, envs_idx=envs_idx)
        else:
            self.cube.set_pos(cube_pos)
            self.cube.set_quat(cube_quat)
            self.robot.set_qpos(qpos, zero_velocity=True)
            self.robot.control_dofs_position(qpos[:, :_NUM_MOTOR_DOFS], self.motor_dofs)
            self.robot.control_dofs_position(qpos[:, _NUM_MOTOR_DOFS:], self.finger_dofs)

    def compute_reward(
        self, scene, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dense shaped rewards for reaching, grasping, and lifting."""
        cube_pos = self.cube.get_pos()
        eef_pos = self.eef_link.get_pos()

        dist = torch.norm(eef_pos - cube_pos, p=2, dim=-1)
        reaching_reward = 1.0 - torch.tanh(10.0 * dist)

        initial_z = 0.02
        z_height = cube_pos[:, 2]
        is_close = dist < 0.05
        lifting_reward = torch.clamp(z_height - initial_z, min=0.0) * 10.0 * is_close.float()

        success = z_height > self._success_height
        success_bonus = success.float() * 20.0

        reward = reaching_reward + lifting_reward + success_bonus

        return reward, success.bool()

    def get_obs(self, scene, num_envs: int) -> dict[str, Any]:
        """Extract images and robot proprioceptive state."""
        images = camera_render_rgb(
            self.camera,
            num_envs,
            scene=scene,
            camera_base_pos=self._camera_base_pos,
            camera_base_lookat=self._camera_base_lookat,
        )
        robot_states = extract_robot_state(
            self.robot,
            self.eef_link,
            num_motor_dofs=_NUM_MOTOR_DOFS,
            num_finger_dofs=_NUM_FINGER_DOFS,
        )

        cube_pos = self.cube.get_pos()
        cube_quat = self.cube.get_quat()
        cube_states = torch.cat([cube_pos, cube_quat], dim=-1).float()
        cube_states = cube_states.to("cpu")

        states = torch.cat([robot_states, cube_states], dim=-1)

        return {
            "images": images,
            "states": states,
        }

    def seed(self, seed: int) -> None:
        """Set the task's random number generator seed."""
        self._rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task("cube_pick", CubePickTask)
