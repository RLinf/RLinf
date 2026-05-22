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

from __future__ import annotations

from typing import Any

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import transform_by_quat

from rlinf.envs.genesis.tasks import register_task
from rlinf.envs.genesis.tasks.base import GenesisTaskBase
from rlinf.envs.genesis.utils import camera_render_rgb, extract_robot_state

_FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
_NUM_MOTOR_DOFS = 7
_NUM_FINGER_DOFS = 2
_HOME_QPOS = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04]

_CUBE_SIZE = (0.04, 0.04, 0.04)
_CUBE_DEFAULT_POS = (0.65, 0.0, 0.02)

_DEFAULT_APPROACH_REWARD_SCALE = 1.0
_DEFAULT_APPROACH_REWARD_SHARPNESS = 6.0
_DEFAULT_GRASP_SUCCESS_REWARD = 100.0

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
    * ``camera_height`` (int): Camera resolution height.
    * ``camera_width`` (int): Camera resolution width.
    * ``dt`` (float): Simulation time-step.
    """

    task_description: str = "Pick up the cube from the table."

    def __init__(self) -> None:
        super().__init__()
        self.cube: Any = None
        self._rng: np.random.Generator | None = None
        self._cube_x_range: tuple[float, float] = (0.45, 0.80)
        self._cube_y_range: tuple[float, float] = (-0.25, 0.25)
        self._success_hold_steps: int = 3
        self._grasp_dist_threshold: float = 0.08
        self._success_hold_counter: torch.Tensor | None = None
        self._approach_reward_scale: float = _DEFAULT_APPROACH_REWARD_SCALE
        self._approach_reward_sharpness: float = _DEFAULT_APPROACH_REWARD_SHARPNESS
        self._grasp_success_reward: float = _DEFAULT_GRASP_SUCCESS_REWARD

    def build_scene(self, scene, cfg) -> None:
        """Add Franka, cube, plane and camera to the scene."""
        init_params = cfg.get("init_params", {})

        scene.add_entity(gs.morphs.Plane())

        scene.add_light(
            pos=(2, 2, 2),
            dir=(-1, -1, -1),
            color=(
                1,
                1,
                1,
            ),
            intensity=1.0,
            directional=False,
            castshadow=True,
            cutoff=80.0,
        )

        robot_file = init_params.get("robot_file", _FRANKA_MJCF)
        self.robot = scene.add_entity(gs.morphs.MJCF(file=robot_file))
        contact_draw_debug = bool(init_params.get("contact_draw_debug", False))
        self.lf_sensor = scene.add_sensor(
            gs.sensors.Contact(
                entity_idx=self.robot.idx,
                draw_debug=contact_draw_debug,
                link_idx_local=self.robot.get_link("left_finger").idx_local,
            )
        )
        self.rf_sensor = scene.add_sensor(
            gs.sensors.Contact(
                entity_idx=self.robot.idx,
                draw_debug=contact_draw_debug,
                link_idx_local=self.robot.get_link("right_finger").idx_local,
            )
        )

        cube_size = tuple(init_params.get("cube_size", _CUBE_SIZE))
        self.cube = scene.add_entity(
            gs.morphs.Box(size=cube_size, pos=_CUBE_DEFAULT_POS),
        )

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
        self._camera_base_pos = cam_pos
        self._camera_base_lookat = cam_lookat

        self.motor_dofs = np.arange(_NUM_MOTOR_DOFS)
        self.finger_dofs = np.arange(
            _NUM_MOTOR_DOFS, _NUM_MOTOR_DOFS + _NUM_FINGER_DOFS
        )

        self._success_hold_steps = int(init_params.get("success_hold_steps", 3))
        self._grasp_dist_threshold = float(
            init_params.get("grasp_dist_threshold", 0.08)
        )
        self._approach_reward_scale = float(
            init_params.get("approach_reward_scale", _DEFAULT_APPROACH_REWARD_SCALE)
        )
        self._approach_reward_sharpness = float(
            init_params.get(
                "approach_reward_sharpness",
                _DEFAULT_APPROACH_REWARD_SHARPNESS,
            )
        )
        self._grasp_success_reward = float(
            init_params.get("grasp_success_reward", _DEFAULT_GRASP_SUCCESS_REWARD)
        )
        self._cube_x_range = tuple(init_params.get("cube_x_range", self._cube_x_range))
        self._cube_y_range = tuple(init_params.get("cube_y_range", self._cube_y_range))

        self._eef_link_name = init_params.get("eef_link_name", "hand")
        self._eef_offset = torch.tensor([0.0, 0.0, 0.11], device=gs.device)

    def post_build(self) -> None:
        """Called right after ``scene.build()`` to resolve link references."""
        self.eef_link = self.robot.get_link(self._eef_link_name)

        self.robot.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        )
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
        reset_state_ids: torch.Tensor | None = None,
    ) -> None:
        """Randomize cube position and reset robot to home pose."""
        if self._rng is None:
            self._rng = np.random.default_rng()
        self._ensure_episode_buffers(num_envs)

        B = num_envs if envs_idx is None else len(envs_idx)

        # Randomize cube position (or deterministically map from state ids).
        if reset_state_ids is None:
            x = self._rng.uniform(*self._cube_x_range, size=(B,))
            y = self._rng.uniform(*self._cube_y_range, size=(B,))
        else:
            ids_np = reset_state_ids.detach().cpu().numpy().astype(np.int64)
            if ids_np.shape[0] != B:
                raise ValueError(
                    "reset_state_ids length must match number of envs being reset."
                )

            x_low, x_high = self._cube_x_range
            y_low, y_high = self._cube_y_range

            # Hash-like deterministic projection from integer state ids to [0, 1).
            x_u = ((ids_np * 1103515245 + 12345) % (2**31)) / float(2**31)
            y_u = ((ids_np * 1664525 + 1013904223) % (2**31)) / float(2**31)
            x = x_low + x_u * (x_high - x_low)
            y = y_low + y_u * (y_high - y_low)
        z = np.full((B,), _CUBE_SIZE[2] / 2.0)
        cube_pos = torch.tensor(
            np.stack([x, y, z], axis=1), dtype=torch.float32, device=gs.device
        )
        cube_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * B, dtype=torch.float32, device=gs.device
        )

        # Robot home qpos
        qpos = (
            torch.tensor(_HOME_QPOS, dtype=torch.float32, device=gs.device)
            .unsqueeze(0)
            .repeat(B, 1)
        )

        if envs_idx is not None:
            self.cube.set_pos(cube_pos, envs_idx=envs_idx)
            self.cube.set_quat(cube_quat, envs_idx=envs_idx)
            self.robot.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True)
            self.robot.control_dofs_position(
                qpos[:, :_NUM_MOTOR_DOFS], self.motor_dofs, envs_idx=envs_idx
            )
            self.robot.control_dofs_position(
                qpos[:, _NUM_MOTOR_DOFS:], self.finger_dofs, envs_idx=envs_idx
            )
            assert self._success_hold_counter is not None
            self._success_hold_counter[envs_idx] = 0
        else:
            self.cube.set_pos(cube_pos)
            self.cube.set_quat(cube_quat)
            self.robot.set_qpos(qpos, zero_velocity=True)
            self.robot.control_dofs_position(qpos[:, :_NUM_MOTOR_DOFS], self.motor_dofs)
            self.robot.control_dofs_position(
                qpos[:, _NUM_MOTOR_DOFS:], self.finger_dofs
            )
            assert self._success_hold_counter is not None
            self._success_hold_counter.zero_()

    def _ensure_episode_buffers(self, num_envs: int) -> None:
        if (
            self._success_hold_counter is None
            or self._success_hold_counter.shape[0] != num_envs
        ):
            self._success_hold_counter = torch.zeros(
                num_envs, dtype=torch.int32, device=gs.device
            )

    def _compute_task_signals(
        self, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cube_pos = self.cube.get_pos()

        eef_pos = self.eef_link.get_pos()
        eef_quat = self.eef_link.get_quat()
        eef_pos += transform_by_quat((self._eef_offset).repeat((num_envs, 1)), eef_quat)

        gripper = self.robot.get_dofs_position()[
            ..., _NUM_MOTOR_DOFS : _NUM_MOTOR_DOFS + _NUM_FINGER_DOFS
        ]

        dist = torch.norm(eef_pos - cube_pos, p=2, dim=-1)
        z_height = cube_pos[:, 2]

        lf_contact = self.lf_sensor.read().squeeze()
        rf_contact = self.rf_sensor.read().squeeze()

        # Franka finger joints are usually positive when the gripper is open.
        # During a stable grasp, fingers are not fully open.
        gripper_not_fully_open = (gripper[:, 0] < 0.04) & (gripper[:, 1] < 0.04)
        is_grasped = lf_contact & rf_contact & gripper_not_fully_open
        is_contact = lf_contact | rf_contact
        return dist, z_height, is_grasped, is_contact

    def compute_reward(self, scene, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reward: approach bonus + large success bonus on stable grasp."""
        dist, _z_height, is_grasped, _is_contact = self._compute_task_signals(num_envs)
        is_close = dist < self._grasp_dist_threshold
        success_instant = is_grasped & is_close

        # Reward 1: nonlinear exponential bonus for approaching the cube.
        approach_reward = self._approach_reward_scale * torch.exp(
            -self._approach_reward_sharpness * dist
        )
        # Reward 2: very large bonus when the gripper stably grasps the cube.
        grasp_success_reward = self._grasp_success_reward * success_instant.float()

        reward = approach_reward + grasp_success_reward
        return reward, success_instant.bool()

    def compute_step_outcomes(
        self,
        scene,
        num_envs: int,
        elapsed_steps: torch.Tensor,
        max_episode_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute reward and robust success signals for one step."""
        self._ensure_episode_buffers(num_envs)
        assert self._success_hold_counter is not None

        reward, success_instant = self.compute_reward(scene, num_envs)
        self._success_hold_counter = torch.where(
            success_instant,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= self._success_hold_steps
        terminations = success.bool()
        truncations = elapsed_steps >= max_episode_steps

        dist, z_height, is_grasped, is_contact = self._compute_task_signals(num_envs)
        infos: dict[str, Any] = {
            "success": success,
            "success_instant": success_instant,
            "success_hold_counter": self._success_hold_counter.clone(),
            "is_grasped": is_grasped,
            "is_contact": is_contact,
            "z_height": z_height,
            "eef_cube_dist": dist,
        }
        return reward, terminations, truncations, infos

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


register_task("cube_pick", CubePickTask)
