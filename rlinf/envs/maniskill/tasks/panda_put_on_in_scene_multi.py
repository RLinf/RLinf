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

import os
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BRIDGE_DATASET_ASSET_PATH,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from sapien.physx import PhysxMaterial
from transforms3d.euler import euler2quat, mat2euler

from rlinf.envs.maniskill.tasks.panda_table_agent import PandaBridgeDatasetFlatTable
from rlinf.envs.maniskill.tasks.pose_utils import (
    matrix2pose_batch_torch,
    pose2matrix_batch_torch,
    pose2matrix_torch,
)

CARROT_DATASET_DIR = (
    Path(os.getenv("MANISKILL_ASSET_DIR", Path(__file__).parent / ".." / "assets"))
    / "carrot"
)
CUSTOM_DATASET_PATH = Path(
    Path(os.getenv("MANISKILL_ASSET_DIR", Path(__file__).parent / ".." / "assets"))
    / "custom_assets"
)

rotation_offset_dict = {
    0: 0,
    1: np.pi / 4,
    2: 0,
    3: -np.pi / 4,
    4: -np.pi / 4,
    5: 0,
    6: np.pi / 4,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: np.pi / 4,
    17: 0,
    18: 0,
    19: -np.pi / 4,
    20: -np.pi / 4,
    21: 0,
    22: np.pi / 4,
    23: np.pi / 4,
    24: 0,
}


class PandaPutOnPlateInScene25(BaseEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_OBS_MODES = ["rgb+segmentation", "state"]
    SUPPORTED_REWARD_MODES = ["none", "dense", "normalized_dense"]

    obj_static_friction = 1.0
    obj_dynamic_friction = 1.0

    rgb_camera_name: str = "3rd_view_camera"
    rgb_overlay_mode: str = (
        "background"  # 'background' or 'object' or 'debug' or combinations of them
    )

    overlay_images_numpy: list[np.ndarray]
    overlay_textures_numpy: list[np.ndarray]
    overlay_mix_numpy: list[float]
    overlay_images: torch.Tensor
    overlay_textures: torch.Tensor
    overlay_mix: torch.Tensor
    model_db_carrot: dict[str, dict]
    model_db_plate: dict[str, dict]
    carrot_names: list[str]
    plate_names: list[str]
    select_carrot_ids: torch.Tensor
    select_plate_ids: torch.Tensor
    select_overlay_ids: torch.Tensor
    select_pos_ids: torch.Tensor
    select_quat_ids: torch.Tensor

    initial_qpos: np.ndarray
    initial_robot_pos: sapien.Pose
    safe_robot_pos: sapien.Pose
    fix_obj_xyz: Optional[torch.Tensor] = None
    fix_obj_quat: Optional[torch.Tensor] = None
    fix_plate_xyz: Optional[torch.Tensor] = None
    fix_plate_quat: Optional[torch.Tensor] = None

    def __init__(self, **kwargs):
        # random pose
        self._generate_init_pose()

        self.initial_qpos = np.array(
            [0, 0.259, 0, -2.289, 0, 2.515, np.pi / 4, 0.04, 0.015]
        )
        self.initial_robot_pos = sapien.Pose([0.3, 0.028, 0.870], q=[0, 0, 0, 1])
        self.safe_robot_pos = sapien.Pose([0.3, 0.028, 1.870], q=[0, 0, 0, 1])

        # stats
        self.extra_stats = {}

        super().__init__(robot_uids=PandaBridgeDatasetFlatTable, **kwargs)

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = (
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.2],
                    [0.0, 0.4],
                    [0.0, 0.6],
                    [0.0, 0.8],
                    [0.0, 1.0],
                    [0.2, 0.0],
                    [0.2, 0.2],
                    [0.2, 0.4],
                    [0.2, 0.6],
                    [0.2, 0.8],
                    [0.2, 1.0],
                    [0.4, 0.0],
                    [0.4, 0.2],
                    [0.4, 0.4],
                    [0.4, 0.6],
                    [0.4, 0.8],
                    [0.4, 1.0],
                    [0.6, 0.0],
                    [0.6, 0.2],
                    [0.6, 0.4],
                    [0.6, 0.6],
                    [0.6, 0.8],
                    [0.6, 1.0],
                    [0.8, 0.0],
                    [0.8, 0.2],
                    [0.8, 0.4],
                    [0.8, 0.6],
                    [0.8, 0.8],
                    [0.8, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.2],
                    [1.0, 0.4],
                    [1.0, 0.6],
                    [1.0, 0.8],
                    [1.0, 1.0],
                ]
            )
            * 2
            - 1
        )  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    def _build_actor_helper(
        self, name: str, path: Path, density: float, scale: float, pose: Pose
    ):
        """helper function to build actors by ID directly and auto configure physical materials"""
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()

        collision_file = str(path / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(path / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(path / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(path / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        builder.initial_pose = pose
        actor = builder.build(name=name)
        return actor

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()  # Warning should be dissmissed, for we set the initial pose below -> actor.set_pose
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])

        scene_file = str(CUSTOM_DATASET_PATH / "bridge_table_long.glb")
        print("path:", BRIDGE_DATASET_ASSET_PATH)

        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        # models
        self.model_bbox_sizes = {}

        # carrot
        self.objs_carrot: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_carrot):
            model_path = CARROT_DATASET_DIR / "more_carrot" / name
            density = self.model_db_carrot[name].get("density", 1000)
            scale_list = self.model_db_carrot[name].get("scale", [1.0])
            bbox = self.model_db_carrot[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([1.0, 0.3 * idx, 1.0]))
            self.objs_carrot[name] = self._build_actor_helper(
                name, model_path, density, scale, pose
            )

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(
                bbox_size * scale, device=self.device
            )  # [3]

        # plate
        self.objs_plate: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_plate):
            model_path = CARROT_DATASET_DIR / "more_plate" / name
            density = self.model_db_plate[name].get("density", 1000)
            scale_list = self.model_db_plate[name].get("scale", [1.0])
            bbox = self.model_db_plate[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([2.0, 0.3 * idx, 1.0]))
            self.objs_plate[name] = self._build_actor_helper(
                name, model_path, density, scale, pose
            )

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(
                bbox_size * scale, device=self.device
            )  # [3]

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack(
            [self.overlay_images_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_images = torch.tensor(
            overlay_images, device=self.device
        )  # [b, H, W, 3]
        overlay_textures = np.stack(
            [self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_textures = torch.tensor(
            overlay_textures, device=self.device
        )  # [b, H, W, 3]
        overlay_mix = np.array(
            [self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids]
        )
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0],
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = (
                torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            if self.fix_obj_xyz is not None and idx == self.select_carrot_ids[0].item():
                p_select = self.fix_obj_xyz[0].reshape(1, -1).repeat(b, 1)  # [b, 3]
            else:
                p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            if (
                self.fix_obj_quat is not None
                and idx == self.select_carrot_ids[0].item()
            ):
                q_select_before = self.fix_obj_quat[0].reshape(1, -1).repeat(b, 1)
            else:
                q_select_before = quat_configs[self.select_quat_ids, 0].reshape(
                    b, 4
                )  # [b, 4]
            q_select = q_select_before.clone()
            q_select[:, 0] = q_select_before[:, 0] * np.cos(
                rotation_offset_dict[idx]
            ) - q_select_before[:, 3] * np.sin(rotation_offset_dict[idx])
            q_select[:, 3] = q_select_before[:, 0] * np.sin(
                rotation_offset_dict[idx]
            ) + q_select_before[:, 3] * np.cos(rotation_offset_dict[idx])
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = (
                torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            if self.fix_plate_xyz is not None:
                p_select = self.fix_plate_xyz[idx].reshape(1, -1).repeat(b, 1)
            else:
                p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            if self.fix_plate_quat is not None:
                q_select = self.fix_plate_quat[idx].reshape(1, -1).repeat(b, 1)
            else:
                q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(carrot_actor)]
        )  # [b, 4]
        self.plate_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(plate_actor)]
        )  # [b, 4]
        corner_signs = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            device=self.device,
        )

        # carrot
        carrot_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_carrot]
        )  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.carrot_q_after_settle
        )  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(
            c_bbox_corners, c_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        c_rotated_bbox_size = (
            c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_plate]
        )  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.plate_q_after_settle
        )  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(
            p_bbox_corners, p_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        p_rotated_bbox_size = (
            p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros(
            (b,), dtype=torch.int32, device=self.device
        )
        self.episode_stats = {
            "is_src_obj_grasped": torch.zeros(
                (b,), dtype=torch.bool, device=self.device
            ),
            "consecutive_grasp": torch.zeros(
                (b,), dtype=torch.bool, device=self.device
            ),
            "src_on_target": torch.zeros((b,), dtype=torch.bool, device=self.device),
            "gripper_carrot_dist": torch.zeros(
                (b,), dtype=torch.float32, device=self.device
            ),
            "gripper_plate_dist": torch.zeros(
                (b,), dtype=torch.float32, device=self.device
            ),
            "carrot_plate_dist": torch.zeros(
                (b,), dtype=torch.float32, device=self.device
            ),
        }
        self.extra_stats = {}

        # save init plate pos
        self.init_plate_pos = self.objs_plate["001_plate_simpler"].pose.p

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _settle(self, t=0.5):
        """run the simulation for some steps to help settle the objects"""
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    def evaluate(self, success_require_src_completely_on_target=True):
        xy_flag_required_offset = 0.01
        z_flag_required_offset = 0.05
        netforce_flag_required_offset = 0.03

        b = self.num_envs

        # actor
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        carrot_p = torch.stack(
            [a.pose.p[idx] for idx, a in enumerate(carrot_actor)]
        )  # [b, 3]
        carrot_q = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(carrot_actor)]
        )  # [b, 4]
        plate_p = torch.stack(
            [a.pose.p[idx] for idx, a in enumerate(plate_actor)]
        )  # [b, 3]
        plate_q = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(plate_actor)]
        )  # [b, 4]

        is_src_obj_grasped = torch.zeros(
            (b,), dtype=torch.bool, device=self.device
        )  # [b]

        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            grasped = self.agent.is_grasping(self.objs_carrot[name])  # [b]
            is_src_obj_grasped = torch.where(
                is_select, grasped, is_src_obj_grasped
            )  # [b]

        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.plate_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.carrot_bbox_world / 2

        pos_src = carrot_p
        pos_tgt = plate_p
        offset = pos_src - pos_tgt
        xy_flag = (
            torch.linalg.norm(offset[:, :2], dim=1)
            <= tgt_obj_half_length_bbox.max(dim=1).values + xy_flag_required_offset
        )
        z_flag = (offset[:, 2] > 0) & (
            offset[:, 2]
            - tgt_obj_half_length_bbox[:, 2]
            - src_obj_half_length_bbox[:, 2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            net_forces = torch.zeros(
                (b,), dtype=torch.float32, device=self.device
            )  # [b]
            for idx in range(self.num_envs):
                force = self.scene.get_pairwise_contact_forces(
                    self.objs_carrot[select_carrot[idx]],
                    self.objs_plate[select_plate[idx]],
                )[idx]
                force = torch.linalg.norm(force)
                net_forces[idx] = force

            src_on_target = src_on_target & (net_forces > netforce_flag_required_offset)

        success = src_on_target

        # prepare dist
        gripper_p = (
            self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p
        ) / 2  # [b, 3]
        gripper_q = (
            self.agent.finger1_link.pose.q + self.agent.finger2_link.pose.q
        ) / 2  # [b, 4]
        gripper_carrot_dist = torch.linalg.norm(gripper_p - carrot_p, dim=1)  # [b, 3]
        gripper_plate_dist = torch.linalg.norm(gripper_p - plate_p, dim=1)  # [b, 3]
        carrot_plate_dist = torch.linalg.norm(carrot_p - plate_p, dim=1)  # [b, 3]

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )
        self.episode_stats["is_src_obj_grasped_current"] = is_src_obj_grasped
        self.episode_stats["gripper_carrot_dist"] = gripper_carrot_dist
        self.episode_stats["gripper_plate_dist"] = gripper_plate_dist
        self.episode_stats["carrot_plate_dist"] = carrot_plate_dist

        self.extra_stats["extra_pos_carrot"] = carrot_p
        self.extra_stats["extra_q_carrot"] = carrot_q
        self.extra_stats["extra_pos_plate"] = plate_p
        self.extra_stats["extra_q_plate"] = plate_q
        self.extra_stats["extra_pos_gripper"] = gripper_p
        self.extra_stats["extra_q_gripper"] = gripper_q

        return dict(**self.episode_stats, success=success)

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True

    def get_language_instruction(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        instruct = []
        for idx in range(self.num_envs):
            carrot_name = self.model_db_carrot[select_carrot[idx]]["name"]
            plate_name = self.model_db_plate[select_plate[idx]]["name"]
            instruct.append(f"put {carrot_name} on {plate_name}")

        return instruct

    def _after_reconfigure(self, options: dict):
        target_object_actor_ids = [
            x._objs[0].per_scene_id
            for x in self.scene.actors.values()
            if x.name not in ["ground", "goal_site", "", "arena"]
        ]
        self.target_object_actor_ids = torch.tensor(
            target_object_actor_ids, dtype=torch.int16, device=self.device
        )
        # get the robot link ids
        robot_links = self.agent.robot.get_links()
        self.robot_link_ids = torch.tensor(
            [x._objs[0].entity.per_scene_id for x in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

    def _green_sceen_rgb(
        self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix
    ):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(
                actor_seg.device
            )

        mask = torch.isin(
            actor_seg, torch.concat([self.robot_link_ids, self.target_object_actor_ids])
        )
        mask = (~mask).to(torch.float32)  # [b, H, W]

        mask = mask.unsqueeze(-1)  # [b, H, W, 1]

        # perform overlay on the RGB observation image
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        rgb = rgb.to(torch.float32)  # [b, H, W, 3]

        rgb_ret = overlay_img * mask  # [b, H, W, 3]
        rgb_ret += rgb * (1 - mask)  # [b, H, W, 3]

        rgb_ret = torch.clamp(rgb_ret, 0, 255)  # [b, H, W, 3]
        rgb_ret = rgb_ret.to(torch.uint8)  # [b, H, W, 3]

        return rgb_ret

    def _get_obs_extra(self, info: dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""

        # One hot for carrot ids
        b = self.select_carrot_ids.shape[0]
        carrot_one_hot = torch.zeros((b, 25), dtype=torch.float32, device=self.device)
        carrot_one_hot[torch.arange(b), self.select_carrot_ids] = 1.0

        # gripper pose
        gripper_pose = self.agent.tcp.pose.raw_pose

        pose_bank = torch.stack(
            [self.objs_carrot[name].pose.raw_pose for name in self.carrot_names], dim=0
        )
        object_poses = pose_bank[self.select_carrot_ids, torch.arange(b)]

        plate_poses = self.objs_plate["001_plate_simpler"].pose.raw_pose

        return {
            "carrot_one_hot": carrot_one_hot,  # [b, 25]
            "gripper_pose": gripper_pose,  # [b, 7]
            "object_poses": object_poses,  # [b, 7]
            "plate_poses": plate_poses,  # [b, 7]
            "init_plate_pos": self.init_plate_pos,  # [b, 3]
        }

    # panda
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose(
            [0.442614, 0.488839, 1.45059], [0.39519, 0.210508, 0.0936785, -0.889233]
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.45, 0.1, 1000)

    def _is_success(self, object_poses: torch.Tensor) -> torch.Tensor:
        """Check if the task is successful, should return a boolean tensor of shape (b,)"""
        delta_pos = self.objs_plate["001_plate_simpler"].pose.p - object_poses[:, :3]
        success_check = (torch.linalg.norm(delta_pos[:, :2], dim=1) < 0.05) & (
            torch.abs(delta_pos[:, 2]) < 0.05
        )

        return success_check

    def _is_lifted(self, object_poses: torch.Tensor) -> torch.Tensor:
        b = object_poses.shape[0]
        grasping_bank = torch.stack(
            [
                self.agent.is_grasping(self.objs_carrot[name])
                for name in self.carrot_names
            ],
            dim=0,
        )
        is_lifted = grasping_bank[self.select_carrot_ids, torch.arange(b)]
        return is_lifted

    def compute_approaching_reward(self, object_poses: torch.Tensor) -> torch.Tensor:
        tcp_poses = self.agent.tcp.pose.raw_pose
        delta_pos = object_poses[:, :3] - tcp_poses[:, :3]
        dist = torch.linalg.norm(delta_pos, dim=1)

        approaching_reward = 1 - torch.tanh(dist * 5)
        return approaching_reward

    def putting_reward(self, object_poses: torch.Tensor) -> torch.Tensor:
        delta_pos = self.objs_plate["001_plate_simpler"].pose.p - object_poses[:, :3]
        put_reward = 1 - torch.tanh(torch.linalg.norm(delta_pos, dim=1) * 5)
        return put_reward

    # give minux reward when plate being moved
    def moving_penalty(self) -> torch.Tensor:
        plate_pos = self.objs_plate["001_plate_simpler"].pose.p
        init_plate_pos = self.init_plate_pos

        delta_pos = plate_pos - init_plate_pos
        dist = torch.linalg.norm(delta_pos, dim=1)

        moving_penalty = torch.tanh(dist * 5)
        return moving_penalty

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        b = self.select_carrot_ids.shape[0]

        pose_bank = torch.stack(
            [self.objs_carrot[name].pose.raw_pose for name in self.carrot_names], dim=0
        )
        object_poses = pose_bank[self.select_carrot_ids, torch.arange(b)]

        success_flag = info["success"]

        rewards = self.compute_approaching_reward(object_poses)
        is_lifted = self._is_lifted(object_poses)

        success_flag = success_flag & (
            ~is_lifted
        )  # only give reward when the object is lifted

        put_rewards = self.putting_reward(object_poses)
        put_rewards[~is_lifted] = 0.0  # only give put reward when the object is lifted
        rewards += is_lifted.float() + put_rewards

        rewards[success_flag] = 5.0

        moving_penalty = self.moving_penalty()
        rewards -= moving_penalty * 3

        return rewards

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return (
            self.compute_dense_reward(obs, action, info) / 5.0
        )  # normalize the reward to [0, 1]


@register_env(
    "PandaPutOnPlateInScene25DigitalTwin-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PandaPutOnPlateInScene25DigitalTwin(PandaPutOnPlateInScene25):
    def __init__(self, use_sparse_reward=False, **kwargs):
        self._prep_init()
        self.use_sparse_reward = use_sparse_reward

        super().__init__(**kwargs)

        if self.use_sparse_reward:
            self.is_successed_flag = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self.successed_but_failed_flag = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self.is_grasped_flag = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {
            k: v for k, v in self.model_db_plate.items() if k == only_plate_name
        }
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(
                cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480)
            )
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(
                cv2.cvtColor(
                    cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB
                ),
                (640, 480),
            )
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"]
            for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)

        # -----------------------------
        # object / config space
        # -----------------------------
        # In this environment, we do not split objects into train set and eval set
        obj_select_set = list(range(25))
        lc = len(obj_select_set)
        lc_offset = 0
        lo = 5
        lo_offset = 0

        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # -----------------------------
        # episode id sampling (batch-level)
        # -----------------------------
        episode_id = options.get(
            "episode_id", torch.randint(low=0, high=ltt, size=(b,), device=self.device)
        ).reshape(b)

        episode_id_add = torch.randint(low=0, high=ltt, size=(b,), device=self.device)
        episode_id = (episode_id + episode_id_add) % ltt

        # -----------------------------
        # lazy initialization of buffers
        # -----------------------------
        if not hasattr(self, "select_carrot_ids"):
            self.select_carrot_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.select_plate_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.select_overlay_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.select_pos_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.select_quat_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

        # -----------------------------
        # decode episode id (batch-level)
        # -----------------------------
        carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset
        carrot_ids = torch.tensor(
            [obj_select_set[i] for i in carrot_ids], device=self.device
        )

        plate_ids = (episode_id // (lo * l1 * l2)) % lp
        overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        pos_ids = (episode_id // l2) % l1
        quat_ids = episode_id % l2

        # -----------------------------
        # write back ONLY selected envs
        # -----------------------------
        self.select_carrot_ids[env_idx] = carrot_ids
        self.select_plate_ids[env_idx] = plate_ids
        self.select_overlay_ids[env_idx] = overlay_ids
        self.select_pos_ids[env_idx] = pos_ids
        self.select_quat_ids[env_idx] = quat_ids

    def get_language_instruction(self):
        lang = "Pick up the object on the table and place it into the white tray."
        return [lang] * self.num_envs

    def _generate_init_pose(self):
        object_xy_center = np.array([-0.098, -0.165]).reshape(1, 2)
        object_half_edge_length = np.array([0.1, 0.1]).reshape(1, 2)
        plate_xy_center = np.array([-0.148, 0.06]).reshape(1, 2)
        plate_half_edge_length = np.array([0.05, 0.075]).reshape(1, 2)

        grid_pos = (
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.2],
                    [0.0, 0.4],
                    [0.0, 0.6],
                    [0.0, 0.8],
                    [0.0, 1.0],
                    [0.2, 0.0],
                    [0.2, 0.2],
                    [0.2, 0.4],
                    [0.2, 0.6],
                    [0.2, 0.8],
                    [0.2, 1.0],
                    [0.4, 0.0],
                    [0.4, 0.2],
                    [0.4, 0.4],
                    [0.4, 0.6],
                    [0.4, 0.8],
                    [0.4, 1.0],
                    [0.6, 0.0],
                    [0.6, 0.2],
                    [0.6, 0.4],
                    [0.6, 0.6],
                    [0.6, 0.8],
                    [0.6, 1.0],
                    [0.8, 0.0],
                    [0.8, 0.2],
                    [0.8, 0.4],
                    [0.8, 0.6],
                    [0.8, 0.8],
                    [0.8, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.2],
                    [1.0, 0.4],
                    [1.0, 0.6],
                    [1.0, 0.8],
                    [1.0, 1.0],
                ]
            )
            * 2
            - 1
        )  # [36, 2]
        object_grid_pos = grid_pos * object_half_edge_length + object_xy_center
        plate_grid_pos = grid_pos * plate_half_edge_length + plate_xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(object_grid_pos):
            for j, grid_pos_2 in enumerate(plate_grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, 0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, -np.pi / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

    def change_the_frame_torch(
        self, original_euler: torch.Tensor, tcp_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert euler delta in root frame to TCP frame (torch version).

        Args:
            original_euler: (3,) or (N, 3), euler angles (xyz) in root frame
            tcp_pose: (7,) or (N, 7), pose(s) in format [x,y,z,w,x,y,z]

        Returns:
            delta_euler_root: (3,) if input single, or (N, 3) if batch
        """
        tcp_pose = tcp_pose.to(dtype=original_euler.dtype)

        # --- Step 0: normalize shapes ---
        if original_euler.ndim == 1:
            original_euler = original_euler.unsqueeze(0)
        if tcp_pose.ndim == 1:
            tcp_pose = tcp_pose.unsqueeze(0)
        assert original_euler.shape[0] == tcp_pose.shape[0], (
            "Batch size mismatch between euler and tcp_pose"
        )

        N = tcp_pose.shape[0]

        # --- Step 1: convert euler (xyz) to rotation matrices ---
        cx, cy, cz = (
            torch.cos(original_euler[:, 0]),
            torch.cos(original_euler[:, 1]),
            torch.cos(original_euler[:, 2]),
        )
        sx, sy, sz = (
            torch.sin(original_euler[:, 0]),
            torch.sin(original_euler[:, 1]),
            torch.sin(original_euler[:, 2]),
        )

        # R = Rz * Ry * Rx  (consistent with scipy 'xyz')
        delta_rot_tcp = torch.stack(
            [
                torch.stack(
                    [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy], dim=-1
                ),
                torch.stack(
                    [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx], dim=-1
                ),
                torch.stack([-sy, cy * sx, cx * cy], dim=-1),
            ],
            dim=1,
        )  # (N, 3, 3)

        # --- Step 2: convert tcp quaternion (wxyz) to rotation matrix ---
        quat_wxyz = tcp_pose[:, 3:]
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # [x,y,z,w]
        qx, qy, qz, qw = (
            quat_xyzw[:, 0],
            quat_xyzw[:, 1],
            quat_xyzw[:, 2],
            quat_xyzw[:, 3],
        )
        norm = torch.sqrt(qx * qx + qy * qy + qz * qz + qw * qw + 1e-8)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        tcp_rot = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qz * qw),
                        2 * (qx * qz + qy * qw),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qx * qy + qz * qw),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qx * qw),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qx * qz - qy * qw),
                        2 * (qy * qz + qx * qw),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                    dim=-1,
                ),
            ],
            dim=1,
        )  # (N, 3, 3)

        # --- Step 3: coordinate transformation ---
        tcp_rot_inv = tcp_rot.transpose(1, 2)
        delta_rot_root = tcp_rot @ delta_rot_tcp @ tcp_rot_inv  # (N,3,3)

        # --- Step 4: convert back to euler (xyz) ---
        # Handle possible numerical errors with atan2
        sy = -delta_rot_root[:, 2, 0]
        cy = torch.sqrt(delta_rot_root[:, 0, 0] ** 2 + delta_rot_root[:, 1, 0] ** 2)
        singular = cy < 1e-6

        euler_x = torch.atan2(delta_rot_root[:, 2, 1], delta_rot_root[:, 2, 2])
        euler_y = torch.atan2(sy, cy)
        euler_z = torch.atan2(delta_rot_root[:, 1, 0], delta_rot_root[:, 0, 0])

        # Handle singularities (gimbal lock)
        if singular.any():
            euler_x[singular] = torch.atan2(
                -delta_rot_root[singular, 1, 2], delta_rot_root[singular, 1, 1]
            )
            euler_y[singular] = torch.atan2(sy[singular], cy[singular])
            euler_z[singular] = 0.0

        delta_euler_root = torch.stack([euler_x, euler_y, euler_z], dim=-1)

        if N == 1:
            return delta_euler_root[0]
        return delta_euler_root

    def reset(
        self,
        seed: Union[None, int, list[int]] = None,
        options: Union[None, dict] = None,
    ):
        self.step_cnt = 0

        self.is_successed_flag = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.successed_but_failed_flag = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.is_grasped_flag = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

        raw_obs, infos = super().reset(seed, options)

        obs_image = raw_obs["sensor_data"]["c19_front_view"]["rgb"].to(torch.uint8)
        gripper_state = (
            self.unwrapped.agent.robot.get_qpos().to(torch.float32)[:, -1:] * 2
        )

        ee_pose_T = (
            self.unwrapped.agent.ee_pose_at_robot_base.to_transformation_matrix()
            .cpu()
            .numpy()
        )  # (num_envs, 4, 4)

        pos = ee_pose_T[:, :3, 3]  # (num_envs, 3)
        euler = np.stack(
            [mat2euler(ee_pose_T[i, :3, :3], "sxyz") for i in range(self.num_envs)],
            axis=0,
        )  # (num_envs, 3)

        pos = torch.from_numpy(pos).to(gripper_state.device)
        euler = torch.from_numpy(euler).to(gripper_state.device)

        proprioception = torch.cat([pos, euler, gripper_state], dim=1)  # (num_envs, 7)

        infos["extracted_obs"] = {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.get_language_instruction(),
        }

        return raw_obs, infos

    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        if self._control_mode != "pd_ee_body_target_delta_pose_real_root_frame":
            raw_obs, _reward, terminations, truncations, infos = super().step(action)
        else:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(self.device)

            pose_tcp_in_world = self.agent.tcp.pose.raw_pose  # (7,) or (N,7)
            pose_tcp_in_world_mat = pose2matrix_batch_torch(
                pose_tcp_in_world
            )  # (4,4) or (N,4,4)

            # root pose in the world frame
            pose_root_in_world_mat = pose2matrix_torch(
                torch.tensor(
                    [0.3, 0.028, -0.870, 0, 0, 0, 1],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            pose_root_in_world_inv = torch.linalg.inv(pose_root_in_world_mat)
            pose_tcp_in_root_mat = pose_root_in_world_inv @ pose_tcp_in_world_mat
            current_tcp_pose = matrix2pose_batch_torch(pose_tcp_in_root_mat)

            new_action = action.clone()
            new_action[:, 3:6] = self.change_the_frame_torch(
                action[:, 3:6], current_tcp_pose
            )

            raw_obs, _reward, terminations, truncations, infos = super().step(
                new_action
            )

        obs_image = raw_obs["sensor_data"]["c19_front_view"]["rgb"].to(torch.uint8)
        gripper_state = (
            self.unwrapped.agent.robot.get_qpos().to(torch.float32)[:, -1:] * 2
        )

        ee_pose_T = (
            self.unwrapped.agent.ee_pose_at_robot_base.to_transformation_matrix()
            .cpu()
            .numpy()
        )  # (num_envs, 4, 4)

        pos = ee_pose_T[:, :3, 3]  # (num_envs, 3)
        euler = np.stack(
            [mat2euler(ee_pose_T[i, :3, :3], "sxyz") for i in range(self.num_envs)],
            axis=0,
        )  # (num_envs, 3)

        pos = torch.from_numpy(pos).to(gripper_state.device)
        euler = torch.from_numpy(euler).to(gripper_state.device)

        proprioception = torch.cat([pos, euler, gripper_state], dim=1)  # (num_envs, 7)

        infos["extracted_obs"] = {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.get_language_instruction(),
        }

        return raw_obs, _reward, terminations, truncations, infos

    def evaluate(self, success_require_src_completely_on_target=True):
        ret_dict = super().evaluate(success_require_src_completely_on_target)
        return ret_dict

    def compute_dense_reward(self, obs, action, info):
        if self.use_sparse_reward:
            rewards = torch.zeros(
                (self.num_envs,), dtype=torch.float32, device=self.device
            )
            # consecutive_grasp reward
            newly_grasped = info["is_src_obj_grasped_current"] & (~self.is_grasped_flag)
            self.is_grasped_flag = (
                self.is_grasped_flag | info["is_src_obj_grasped_current"]
            )
            rewards += newly_grasped.float() * 1.0

            # success reward
            newly_successed = info["success"] & (~self.is_successed_flag)
            self.is_successed_flag = self.is_successed_flag | info["success"]
            rewards += newly_successed.float() * 5.0

            # If already success but current failed, give a negative reward onces
            failed_after_success = (~info["success"]) & self.is_successed_flag
            newly_failed = failed_after_success & (~self.successed_but_failed_flag)
            self.successed_but_failed_flag = (
                self.successed_but_failed_flag | failed_after_success
            )
            rewards += newly_failed.float() * (-2.0)
        else:
            # actor
            select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
            select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
            carrot_actor = [self.objs_carrot[n] for n in select_carrot]
            plate_actor = [self.objs_plate[n] for n in select_plate]

            carrot_p = torch.stack(
                [a.pose.p[idx] for idx, a in enumerate(carrot_actor)]
            )  # [b, 3]
            plate_p = torch.stack(
                [a.pose.p[idx] for idx, a in enumerate(plate_actor)]
            )  # [b, 3]

            pos_src = carrot_p
            pos_tgt = plate_p
            offset = pos_src - pos_tgt
            xy_dist = torch.linalg.norm(offset[:, :2], dim=1)

            tcp_poses = self.agent.tcp.pose.raw_pose
            tcp_dist = torch.linalg.norm(pos_src - tcp_poses[:, :3], dim=1)

            rewards = 1 - torch.tanh(tcp_dist * 10)
            rewards += info["is_src_obj_grasped_current"].float()
            putting_reward = 1 - torch.tanh(xy_dist * 10)
            putting_reward[~info["is_src_obj_grasped_current"]] = 0.0
            rewards += putting_reward

            rewards[info["success"]] = 5.0

        return rewards

    def compute_normalized_dense_reward(self, obs, action, info):
        original_reward = self.compute_dense_reward(obs, action, info)
        return original_reward / 5.0
