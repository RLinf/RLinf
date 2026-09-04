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

import os
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BRIDGE_DATASET_ASSET_PATH,
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from sapien.physx import PhysxMaterial
from transforms3d.euler import euler2quat

CARROT_DATASET_DIR = (
    Path(os.getenv("MANISKILL_ASSET_DIR", Path(__file__).parent / ".." / "assets"))
    / "carrot"
)


class PutOnPlateInScene25(BaseEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]

    obj_static_friction = 1.0
    obj_dynamic_friction = 1.0

    rgb_camera_name: str = "3rd_view_camera"
    rgb_overlay_mode: str = (
        "background"  # 'background' or 'object' or 'debug' or combinations of them
    )

    overlay_images_numpy: list[np.ndarray]
    overlay_images_hw = (480, 640)
    overlay_images: torch.Tensor

    overlay_textures_numpy: list[np.ndarray]
    overlay_texture_hw = (480, 640)
    overlay_textures: torch.Tensor

    overlay_mix_numpy: list[float]
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

    def __init__(self, **kwargs):
        # random pose
        self._generate_init_pose()

        # widowx
        self.initial_qpos = np.array(
            [
                -0.01840777,
                0.0398835,
                0.22242722,
                -0.00460194,
                1.36524296,
                0.00153398,
                0.037,
                0.037,
            ]
        )
        self.initial_robot_pos = sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
        self.safe_robot_pos = sapien.Pose([0.147, 0.028, 1.870], q=[0, 0, 0, 1])

        # stats
        self.extra_stats = {}

        super().__init__(robot_uids=WidowX250SBridgeDatasetFlatTable, **kwargs)

    def _generate_init_pose(self, obj_z=0.95, plate_z=0.869532):
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
                                np.append(grid_pos_1, obj_z),
                                np.append(grid_pos_2, plate_z),
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

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")

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

        scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

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

    def _reset_overlay(self, env_idx):
        # Handle overlay tensors for full vs partial reset - Keep full size for get_obs() method
        overlay_images = np.stack(
            [self.overlay_images_numpy[idx] for idx in self.select_overlay_ids[env_idx]]
        )
        overlay_textures = np.stack(
            [
                self.overlay_textures_numpy[idx]
                for idx in self.select_overlay_ids[env_idx]
            ]
        )
        overlay_mix = np.array(
            [self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids[env_idx]]
        )
        if not hasattr(self, "overlay_images") or len(env_idx) == self.num_envs:
            self.overlay_images = torch.zeros(
                (self.num_envs, *self.overlay_images_hw, 3), device=self.device
            )
        # Update only the environments being reset
        self.overlay_images[env_idx] = torch.tensor(
            overlay_images, device=self.device, dtype=torch.float32
        )
        if not hasattr(self, "overlay_textures") or len(env_idx) == self.num_envs:
            self.overlay_textures = torch.zeros(
                (self.num_envs, *self.overlay_texture_hw, 3),
                device=self.device,
                dtype=torch.float32,
            )
        # Update only the environments being reset
        self.overlay_textures[env_idx] = torch.tensor(
            overlay_textures, device=self.device, dtype=torch.float32
        )
        if not hasattr(self, "overlay_mix") or len(env_idx) == self.num_envs:
            self.overlay_mix = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.float32
            )
        # Update only the environments being reset
        self.overlay_mix[env_idx] = torch.tensor(
            overlay_mix, device=self.device, dtype=torch.float32
        )

    def _reset_bbox(
        self, env_idx, carrot_actor, plate_actor, select_carrot, select_plate
    ):
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
        # Handle bbox tensors for full vs partial reset - Keep full size for evaluate() method
        if not hasattr(self, "carrot_bbox_world") or len(env_idx) == self.num_envs:
            self.carrot_bbox_world = torch.zeros((self.num_envs, 3), device=self.device)
        # Update only the environments being reset
        self.carrot_bbox_world[env_idx] = c_rotated_bbox_size.clone()

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
        if not hasattr(self, "plate_bbox_world") or len(env_idx) == self.num_envs:
            self.plate_bbox_world = torch.zeros((self.num_envs, 3), device=self.device)
        # Update only the environments being reset
        self.plate_bbox_world[env_idx] = p_rotated_bbox_size.clone()

    def _reset_stats(self, env_idx):
        # stats to track (for all the environments.)
        if not hasattr(self, "consecutive_grasp"):
            self.consecutive_grasp = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device=self.device
            )
            self.episode_stats = {
                "is_src_obj_grasped": torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                ),
                "consecutive_grasp": torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                ),
                "src_on_target": torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                ),
                "gripper_carrot_dist": torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                ),
                "gripper_plate_dist": torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                ),
                "carrot_plate_dist": torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                ),
            }
        else:
            # only update info for the environments that are being reset.
            self.consecutive_grasp[env_idx] = 0
            self.episode_stats["is_src_obj_grasped"][env_idx] = False
            self.episode_stats["consecutive_grasp"][env_idx] = False
            self.episode_stats["src_on_target"][env_idx] = False
            self.episode_stats["gripper_carrot_dist"][env_idx] = 0.0
            self.episode_stats["gripper_plate_dist"][env_idx] = 0.0
            self.episode_stats["carrot_plate_dist"][env_idx] = 0.0
        self.extra_stats = {}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = len(env_idx)

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        self._reset_overlay(env_idx)

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [
            self.carrot_names[idx] for idx in self.select_carrot_ids[env_idx]
        ]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids[env_idx]]
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

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids[env_idx] == idx  # [b]
            p_reset = (
                torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids[env_idx], 0].reshape(
                b, 3
            )  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids[env_idx], 0].reshape(
                b, 4
            )  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids[env_idx] == idx  # [b]
            p_reset = (
                torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids[env_idx], 1].reshape(
                b, 3
            )  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids[env_idx], 1].reshape(
                b, 4
            )  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            pass
            # self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        self._reset_bbox(
            env_idx, carrot_actor, plate_actor, select_carrot, select_plate
        )
        self._reset_stats(env_idx)

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
        self.episode_stats["gripper_carrot_dist"] = gripper_carrot_dist
        self.episode_stats["gripper_plate_dist"] = gripper_plate_dist
        self.episode_stats["carrot_plate_dist"] = carrot_plate_dist

        self.extra_stats["extra_pos_carrot"] = carrot_p
        self.extra_stats["extra_q_carrot"] = carrot_q
        self.extra_stats["extra_pos_plate"] = plate_p
        self.extra_stats["extra_q_plate"] = plate_q
        self.extra_stats["extra_pos_gripper"] = gripper_p
        self.extra_stats["extra_q_gripper"] = gripper_q

        self.task_metric_states = {
            "episode_stats": self.episode_stats,
            "extra_stats": self.extra_stats,
        }

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
        # m = torch.isin(actor_seg, self.robot_link_ids) # "object" mode

        mask = mask.unsqueeze(-1)  # [b, H, W, 1]
        # mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]

        # perform overlay on the RGB observation image
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        rgb = rgb.to(torch.float32)  # [b, H, W, 3]

        rgb_ret = overlay_img * mask  # [b, H, W, 3]
        rgb_ret += rgb * (1 - mask)  # [b, H, W, 3]

        rgb_ret = torch.clamp(rgb_ret, 0, 255)  # [b, H, W, 3]
        rgb_ret = rgb_ret.to(torch.uint8)  # [b, H, W, 3]

        return rgb_ret

    def _get_obs_sensor_data(self, apply_texture_transforms=True):
        sensor_obs = super()._get_obs_sensor_data(apply_texture_transforms)
        if (
            self.obs_mode_struct.visual.rgb
            and self.obs_mode_struct.visual.segmentation
            and self.overlay_images_numpy
        ):
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            camera_name = self.rgb_camera_name
            assert "segmentation" in sensor_obs[camera_name].keys()

            raw_rgb_device = sensor_obs[camera_name]["rgb"].device

            overlay_img = self.overlay_images.to(raw_rgb_device)
            overlay_texture = self.overlay_textures.to(raw_rgb_device)
            overlay_mix = self.overlay_mix.to(raw_rgb_device)

            green_screened_rgb = self._green_sceen_rgb(
                sensor_obs[camera_name]["rgb"],
                sensor_obs[camera_name]["segmentation"],
                overlay_img,
                overlay_texture,
                overlay_mix,
            )
            sensor_obs[camera_name]["rgb"] = green_screened_rgb
        return sensor_obs

    # widowx
    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [
                    [623.588, 0, 319.501],
                    [0, 623.588, 239.545],
                    [0, 0, 1],
                ]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )


@register_env(
    "PutOnPlateInScene25Main-v3",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25MainV3(PutOnPlateInScene25):
    def __init__(self, obj_set, use_multiple_plates=False, **kwargs):
        self.obj_set = obj_set
        self.use_multiple_plates = use_multiple_plates
        self._prep_init()

        super().__init__(**kwargs)

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        # Allow using multiple plates during testing.
        if not self.use_multiple_plates:
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
                cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB),
                (self.overlay_images_hw[1], self.overlay_images_hw[0]),
            )
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(
                cv2.cvtColor(
                    cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB
                ),
                (self.overlay_texture_hw[1], self.overlay_texture_hw[0]),
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

    @property
    def total_num_trials(self):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2 = self.basic_obj_infos
        ltt = lc * lp * lo * l1 * l2
        return ltt

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            lc = 16
            lc_offset = 0
            lo = 16
            lo_offset = 0
        elif self.obj_set == "test":
            lc = 9
            lc_offset = 16
            lo = 5
            lo_offset = 16
        elif self.obj_set == "all":
            lc = 25
            lc_offset = 0
            lo = 21
            lo_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")
        lp_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2

    def _reset_episode_idx(self, env_idx, ltt, options):
        num_envs_to_reset = len(env_idx)

        episode_id = options.get(
            "episode_id",
            torch.randint(
                low=0,
                high=self.total_num_trials,
                size=(num_envs_to_reset,),
                device=self.device,
            ),
        )
        # Synchronized reset:
        if num_envs_to_reset == self.num_envs:
            episode_id = episode_id.reshape(self.num_envs)
            episode_id = episode_id % ltt
            self.episode_id: torch.Tensor = episode_id
        elif 0 < len(env_idx) and len(env_idx) < self.num_envs:
            # Asynchronous (partial) reset
            # env_idx are the ids of environments to be reset.
            # When there is a done env, env_idx is not empty.
            # Generate random integers within (0,ltt) at those coordinates in list env_idx (which should be reset)
            # and keep the episode_id the same as before (queried from self.episode_id) at other coordinates
            if isinstance(env_idx, torch.Tensor):
                reset_env_ids = (
                    env_idx.clone().detach().to(device=self.device, dtype=torch.long)
                )
            elif isinstance(env_idx, list):
                reset_env_ids = torch.tensor(
                    env_idx, device=self.device, dtype=torch.long
                )
            else:
                raise ValueError(f"Unknown type of env_idx: {type(env_idx)}")
            # Use the environment's proper RNG instead of global torch.randint.
            # Warning: if you use the same seed during reset, each environment will always reset to the same scene, which correspond to the same episode_id.
            # To add diversity, you can pass different (however, reproducible) seeds during reset, like, increase the base_seed by 1 for each manual reset at the start of a new parallel rollout.
            self.episode_id[reset_env_ids] = episode_id
        else:
            raise ValueError(f"Invalid env_idx: {env_idx}")

        self.task_reset_states = {
            "episode_id": self.episode_id,
        }

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2 = self.basic_obj_infos
        self._reset_episode_idx(env_idx, self.total_num_trials, options)

        self.select_carrot_ids = (
            self.episode_id // (lp * lo * l1 * l2) + lc_offset
        )  # [b]
        self.select_plate_ids = (self.episode_id // (lo * l1 * l2)) % lp + lp_offset
        self.select_overlay_ids = (self.episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (self.episode_id // l2) % l1
        self.select_quat_ids = self.episode_id % l2

    def _generate_init_pose(self):
        super()._generate_init_pose(0.90, 0.90)


__all__ = ["PutOnPlateInScene25MainV3", "PutOnPlateInScene25"]
