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

import gymnasium as gym
import numpy as np

PEG_INSERTION_SIDE_WIDE_ENV_ID = "PegInsertionSideWideClearance-v1"
PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID = (
    "PegInsertionSideWideClearanceObserverWideWrist-v1"
)
PANDA_WIDE_WRISTCAM_UID = "panda_wristcam_wide"
PEG_INSERTION_SIDE_BASE_CLEARANCE = 0.003
PEG_INSERTION_SIDE_WIDE_CLEARANCE = 0.02

_PEG_VARIANTS_REGISTERED = False


def is_peg_insertion_side_env_id(env_id: str | None) -> bool:
    return env_id in {
        PEG_INSERTION_SIDE_WIDE_ENV_ID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
    }


def get_joint_observer_env_id(env_id: str | None) -> str | None:
    if env_id in {
        PEG_INSERTION_SIDE_WIDE_ENV_ID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
    }:
        return PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID
    return None


def register_rlinf_peg_insertion_side_variants() -> None:
    global _PEG_VARIANTS_REGISTERED
    if _PEG_VARIANTS_REGISTERED:
        return
    if (
        PEG_INSERTION_SIDE_WIDE_ENV_ID in gym.registry
        and PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID in gym.registry
    ):
        _PEG_VARIANTS_REGISTERED = True
        return

    import sapien
    import torch
    from mani_skill.agents.registration import register_agent
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import common, sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose

    def _build_box_with_hole(
        scene,
        inner_radius: float,
        outer_radius: float,
        depth: float,
        center=(0, 0),
    ):
        builder = scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            sapien.Pose([0, offset + half_center[0], 0]),
            sapien.Pose([0, -offset + half_center[0], 0]),
            sapien.Pose([0, 0, offset + half_center[1]]),
            sapien.Pose([0, 0, -offset + half_center[1]]),
        ]
        mat = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
        )
        for half_size, pose in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder

    @register_agent()
    class PandaWristWideCam(PandaWristCam):  # type: ignore[unused-ignore]
        uid = PANDA_WIDE_WRISTCAM_UID

        @property
        def _sensor_configs(self):
            configs = list(super()._sensor_configs)
            tilt = np.deg2rad(-15.0)
            configs.append(
                CameraConfig(
                    uid="wide_hand_camera",
                    pose=sapien.Pose(
                        p=[-0.055, 0.0, 0.035],
                        q=[float(np.cos(tilt / 2)), 0.0, float(np.sin(tilt / 2)), 0.0],
                    ),
                    width=384,
                    height=384,
                    fov=1.45,
                    near=0.01,
                    far=100,
                    mount=self.robot.links_map["camera_link"],
                )
            )
            return configs

    def _observer_sensor_configs(base_env) -> list[CameraConfig]:
        configs = list(base_env._default_sensor_configs)
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        configs.append(
            CameraConfig(
                "3rd_view_camera",
                pose,
                384,
                384,
                1.0,
                0.01,
                100,
            )
        )
        return configs

    class _PegInsertionSideWideSceneMixin:
        hole_clearance = PEG_INSERTION_SIDE_WIDE_CLEARANCE
        success_clearance = PEG_INSERTION_SIDE_WIDE_CLEARANCE

        def _load_scene(self, options: dict):
            del options
            with torch.device(self.device):
                self.table_scene = TableSceneBuilder(self)
                self.table_scene.build()
                lengths = self._batched_episode_rng.uniform(0.085, 0.125)
                radii = self._batched_episode_rng.uniform(0.015, 0.025)
                centers = np.zeros((self.num_envs, 2), dtype=np.float32)
                self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
                peg_head_offsets = torch.zeros((self.num_envs, 3))
                peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
                self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)
                box_hole_offsets = torch.zeros((self.num_envs, 3))
                box_hole_offsets[:, 1:] = common.to_tensor(centers)
                self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
                self.box_hole_radii = common.to_tensor(radii + self.success_clearance)

                pegs = []
                boxes = []
                for i in range(self.num_envs):
                    scene_idxs = [i]
                    length = lengths[i]
                    radius = radii[i]

                    builder = self.scene.create_actor_builder()
                    builder.add_box_collision(half_size=[length, radius, radius])
                    mat = sapien.render.RenderMaterial(
                        base_color=sapien_utils.hex2rgba("#EC7357"),
                        roughness=0.5,
                        specular=0.5,
                    )
                    builder.add_box_visual(
                        sapien.Pose([length / 2, 0, 0]),
                        half_size=[length / 2, radius, radius],
                        material=mat,
                    )
                    mat = sapien.render.RenderMaterial(
                        base_color=sapien_utils.hex2rgba("#EDF6F9"),
                        roughness=0.5,
                        specular=0.5,
                    )
                    builder.add_box_visual(
                        sapien.Pose([-length / 2, 0, 0]),
                        half_size=[length / 2, radius, radius],
                        material=mat,
                    )
                    builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                    builder.set_scene_idxs(scene_idxs)
                    peg = builder.build(f"peg_{i}")
                    self.remove_from_state_dict_registry(peg)

                    inner_radius, outer_radius, depth = (
                        radius + self.hole_clearance,
                        length,
                        length,
                    )
                    builder = _build_box_with_hole(
                        self.scene,
                        inner_radius,
                        outer_radius,
                        depth,
                        center=centers[i],
                    )
                    builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                    builder.set_scene_idxs(scene_idxs)
                    box = builder.build_kinematic(f"box_with_hole_{i}")
                    self.remove_from_state_dict_registry(box)
                    pegs.append(peg)
                    boxes.append(box)

                self.peg = Actor.merge(pegs, "peg")
                self.box = Actor.merge(boxes, "box_with_hole")
                self.add_to_state_dict_registry(self.peg)
                self.add_to_state_dict_registry(self.box)

    @register_env(PEG_INSERTION_SIDE_WIDE_ENV_ID, max_episode_steps=100)
    class PegInsertionSideWideClearanceEnv(
        _PegInsertionSideWideSceneMixin, PegInsertionSideEnv
    ):  # type: ignore[unused-ignore]
        _clearance = PEG_INSERTION_SIDE_BASE_CLEARANCE

    @register_env(
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        max_episode_steps=100,
    )
    class PegInsertionSideWideClearanceObserverWideWristEnv(
        PegInsertionSideWideClearanceEnv
    ):  # type: ignore[unused-ignore]
        SUPPORTED_ROBOTS = ["panda_wristcam", PANDA_WIDE_WRISTCAM_UID]

        def __init__(self, *args, robot_uids=PANDA_WIDE_WRISTCAM_UID, **kwargs):
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sensor_configs(self):
            return _observer_sensor_configs(super())

    _PEG_VARIANTS_REGISTERED = True
