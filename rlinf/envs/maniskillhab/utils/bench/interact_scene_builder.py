import gzip
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import transforms3d

import sapien
import sapien.physx as physx

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
    Fetch,
)
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor, Articulation


IGNORE_FETCH_COLLISION_STRS = ["mat", "rug", "carpet"]


class ReplicaCADInteractSceneBuilder(SceneBuilder):
    robot_initial_pose = sapien.Pose(
        p=[-1, 0, 0.02]
    )  # generally a safe initial spawn pose for the Fetch robot

    # interact uses ReplicaCAD w/ baked lighting
    builds_lighting = False

    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise)

        with gzip.open(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange/hab2_bench_assets/bench_scene.json.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            self.interact_config: Dict = json.load(f)["episodes"][0]

        with open(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange"
            / Path("/".join(self.interact_config["scene_id"].split("/")[1:])),
            "r",
        ) as f:
            self.build_config: Dict = json.load(f)

    def build(self):
        self.scene_objects: Dict[str, Actor] = dict()
        self.movable_objects: Dict[str, Actor] = dict()
        self.articulations: Dict[str, Articulation] = dict()
        self._default_object_poses: List[Tuple[Actor, sapien.Pose]] = []

        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )

        # Note all ReplicaCAD assets are rotated by 90 degrees as they use a different xyz convention to SAPIEN/ManiSkill.
        bg_pose = sapien.Pose(q=q)

        # ReplicaCAD stores the background model here
        background_template_name = osp.basename(
            self.build_config["stage_instance"]["template_name"]
        )
        bg_path = str(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange"
            / f"hab2_bench_assets/stages/{background_template_name}.glb"
        )
        builder = self.scene.create_actor_builder()

        # When creating objects that do not need to be moved ever, you must provide the pose of the object ahead of time
        # and use builder.build_static. Objects such as the scene background (also called a stage) fits in this category
        builder.add_visual_from_file(bg_path)
        builder.add_nonconvex_collision_from_file(bg_path)
        builder.initial_pose = bg_pose
        self.bg = builder.build_static(name=f"scene_background")

        # articulations
        articulation_to_num = defaultdict(int)
        for articulated_meta in self.build_config["articulated_object_instances"]:

            template_name = articulated_meta["template_name"]
            if "door" in template_name:
                continue
            pos = articulated_meta["translation"]
            rot = articulated_meta["rotation"]
            urdf_path = osp.join(
                ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange",
                f"hab2_bench_assets/urdf/{template_name}/{template_name}.urdf",
            )
            urdf_loader = self.scene.create_urdf_loader()
            articulation_name = f"{template_name}-{articulation_to_num[template_name]}"
            urdf_loader.name = articulation_name
            urdf_loader.fix_root_link = articulated_meta["fixed_base"]
            urdf_loader.disable_self_collisions = True
            if "uniform_scale" in articulated_meta:
                urdf_loader.scale = articulated_meta["uniform_scale"]
            builder = urdf_loader.parse(urdf_path)["articulation_builders"][0]
            pose = sapien.Pose(q=q) * sapien.Pose(pos, rot)
            builder.initial_pose = pose
            articulation = builder.build()
            self._default_object_poses.append((articulation, pose))

            self.articulations[articulation_name] = articulation
            self.scene_objects[articulation_name] = articulation

            for link in articulation.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
                )

            articulation_to_num[template_name] += 1

        world_transform = sapien.Pose(q=q).inv()
        obj_transform = sapien.Pose(q=q, p=[0, 0, 0.01])
        actor_id_to_num_made = defaultdict(int)
        for actor_id, transformation in self.interact_config["rigid_objs"]:
            actor_id = actor_id.split(".")[0]
            pose = obj_transform * sapien.Pose(transformation) * world_transform

            obj_instance_name = f"{actor_id}-{actor_id_to_num_made[actor_id]}"
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{actor_id}")
            builder.initial_pose = pose
            actor = builder.build(name=obj_instance_name)

            self._default_object_poses.append((actor, pose))
            self.scene_objects[obj_instance_name] = actor
            self.movable_objects[obj_instance_name] = actor

            actor_id_to_num_made[actor_id] += 1

    def initialize(self, env_idx: torch.Tensor):

        # teleport robot away for init
        self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                # note that during initialization you may only ever change poses/qpos of objects in scenes being reset
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

        if physx.is_gpu_enabled():
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        # teleport robot back to correct location
        if self.env.robot_uids == "fetch":
            self.env.agent.reset(
                [
                    0,
                    0,
                    0,
                    0.15,
                    0,
                    -0.45,
                    0.562,
                    -1.08,
                    0.1,
                    0.935,
                    -0.001,
                    1.573,
                    0.005,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.robot.set_pose(
                sapien.Pose(
                    p=[0.7, -0.6, 0.02],
                    q=transforms3d.quaternions.axangle2quat(
                        np.array([0, 0, 1]), theta=np.deg2rad(90)
                    ),
                ),
            )
        else:
            raise NotImplementedError(self.env.robot_uids)

    def disable_fetch_move_collisions(
        self,
        actor: Actor,
        disable_base_collisions=False,
    ):
        actor.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
        if disable_base_collisions:
            actor.set_collision_group_bit(
                group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
            )
