import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

import gymnasium as gym

import numpy as np
import torch
import transforms3d

import sapien

import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.utils.scene_builder.replicacad.rearrange import (
    ReplicaCADPrepareGroceriesTrainSceneBuilder,
    ReplicaCADPrepareGroceriesValSceneBuilder,
    ReplicaCADSetTableTrainSceneBuilder,
    ReplicaCADSetTableValSceneBuilder,
    ReplicaCADTidyHouseTrainSceneBuilder,
    ReplicaCADTidyHouseValSceneBuilder,
)
from mani_skill.utils.scene_builder.replicacad.rearrange.scene_builder import (
    ReplicaCADRearrangeSceneBuilder,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.pose import to_sapien_pose

from mshab.envs.planner import (
    CloseSubtask,
    NavigateSubtask,
    OpenSubtask,
    PickSubtask,
    PlaceSubtask,
    TaskPlan,
    plan_data_from_file,
)


GOAL_POSE_Q = transforms3d.quaternions.axangle2quat(
    np.array([0, 1, 0]), theta=np.deg2rad(90)
)


@dataclass
class GenSpawnPositionArgs:
    task: str
    subtask: str
    split: str
    seed: int
    num_workers: int
    # not passable (for now)
    root = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans"
    num_spawns_per_task_plan = 1
    init_check_scene_steps = 1
    robot_init_qpos_noise = 0.2
    spawn_loc_radius = 2
    place_obj_goal_thresh = 0.15

    def __post_init__(self):
        assert self.task in ["tidy_house", "prepare_groceries", "set_table"]
        assert self.num_workers >= 1

        self.root = Path(self.root)


def make_env(scene_builder_cls) -> SceneManipulationEnv:
    return gym.make(
        # important sapien_env kwargs
        "SceneManipulation-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        reconfiguration_freq=0,
        # scene manip kwargs
        scene_builder_cls=scene_builder_cls,
        # other sapien_env kwargs
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        shader_dir="minimal",
        robot_uids="fetch",
        # time limit
        max_episode_steps=100,
    )


def gen_pick_spawn_data(
    proc_num,
    args: GenSpawnPositionArgs,
    scene_builder_cls,
    task_plans: List[TaskPlan],
):
    build_config_name = task_plans[0].build_config_name
    env = make_env(scene_builder_cls)

    scene_builder: ReplicaCADRearrangeSceneBuilder = env.scene_builder
    build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
    init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

    subtask_uid_to_spawn_data = dict()

    env.reset(
        seed=args.seed + proc_num,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )

    agent_bodies = []
    for link in env.agent.robot.links:
        agent_bodies += link._bodies
    agent_bodies = set(agent_bodies)
    num_agent_contacts = lambda contacts: len(
        [c for c in contacts if any([b in agent_bodies for b in c.bodies])]
    )

    for tp in tqdm(task_plans):
        env.reset(
            seed=args.seed + proc_num,
            options=dict(
                reconfigure=False,
                init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
            ),
        )

        assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], PickSubtask)
        subtask: PickSubtask = tp.subtasks[0]

        subtask_obj = scene_builder.movable_objects[f"env-0_{subtask.obj_id}"]

        navigable_positions = torch.tensor(
            scene_builder.navigable_positions[0].vertices
        )

        if args.task == "set_table":
            subtask_articulation = scene_builder.articulations[
                f"env-0_{subtask.articulation_config.articulation_id}"
            ]
            if subtask.articulation_config.articulation_type == "fridge":
                min_open_qpos_frac = 0.75
            elif subtask.articulation_config.articulation_type == "kitchen_counter":
                min_open_qpos_frac = 0.9
            else:
                raise NotImplementedError(
                    f"{subtask.articulation_config.articulation_type=} not supported"
                )
            spawn_articulation_qpos = []
            spawn_obj_raw_pose = []

        spawn_pos, spawn_qpos = [], []
        while len(spawn_pos) < args.num_spawns_per_task_plan:
            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            if args.task == "set_table":
                if subtask.articulation_config.articulation_type == "kitchen_counter":
                    subtask_obj_pose_wrt_container = (
                        subtask_articulation.links[
                            subtask.articulation_config.articulation_handle_link_idx
                        ].pose.inv()
                        * subtask_obj.pose
                    )

                robot_init_pos = env.agent.robot.pose.p
                robot_init_pos[:, :2] = 99999
                env.agent.robot.set_pose(Pose.create_from_pq(p=robot_init_pos))

                new_subtask_articulation_qpos = subtask_articulation.qpos * 0
                joint_qmax = subtask_articulation.qlimits[
                    :,
                    subtask.articulation_config.articulation_handle_active_joint_idx,
                    1,
                ]
                joint_qmin = subtask_articulation.qlimits[
                    :,
                    subtask.articulation_config.articulation_handle_active_joint_idx,
                    0,
                ]
                joint_qrange = joint_qmax - joint_qmin
                joint_open_qmin = joint_qrange * min_open_qpos_frac + joint_qmin
                rand_joint_qpos = (
                    torch.rand_like(joint_qmax) * (joint_qmax - joint_open_qmin)
                ) + joint_open_qmin
                new_subtask_articulation_qpos[
                    :, subtask.articulation_config.articulation_handle_active_joint_idx
                ] = rand_joint_qpos
                subtask_articulation.set_qpos(new_subtask_articulation_qpos)

                if subtask.articulation_config.articulation_type == "kitchen_counter":
                    subtask_obj.set_pose(
                        subtask_articulation.links[
                            subtask.articulation_config.articulation_handle_link_idx
                        ].pose
                        * subtask_obj_pose_wrt_container
                    )

            positions_wrt_centers = navigable_positions - subtask_obj.pose.p[0, :2]
            dists = torch.norm(positions_wrt_centers, dim=-1)

            new_navigable_positions = navigable_positions[dists < args.spawn_loc_radius]
            positions_wrt_centers = positions_wrt_centers[dists < args.spawn_loc_radius]
            dists = dists[dists < args.spawn_loc_radius]
            rots = (
                torch.sign(positions_wrt_centers[..., 1])
                * torch.arccos(positions_wrt_centers[..., 0] / dists)
                + torch.pi
            ) % (2 * torch.pi)

            # spawn to try
            spawn_num = torch.randint(
                low=0, high=len(new_navigable_positions), size=(1,)
            )

            # base pos
            loc = new_navigable_positions[spawn_num]
            robot_pos = env.agent.robot.pose.p
            robot_pos[:, :2] = loc
            robot_pos[:, :2] += torch.clamp(
                torch.normal(0, 0.1, robot_pos[:, :2].shape), -0.2, 0.2
            )
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            # base rot
            env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
            qpos = env.agent.robot.get_qpos()
            rot = rots[spawn_num]
            qpos[:, 2] = rot
            qpos[:, 2:3] += torch.clamp(
                torch.normal(0, 0.25, qpos[:, 2:3].shape), -0.5, 0.5
            )
            # arm qpos
            qpos[:, 5:6] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 5:6].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            qpos[:, 7:-2] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 7:-2].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            env.agent.reset(qpos)

            robot_force = 0
            total_agent_contacts = 0
            for _ in range(args.init_check_scene_steps):
                env.scene.step()

                robot_force = robot_force + env.agent.robot.get_net_contact_forces(
                    env.agent.robot_link_names
                ).norm(dim=-1).sum(dim=-1)
                total_agent_contacts += num_agent_contacts(env.scene.get_contacts())

            if (
                robot_force.item() == 0
                and total_agent_contacts == args.init_check_scene_steps
            ):
                spawn_pos.append(env.agent.robot.pose.p[0])
                spawn_qpos.append(env.agent.robot.qpos[0])
                if args.task == "set_table":
                    spawn_articulation_qpos.append(subtask_articulation.qpos[0])
                    spawn_obj_raw_pose.append(subtask_obj.pose.raw_pose[0])

        if args.task == "set_table":
            subtask_uid_to_spawn_data[subtask.uid] = dict(
                robot_pos=torch.stack(spawn_pos),
                robot_qpos=torch.stack(spawn_qpos),
                articulation_qpos=torch.stack(spawn_articulation_qpos),
                obj_raw_pose=torch.stack(spawn_obj_raw_pose),
            )
        else:
            subtask_uid_to_spawn_data[subtask.uid] = dict(
                robot_pos=torch.stack(spawn_pos),
                robot_qpos=torch.stack(spawn_qpos),
            )

    return subtask_uid_to_spawn_data


def gen_place_spawn_data(
    proc_num,
    args: GenSpawnPositionArgs,
    scene_builder_cls,
    task_plans: List[TaskPlan],
):
    build_config_name = task_plans[0].build_config_name
    env = make_env(scene_builder_cls)

    scene_builder: ReplicaCADRearrangeSceneBuilder = env.scene_builder
    build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
    init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

    grasping_spawns = dict()
    if args.task == "set_table":
        task_obj_names = [
            "013_apple",
            "024_bowl",
        ]
    else:
        task_obj_names = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "008_pudding_box",
            "007_tuna_fish_can",
            "009_gelatin_box",
            "010_potted_meat_can",
            "024_bowl",
        ]
    for obj_name in task_obj_names:
        with open(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange/grasp_poses"
            / args.task
            / obj_name
            / "grasp_poses.pt",
            "rb",
        ) as spawns_fp:
            grasping_spawns[obj_name] = torch.load(spawns_fp)

    subtask_uid_to_spawn_data = dict()
    env.reset(
        seed=args.seed + proc_num,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )

    agent_bodies = []
    for link in env.agent.robot.links:
        agent_bodies += link._bodies
    agent_bodies = set(agent_bodies)
    num_agent_contacts = lambda contacts: len(
        [c for c in contacts if any([b in agent_bodies for b in c.bodies])]
    )

    for tp in tqdm(task_plans):
        env.reset(
            seed=args.seed + proc_num,
            options=dict(
                reconfigure=False,
                init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
            ),
        )

        assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], PlaceSubtask)
        subtask: PlaceSubtask = tp.subtasks[0]

        subtask_obj = scene_builder.movable_objects[f"env-0_{subtask.obj_id}"]
        obj_name = subtask.obj_id.split("-")[0]

        subtask_obj_bodies = set(subtask_obj._bodies)
        num_subtask_obj_contacts = lambda contacts: len(
            [c for c in contacts if any([b in subtask_obj_bodies for b in c.bodies])]
        )

        navigable_positions = torch.tensor(
            scene_builder.navigable_positions[0].vertices
        )

        spawn_pos, spawn_qpos = [], []
        spawn_obj_raw_pose_wrt_tcp = []
        while len(spawn_pos) < args.num_spawns_per_task_plan:
            # grasp spawn qpos and obj_raw_pose_wrt_tcp
            grasp_spawn = grasping_spawns[obj_name]
            grasp_spawn_num = torch.randint(
                low=0, high=len(grasp_spawn["success_qpos"]), size=(1,)
            )
            qpos = grasp_spawn["success_qpos"][grasp_spawn_num]
            obj_raw_pose_wrt_tcp = grasp_spawn["success_obj_raw_pose_wrt_tcp"][
                grasp_spawn_num
            ].clone()
            if torch.norm(Pose.create(obj_raw_pose_wrt_tcp).p[0]) > 0.15:
                continue

            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            subtask_goal_pose = Pose.create_from_pq(q=GOAL_POSE_Q, p=subtask.goal_pos)

            goal_center = subtask_goal_pose.p[0, :2]

            positions_wrt_centers = navigable_positions - goal_center
            dists = torch.norm(positions_wrt_centers, dim=-1)

            new_navigable_positions = navigable_positions[dists < args.spawn_loc_radius]
            positions_wrt_centers = positions_wrt_centers[dists < args.spawn_loc_radius]
            dists = dists[dists < args.spawn_loc_radius]
            rots = (
                torch.sign(positions_wrt_centers[..., 1])
                * torch.arccos(positions_wrt_centers[..., 0] / dists)
                + torch.pi
            ) % (2 * torch.pi)

            # spawn to try
            spawn_num = torch.randint(
                low=0, high=len(new_navigable_positions), size=(1,)
            )

            # base pos
            loc = new_navigable_positions[spawn_num]
            robot_pos = env.agent.robot.pose.p
            robot_pos[:, :2] = loc
            robot_pos[:, :2] += torch.clamp(
                torch.normal(0, 0.1, robot_pos[:, :2].shape), -0.2, 0.2
            )
            robot_pos[:, 2] = 0.02
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            # base rot
            rot = rots[spawn_num]
            qpos[:, 2] = rot
            qpos[:, 2:3] += torch.clamp(
                torch.normal(0, 0.25, qpos[:, 2:3].shape), -0.5, 0.5
            )
            # arm qpos
            qpos[:, 5:6] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 5:6].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            qpos[:, 7:-2] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 7:-2].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            env.agent.reset(qpos)
            grasp_tcp_raw_pose = env.agent.tcp.pose.raw_pose.clone()
            robot_pose_p = env.agent.robot.pose.p[0].clone()
            robot_qpos = env.agent.robot.qpos[0].clone()

            robot_force = None
            total_agent_contacts = 0
            for _ in range(args.init_check_scene_steps):
                env.scene.step()
                rforce = (
                    env.agent.robot.get_net_contact_forces(env.agent.robot_link_names)
                    .norm(dim=-1)
                    .sum(dim=-1)
                )
                if robot_force is None:
                    robot_force = rforce
                else:
                    robot_force += rforce
                total_agent_contacts += num_agent_contacts(env.scene.get_contacts())
            robot_spawn_success = (
                robot_force.item() == 0
                and total_agent_contacts == args.init_check_scene_steps
            )

            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            env.agent.robot.set_pose(sapien.Pose(p=[99999, 99999, 99999]))
            subtask_obj.set_pose(
                Pose.create(grasp_tcp_raw_pose) * Pose.create(obj_raw_pose_wrt_tcp)
            )

            env.scene.step()

            obj_force = subtask_obj.get_net_contact_forces().norm(dim=-1)
            obj_spawn_success = (
                obj_force.item() == 0
                and num_subtask_obj_contacts(env.scene.get_contacts()) == 0
            )

            if robot_spawn_success and obj_spawn_success:
                spawn_pos.append(robot_pose_p)
                spawn_qpos.append(robot_qpos)
                spawn_obj_raw_pose_wrt_tcp.append(obj_raw_pose_wrt_tcp[0])

        subtask_uid_to_spawn_data[subtask.uid] = dict(
            robot_pos=torch.stack(spawn_pos),
            robot_qpos=torch.stack(spawn_qpos),
            obj_raw_pose_wrt_tcp=torch.stack(spawn_obj_raw_pose_wrt_tcp),
        )

    return subtask_uid_to_spawn_data


def gen_open_spawn_data(
    proc_num,
    args: GenSpawnPositionArgs,
    scene_builder_cls,
    task_plans: List[TaskPlan],
):
    build_config_name = task_plans[0].build_config_name
    env = make_env(scene_builder_cls)

    scene_builder: ReplicaCADRearrangeSceneBuilder = env.scene_builder
    build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
    init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

    subtask_uid_to_spawn_data = dict()

    env.reset(
        seed=args.seed + proc_num,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )

    agent_bodies = []
    for link in env.agent.robot.links:
        agent_bodies += link._bodies
    agent_bodies = set(agent_bodies)
    num_agent_contacts = lambda contacts: len(
        [c for c in contacts if any([b in agent_bodies for b in c.bodies])]
    )

    for tp in tqdm(task_plans):
        env.reset(
            seed=args.seed + proc_num,
            options=dict(
                reconfigure=False,
                init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
            ),
        )

        assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], OpenSubtask)
        subtask: OpenSubtask = tp.subtasks[0]

        subtask_articulation = scene_builder.articulations[
            f"env-0_{subtask.articulation_id}"
        ]
        subtask_obj = scene_builder.movable_objects[f"env-0_{subtask.obj_id}"]

        if subtask.articulation_type == "fridge":
            subtask_articulation_sapien_pose = to_sapien_pose(subtask_articulation.pose)
            xmin = 0.933
            xmax = 1.833
            ymin = -0.6
            ymax = 0.6
        elif subtask.articulation_type == "kitchen_counter":
            subtask_articulation_sapien_pose = to_sapien_pose(
                subtask_articulation.links[subtask.articulation_handle_link_idx].pose
            )
            xmin = 0.3
            xmax = 1.5
            ymin = -0.6
            ymax = 0.6
        else:
            raise NotImplementedError(f"{subtask.articulation_type=} not supported")

        xmin = (subtask_articulation_sapien_pose * sapien.Pose(p=[xmin, 0, 0])).p[0]
        xmax = (subtask_articulation_sapien_pose * sapien.Pose(p=[xmax, 0, 0])).p[0]
        # NOTE (arth): hab uses y axis as up/down
        ymin = (subtask_articulation_sapien_pose * sapien.Pose(p=[0, 0, ymin])).p[1]
        ymax = (subtask_articulation_sapien_pose * sapien.Pose(p=[0, 0, ymax])).p[1]

        if xmin > xmax:
            xmin, xmax = (xmax, xmin)
        if ymin > ymax:
            ymin, ymax = (ymax, ymin)

        obj_center = subtask_obj.pose.p[0, :2]
        navigable_positions = torch.tensor(
            scene_builder.navigable_positions[0].vertices
        )

        spawn_pos, spawn_qpos = [], []
        while len(spawn_pos) < args.num_spawns_per_task_plan:
            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            positions_wrt_centers = navigable_positions - obj_center
            dists = torch.norm(positions_wrt_centers, dim=-1)

            criterion = (
                (xmin <= navigable_positions[:, 0])
                & (navigable_positions[:, 0] <= xmax)
                & (ymin <= navigable_positions[:, 1])
                & (navigable_positions[:, 1] <= ymax)
            )
            new_navigable_positions = navigable_positions[criterion]
            positions_wrt_centers = positions_wrt_centers[criterion]
            dists = dists[criterion]
            rots = (
                torch.sign(positions_wrt_centers[..., 1])
                * torch.arccos(positions_wrt_centers[..., 0] / dists)
                + torch.pi
            ) % (2 * torch.pi)

            # spawn to try
            spawn_num = torch.randint(
                low=0, high=len(new_navigable_positions), size=(1,)
            )

            # base pos
            loc = new_navigable_positions[spawn_num]
            robot_pos = env.agent.robot.pose.p
            robot_pos[:, :2] = loc
            robot_pos[:, :2] += torch.clamp(
                torch.normal(0, 0.1, robot_pos[:, :2].shape), -0.2, 0.2
            )
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            # base rot
            env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
            qpos = env.agent.robot.get_qpos()
            rot = rots[spawn_num]
            qpos[:, 2] = rot
            qpos[:, 2:3] += torch.clamp(
                torch.normal(0, 0.25, qpos[:, 2:3].shape), -0.5, 0.5
            )
            # arm qpos
            qpos[:, 5:6] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 5:6].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            qpos[:, 7:-2] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 7:-2].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            env.agent.reset(qpos)

            robot_force = 0
            total_agent_contacts = 0
            for _ in range(args.init_check_scene_steps):
                env.scene.step()

                robot_force = robot_force + env.agent.robot.get_net_contact_forces(
                    env.agent.robot_link_names
                ).norm(dim=-1).sum(dim=-1)
                total_agent_contacts += num_agent_contacts(env.scene.get_contacts())

            if (
                robot_force.item() == 0
                and total_agent_contacts == args.init_check_scene_steps
            ):
                spawn_pos.append(env.agent.robot.pose.p[0])
                spawn_qpos.append(env.agent.robot.qpos[0])

        subtask_uid_to_spawn_data[subtask.uid] = dict(
            robot_pos=torch.stack(spawn_pos),
            robot_qpos=torch.stack(spawn_qpos),
        )

    return subtask_uid_to_spawn_data


def gen_close_spawn_data(
    proc_num,
    args: GenSpawnPositionArgs,
    scene_builder_cls,
    task_plans: List[TaskPlan],
):
    build_config_name = task_plans[0].build_config_name
    env = make_env(scene_builder_cls)

    scene_builder: ReplicaCADRearrangeSceneBuilder = env.scene_builder
    build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
    init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

    subtask_uid_to_spawn_data = dict()

    env.reset(
        seed=args.seed + proc_num,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )

    agent_bodies = []
    for link in env.agent.robot.links:
        agent_bodies += link._bodies
    agent_bodies = set(agent_bodies)
    num_agent_contacts = lambda contacts: len(
        [c for c in contacts if any([b in agent_bodies for b in c.bodies])]
    )

    for tp in tqdm(task_plans):
        env.reset(
            seed=args.seed + proc_num,
            options=dict(
                reconfigure=False,
                init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
            ),
        )

        assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], CloseSubtask)
        subtask: CloseSubtask = tp.subtasks[0]

        subtask_articulation = scene_builder.articulations[
            f"env-0_{subtask.articulation_id}"
        ]

        if subtask.articulation_type == "fridge":
            subtask_articulation_sapien_pose = to_sapien_pose(subtask_articulation.pose)
            xmin = 0.933
            xmax = 1.833
            ymin = -0.6
            ymax = 0.6
            min_open_qpos_frac = 0.75
        elif subtask.articulation_type == "kitchen_counter":
            subtask_articulation_sapien_pose = to_sapien_pose(
                subtask_articulation.links[subtask.articulation_handle_link_idx].pose
            )
            xmin = 0.3
            xmax = 1.5
            ymin = -0.6
            ymax = 0.6
            min_open_qpos_frac = 0.9
        else:
            raise NotImplementedError(f"{subtask.articulation_type=} not supported")

        xmin = (subtask_articulation_sapien_pose * sapien.Pose(p=[xmin, 0, 0])).p[0]
        xmax = (subtask_articulation_sapien_pose * sapien.Pose(p=[xmax, 0, 0])).p[0]
        # NOTE (arth): hab uses y axis as up/down
        ymin = (subtask_articulation_sapien_pose * sapien.Pose(p=[0, 0, ymin])).p[1]
        ymax = (subtask_articulation_sapien_pose * sapien.Pose(p=[0, 0, ymax])).p[1]

        if xmin > xmax:
            xmin, xmax = (xmax, xmin)
        if ymin > ymax:
            ymin, ymax = (ymax, ymin)

        navigable_positions = torch.tensor(
            scene_builder.navigable_positions[0].vertices
        )
        original_subtask_articulation_pos = torch.from_numpy(
            subtask_articulation_sapien_pose.p.copy()
        )

        spawn_pos, spawn_qpos = [], []
        spawn_articulation_qpos = []
        while len(spawn_pos) < args.num_spawns_per_task_plan:
            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            robot_init_pos = env.agent.robot.pose.p
            robot_init_pos[:, :2] = 99999
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_init_pos))

            new_subtask_articulation_qpos = subtask_articulation.qpos * 0
            joint_qmax = subtask_articulation.qlimits[
                :, subtask.articulation_handle_active_joint_idx, 1
            ]
            joint_qmin = subtask_articulation.qlimits[
                :, subtask.articulation_handle_active_joint_idx, 0
            ]
            joint_qrange = joint_qmax - joint_qmin
            joint_open_qmin = joint_qrange * min_open_qpos_frac + joint_qmin
            rand_joint_qpos = (
                torch.rand_like(joint_qmax) * (joint_qmax - joint_open_qmin)
            ) + joint_open_qmin
            new_subtask_articulation_qpos[
                :, subtask.articulation_handle_active_joint_idx
            ] = rand_joint_qpos
            subtask_articulation.set_qpos(new_subtask_articulation_qpos)

            positions_wrt_centers = (
                navigable_positions - original_subtask_articulation_pos[:2]
            )
            dists = torch.norm(positions_wrt_centers, dim=-1)

            criterion = (
                (xmin <= navigable_positions[:, 0])
                & (navigable_positions[:, 0] <= xmax)
                & (ymin <= navigable_positions[:, 1])
                & (navigable_positions[:, 1] <= ymax)
            )
            new_navigable_positions = navigable_positions[criterion]
            positions_wrt_centers = positions_wrt_centers[criterion]
            dists = dists[criterion]
            rots = (
                torch.sign(positions_wrt_centers[..., 1])
                * torch.arccos(positions_wrt_centers[..., 0] / dists)
                + torch.pi
            ) % (2 * torch.pi)

            # spawn to try
            spawn_num = torch.randint(
                low=0, high=len(new_navigable_positions), size=(1,)
            )

            # base pos
            loc = new_navigable_positions[spawn_num]
            robot_pos = env.agent.robot.pose.p
            robot_pos[:, :2] = loc
            robot_pos[:, :2] += torch.clamp(
                torch.normal(0, 0.1, robot_pos[:, :2].shape), -0.2, 0.2
            )
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            # base rot
            env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
            qpos = env.agent.robot.get_qpos()
            rot = rots[spawn_num]
            qpos[:, 2] = rot
            qpos[:, 2:3] += torch.clamp(
                torch.normal(0, 0.25, qpos[:, 2:3].shape), -0.5, 0.5
            )
            # arm qpos
            qpos[:, 5:6] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 5:6].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            qpos[:, 7:-2] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 7:-2].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            env.agent.reset(qpos)

            robot_force = 0
            total_agent_contacts = 0
            for _ in range(args.init_check_scene_steps):
                env.scene.step()

                robot_force = robot_force + env.agent.robot.get_net_contact_forces(
                    env.agent.robot_link_names
                ).norm(dim=-1).sum(dim=-1)
                total_agent_contacts += num_agent_contacts(env.scene.get_contacts())

            if (
                robot_force.item() == 0
                and total_agent_contacts == args.init_check_scene_steps
            ):
                spawn_pos.append(env.agent.robot.pose.p[0])
                spawn_qpos.append(env.agent.robot.qpos[0])
                spawn_articulation_qpos.append(subtask_articulation.qpos[0])

        subtask_uid_to_spawn_data[subtask.uid] = dict(
            robot_pos=torch.stack(spawn_pos),
            robot_qpos=torch.stack(spawn_qpos),
            articulation_qpos=torch.stack(spawn_articulation_qpos),
        )

    return subtask_uid_to_spawn_data


def gen_navigate_spawn_data(
    proc_num,
    args: GenSpawnPositionArgs,
    scene_builder_cls,
    task_plans: List[TaskPlan],
):
    build_config_name = task_plans[0].build_config_name
    env = make_env(scene_builder_cls)

    scene_builder: ReplicaCADRearrangeSceneBuilder = env.scene_builder
    build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
    init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

    grasping_spawns = dict()
    if args.task == "set_table":
        task_obj_names = [
            "013_apple",
            "024_bowl",
        ]
    else:
        task_obj_names = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "008_pudding_box",
            "007_tuna_fish_can",
            "009_gelatin_box",
            "010_potted_meat_can",
            "024_bowl",
        ]
    for obj_name in task_obj_names:
        with open(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange/grasp_poses"
            / args.task
            / obj_name
            / "grasp_poses.pt",
            "rb",
        ) as spawns_fp:
            grasping_spawns[obj_name] = torch.load(spawns_fp)

    subtask_uid_to_spawn_data = dict()
    env.reset(
        seed=args.seed + proc_num,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )

    agent_bodies = []
    for link in env.agent.robot.links:
        agent_bodies += link._bodies
    agent_bodies = set(agent_bodies)
    num_agent_contacts = lambda contacts: len(
        [c for c in contacts if any([b in agent_bodies for b in c.bodies])]
    )

    for tp in tqdm(task_plans):
        env.reset(
            seed=args.seed + proc_num,
            options=dict(
                reconfigure=False,
                init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
            ),
        )

        assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], NavigateSubtask)
        subtask: NavigateSubtask = tp.subtasks[0]

        subtask_obj = None
        if subtask.obj_id is not None:
            subtask_obj = scene_builder.movable_objects[f"env-0_{subtask.obj_id}"]
            obj_name = subtask.obj_id.split("-")[0]

            subtask_obj_bodies = set(subtask_obj._bodies)
            num_subtask_obj_contacts = lambda contacts: len(
                [
                    c
                    for c in contacts
                    if any([b in subtask_obj_bodies for b in c.bodies])
                ]
            )

        navigable_positions = torch.tensor(
            scene_builder.navigable_positions[0].vertices
        )

        if args.task == "set_table":
            subtask_articulation = None
            # NOTE (arth): for set table:
            #       obj_id is None && remove_obj_id is None -> nav to open subtask, leave articulation closed
            #       obj_id is not None -> nav to Place (after Picking from open articulation)
            #       remove_obj_id is not None -> nav to close subtask
            if subtask.obj_id is not None or subtask.remove_obj_id is not None:
                subtask_articulation = scene_builder.articulations[
                    f"env-0_{subtask.articulation_config.articulation_id}"
                ]
                if subtask.articulation_config.articulation_type == "fridge":
                    min_open_qpos_frac = 0.75
                elif subtask.articulation_config.articulation_type == "kitchen_counter":
                    min_open_qpos_frac = 0.9
                else:
                    raise NotImplementedError(
                        f"{subtask.articulation_config.articulation_type=} not supported"
                    )
            spawn_articulation_qpos = []
        elif args.task == "prepare_groceries":
            spawn_lookat = torch.tensor(
                subtask.prev_goal_pos
                if subtask.prev_goal_pos is not None
                else [-1, 0, 0.02]
            )

        spawn_pos, spawn_qpos = [], []
        spawn_obj_raw_pose_wrt_tcp = []
        while len(spawn_pos) < args.num_spawns_per_task_plan:
            if subtask_obj is None:
                qpos = torch.tensor(env.agent.keyframes["rest"].qpos).unsqueeze(0)
            else:
                grasp_spawn = grasping_spawns[obj_name]
                grasp_spawn_num = torch.randint(
                    low=0, high=len(grasp_spawn["success_qpos"]), size=(1,)
                )
                qpos = grasp_spawn["success_qpos"][grasp_spawn_num]
                obj_raw_pose_wrt_tcp = grasp_spawn["success_obj_raw_pose_wrt_tcp"][
                    grasp_spawn_num
                ].clone()
                if torch.norm(Pose.create(obj_raw_pose_wrt_tcp).p[0]) > 0.15:
                    continue

            env.reset(
                seed=args.seed + proc_num,
                options=dict(
                    reconfigure=False,
                    init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                ),
            )

            if args.task == "set_table":
                if subtask_articulation is None:
                    new_subtask_articulation_qpos = torch.zeros((1, 1))
                else:
                    robot_init_pos = env.agent.robot.pose.p
                    robot_init_pos[:, :2] = 99999
                    env.agent.robot.set_pose(Pose.create_from_pq(p=robot_init_pos))

                    new_subtask_articulation_qpos = subtask_articulation.qpos * 0
                    joint_qmax = subtask_articulation.qlimits[
                        :,
                        subtask.articulation_config.articulation_handle_active_joint_idx,
                        1,
                    ]
                    joint_qmin = subtask_articulation.qlimits[
                        :,
                        subtask.articulation_config.articulation_handle_active_joint_idx,
                        0,
                    ]
                    joint_qrange = joint_qmax - joint_qmin
                    joint_open_qmin = joint_qrange * min_open_qpos_frac + joint_qmin
                    rand_joint_qpos = (
                        torch.rand_like(joint_qmax) * (joint_qmax - joint_open_qmin)
                    ) + joint_open_qmin
                    new_subtask_articulation_qpos[
                        :,
                        subtask.articulation_config.articulation_handle_active_joint_idx,
                    ] = rand_joint_qpos
                    subtask_articulation.set_qpos(new_subtask_articulation_qpos)

            if args.task == "prepare_groceries":
                # NOTE (arth): since the robot stays in the kitchen during prepare_groceries, we constrain spawn positions
                _spawn_loc_radius = (
                    0.5 if subtask.prev_goal_pos is None else args.spawn_loc_radius
                )
                positions_wrt_centers = navigable_positions - spawn_lookat[:2]
                dists = torch.norm(positions_wrt_centers, dim=-1)

                new_navigable_positions = navigable_positions[dists < _spawn_loc_radius]
                positions_wrt_centers = positions_wrt_centers[dists < _spawn_loc_radius]
                dists = dists[dists < _spawn_loc_radius]
                if subtask.prev_goal_pos is None:
                    # NOTE (arth): if init at default start pose, just add some rot noise
                    rots = (
                        torch.rand_like(positions_wrt_centers[..., 1]) - 0.5
                    ) * torch.pi
                else:
                    rots = (
                        torch.sign(positions_wrt_centers[..., 1])
                        * torch.arccos(positions_wrt_centers[..., 0] / dists)
                        + torch.pi
                    ) % (2 * torch.pi)
            else:
                # NOTE (arth): for nav tidy_house and set_table, we spawn all over the apartment
                new_navigable_positions = navigable_positions
                rots = None

            # spawn to try
            spawn_num = torch.randint(
                low=0, high=len(new_navigable_positions), size=(1,)
            )

            # base pos
            loc = new_navigable_positions[spawn_num]
            robot_pos = env.agent.robot.pose.p
            robot_pos[:, :2] = loc
            robot_pos[:, :2] += torch.clamp(
                torch.normal(0, 0.1, robot_pos[:, :2].shape), -0.2, 0.2
            )
            robot_pos[:, 2] = 0.02
            env.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            # base rot
            if rots is None:
                rot = torch.rand((1,)) * 2 * torch.pi
            else:
                rot = rots[spawn_num]
            qpos[:, 2] = rot
            qpos[:, 2:3] += torch.clamp(
                torch.normal(0, 0.25, qpos[:, 2:3].shape), -0.5, 0.5
            )
            # arm qpos
            qpos[:, 5:6] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 5:6].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            qpos[:, 7:-2] += torch.clamp(
                torch.normal(0, args.robot_init_qpos_noise / 2, qpos[:, 7:-2].shape),
                -args.robot_init_qpos_noise,
                args.robot_init_qpos_noise,
            )
            env.agent.reset(qpos)
            grasp_tcp_raw_pose = env.agent.tcp.pose.raw_pose.clone()
            robot_pose_p = env.agent.robot.pose.p[0].clone()
            robot_qpos = env.agent.robot.qpos[0].clone()

            robot_force = None
            total_agent_contacts = 0
            for _ in range(args.init_check_scene_steps):
                env.scene.step()
                rforce = (
                    env.agent.robot.get_net_contact_forces(env.agent.robot_link_names)
                    .norm(dim=-1)
                    .sum(dim=-1)
                )
                if robot_force is None:
                    robot_force = rforce
                else:
                    robot_force += rforce
                total_agent_contacts += num_agent_contacts(env.scene.get_contacts())
            robot_spawn_success = (
                robot_force.item() == 0
                and total_agent_contacts == args.init_check_scene_steps
            )

            obj_spawn_success = False
            if subtask_obj is not None and robot_spawn_success:
                env.reset(
                    seed=args.seed + proc_num,
                    options=dict(
                        reconfigure=False,
                        init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
                    ),
                )

                env.agent.robot.set_pose(sapien.Pose(p=[99999, 99999, 99999]))
                subtask_obj.set_pose(
                    Pose.create(grasp_tcp_raw_pose) * Pose.create(obj_raw_pose_wrt_tcp)
                )

                env.scene.step()

                obj_force = subtask_obj.get_net_contact_forces().norm(dim=-1)
                obj_spawn_success = (
                    obj_force.item() == 0
                    and num_subtask_obj_contacts(env.scene.get_contacts()) == 0
                )

            if robot_spawn_success and (subtask_obj is None or obj_spawn_success):
                spawn_pos.append(robot_pose_p)
                spawn_qpos.append(robot_qpos)
                if subtask_obj is None:
                    spawn_obj_raw_pose_wrt_tcp.append(torch.zeros(7))
                else:
                    spawn_obj_raw_pose_wrt_tcp.append(obj_raw_pose_wrt_tcp[0])
                if args.task == "set_table":
                    spawn_articulation_qpos.append(new_subtask_articulation_qpos[0])

        if args.task == "set_table":
            subtask_uid_to_spawn_data[subtask.uid] = dict(
                robot_pos=torch.stack(spawn_pos),
                robot_qpos=torch.stack(spawn_qpos),
                obj_raw_pose_wrt_tcp=torch.stack(spawn_obj_raw_pose_wrt_tcp),
                articulation_qpos=torch.stack(spawn_articulation_qpos),
            )
        else:
            subtask_uid_to_spawn_data[subtask.uid] = dict(
                robot_pos=torch.stack(spawn_pos),
                robot_qpos=torch.stack(spawn_qpos),
                obj_raw_pose_wrt_tcp=torch.stack(spawn_obj_raw_pose_wrt_tcp),
            )

    return subtask_uid_to_spawn_data


def gen_spawn_data(
    proc_num: int,
    build_config_name: str,
    args: GenSpawnPositionArgs,
):
    with torch.random.fork_rng():
        print("starting", proc_num)
        torch.manual_seed(args.seed + proc_num)

        task_plan_fp = args.root / args.task / args.subtask / args.split / "all.json"
        plan_data = plan_data_from_file(task_plan_fp)
        scene_builder_cls = {
            "ReplicaCADTidyHouseTrain": ReplicaCADTidyHouseTrainSceneBuilder,
            "ReplicaCADTidyHouseVal": ReplicaCADTidyHouseValSceneBuilder,
            "ReplicaCADPrepareGroceriesTrain": ReplicaCADPrepareGroceriesTrainSceneBuilder,
            "ReplicaCADPrepareGroceriesVal": ReplicaCADPrepareGroceriesValSceneBuilder,
            "ReplicaCADSetTableTrain": ReplicaCADSetTableTrainSceneBuilder,
            "ReplicaCADSetTableVal": ReplicaCADSetTableValSceneBuilder,
        }[plan_data.dataset]
        task_plans = [
            tp for tp in plan_data.plans if tp.build_config_name == build_config_name
        ]
        return dict(
            pick=gen_pick_spawn_data,
            place=gen_place_spawn_data,
            open=gen_open_spawn_data,
            close=gen_close_spawn_data,
            navigate=gen_navigate_spawn_data,
        )[args.subtask](proc_num, args, scene_builder_cls, task_plans)


def parse_args(args=None) -> GenSpawnPositionArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        help="Long-horizon task to make for. Valid values include tidy_house, prepare_groceries, and set_table. Defaults to tidy_house.",
        default="tidy_house",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        help="Subtask to make TaskPlans for. Valid values include pick and place. Defaults to pick.",
        default="pick",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to make TaskPlans for. Valid values include train and val. Defaults to train.",
        default="train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for sampling navigable positions",
        default=2024,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Num workers in multiprocessing pool.",
        default=4,
    )
    return GenSpawnPositionArgs(**parser.parse_args(args).__dict__)


def main():
    import time

    stime = time.time()
    args = parse_args()

    task_plan_fp = args.root / args.task / args.subtask / args.split / "all.json"
    plan_data = plan_data_from_file(task_plan_fp)

    build_config_names = set()
    for tp in plan_data.plans:
        build_config_names.add(tp.build_config_name)
    build_config_names = sorted(list(build_config_names))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers) as pool:
        procs = [
            pool.apply_async(
                gen_spawn_data,
                (
                    proc_num,
                    bc,
                    args,
                ),
            )
            for proc_num, bc in enumerate(build_config_names)
        ]
        results = [res.get() for res in procs]

    subtask_uid_to_spawn_data = dict()
    for res in results:
        subtask_uid_to_spawn_data.update(res)

    output_dir = (
        ASSET_DIR
        / "scene_datasets/replica_cad_dataset/rearrange/spawn_data"
        / args.task
        / args.subtask
        / args.split
    )
    os.makedirs(output_dir, exist_ok=True)

    torch.save(subtask_uid_to_spawn_data, output_dir / f"spawn_data.pt")

    print("finished in", time.time() - stime, "seconds")


if __name__ == "__main__":
    main()
