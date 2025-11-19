import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from lxml import etree as ET
from tqdm import tqdm

import numpy as np
import transforms3d

import sapien
from sapien.wrapper.urchin import URDF

from mani_skill import ASSET_DIR
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.replicacad.rearrange import (
    ReplicaCADPrepareGroceriesTrainSceneBuilder,
    ReplicaCADPrepareGroceriesValSceneBuilder,
    ReplicaCADSetTableTrainSceneBuilder,
    ReplicaCADSetTableValSceneBuilder,
    ReplicaCADTidyHouseTrainSceneBuilder,
    ReplicaCADTidyHouseValSceneBuilder,
)

from mshab.envs.planner import (
    ArticulationConfig,
    CloseSubtask,
    NavigateSubtask,
    OpenSubtask,
    PickSubtask,
    PlaceSubtask,
    PlanData,
    Subtask,
    TaskPlan,
    plan_data_from_file,
)
from mshab.utils.array import all_equal


@dataclass
class GenTaskPlanArgs:
    task: str
    subtask: str
    split: str

    root = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans"

    def __post_init__(self):
        assert self.task in ["tidy_house", "prepare_groceries", "set_table"]
        assert not self.subtask in ["open", "close"] or self.task == "set_table"
        assert self.subtask in [
            "pick",
            "place",
            "open",
            "close",
            "sequential",
            "navigate",
        ]


def get_name_num(name_num, split=":"):
    name, num = name_num.split(split)
    name, num = name[:-1], int(num)
    return name, num


def gen_pick_task_plans(scene_builder):
    task_plans = []
    cached_urdfs = dict()
    for init_config_name in tqdm(scene_builder._rearrange_configs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange",
                init_config_name,
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        build_config_name = Path(episode_json["scene_id"]).name

        for actor_id, (base_articulation_id, link_num) in zip(
            episode_json["info"]["object_labels"].keys(),
            episode_json["target_receptacles"],
        ):
            actor_id, actor_num = get_name_num(actor_id)

            if args.task == "set_table" or (
                args.task == "prepare_groceries" and "fridge" in base_articulation_id
            ):
                articulation_id, articulation_num = get_name_num(base_articulation_id)

                if articulation_id in cached_urdfs:
                    robot = cached_urdfs[articulation_id]
                else:
                    urdf_dir = (
                        ASSET_DIR
                        / "scene_datasets/replica_cad_dataset/urdf"
                        / articulation_id
                    )
                    urdf_file = urdf_dir / f"{articulation_id}.urdf"
                    with open(urdf_file, "r") as f:
                        urdf_string = f.read()
                    xml = ET.fromstring(urdf_string.encode("utf-8"))
                    robot = URDF._from_xml(
                        xml, str(urdf_dir.absolute()), lazy_load_meshes=True
                    )
                    cached_urdfs[articulation_id] = robot

                if articulation_id == "fridge":
                    marker_link_name = "top_door"
                    handle_link_idx = [link.name for link in robot.links].index(
                        marker_link_name
                    )
                else:
                    marker_link_name = robot.links[link_num].name
                    handle_link_idx = link_num

                # NOTE (arth): For RCAD kitchen_counter and fridge, the above URDF object has one less joint than links.
                #       SAPIEN left-appends an extra `''` joint so the links and joints correspond, so we follow suit
                # NOTE (arth): not sure if this holds true for *all* articulations, or just these specific ones. mileage may vary
                padded_joint_names = [""] + [j.name for j in robot.joints]
                assert len(padded_joint_names) == len(
                    robot.links
                ), "Mismatch in num joints and links"

                handle_active_joint_idx = robot.actuated_joint_names.index(
                    padded_joint_names[handle_link_idx]
                )

                task_plans.append(
                    TaskPlan(
                        subtasks=[
                            PickSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                articulation_config=ArticulationConfig(
                                    articulation_type=articulation_id,
                                    articulation_id=f"{articulation_id}-{articulation_num}",
                                    articulation_handle_link_idx=handle_link_idx,
                                    articulation_handle_active_joint_idx=handle_active_joint_idx,
                                ),
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )
            else:
                task_plans.append(
                    TaskPlan(
                        subtasks=[PickSubtask(obj_id=f"{actor_id}-{actor_num}")],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )

    print(f"generated {len(task_plans)} task_plans")
    return task_plans


def gen_place_task_plans(scene_builder):
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    task_plans = []
    cached_urdfs = dict()
    for init_config_name in tqdm(scene_builder._rearrange_configs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange",
                init_config_name,
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        build_config_name = Path(episode_json["scene_id"]).name

        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/configs/scenes",
                build_config_name,
            ),
            "rb",
        ) as rcad_f:
            rcad_json = json.load(rcad_f)

        objects = defaultdict(list)
        for instance in (
            rcad_json["object_instances"] + rcad_json["articulated_object_instances"]
        ):
            objects[instance["template_name"]].append(
                (instance["rotation"], instance["translation"])
            )

        target_poses = episode_json["targets"]

        for (
            base_actor_id,
            (goal_receptacle_id, link_num),
        ) in zip(
            episode_json["info"]["object_labels"].keys(),
            episode_json["goal_receptacles"],
        ):
            actor_id, actor_num = get_name_num(base_actor_id)
            goal_receptacle, goal_num = get_name_num(goal_receptacle_id)

            receptacle_config_fp = osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/configs/objects",
                f"{goal_receptacle}.object_config.json",
            )
            if not Path(receptacle_config_fp).exists():
                receptacle_config_fp = osp.join(
                    ASSET_DIR,
                    "scene_datasets/replica_cad_dataset/urdf",
                    goal_receptacle,
                    f"{goal_receptacle}.ao_config.json",
                )

            with open(
                receptacle_config_fp,
                "rb",
            ) as receptacle_config_f:
                receptacle_config = json.load(receptacle_config_f)

            receptable_rot, receptable_pos = objects[goal_receptacle][goal_num]

            if goal_receptacle == "kitchen_counter":
                goal_configs = list(receptacle_config["user_defined"].values())[:3]
            elif goal_receptacle == "fridge":
                goal_configs = list(receptacle_config["user_defined"].values())[1:2]
            else:
                goal_configs = list(receptacle_config["user_defined"].values())[:1]

            goal_rectangle_corners = []
            for gc in goal_configs:
                goal_transform = sapien.Pose()
                gc_up = gc.get("up", None)
                if gc_up is not None:
                    hab_up = np.array([0, 1, 0])
                    hab_up = hab_up / np.linalg.norm(hab_up)
                    gc_up = np.array(gc_up)
                    gc_up = gc_up / np.linalg.norm(gc_up)

                    goal_q = np.array(
                        [1 + np.dot(gc_up, hab_up), *np.cross(gc_up, hab_up)]
                    )
                    goal_q = goal_q / np.linalg.norm(goal_q)
                    goal_transform = sapien.Pose(q=goal_q)

                goal_area_center, goal_area_half_size = (
                    gc["position"],
                    gc["scale"],
                )
                x, y, z = (goal_transform * sapien.Pose(p=goal_area_center)).p
                dx, dy, dz = (goal_transform * sapien.Pose(p=goal_area_half_size)).p

                grcs = [
                    (
                        sapien.Pose(q=q)
                        * sapien.Pose(
                            p=receptable_pos,
                            q=receptable_rot,
                        )
                        * sapien.Pose(p=[x + sx * dx, y - dy, z + sz * dz])
                    ).p.tolist()
                    for sx, sz in [(1, 1), (-1, 1), (-1, -1), (1, -1)]
                ]
                # sort in ABCD order
                goal_rectangle_corners.append(grcs)
            goal_rectangle_corners = tuple(goal_rectangle_corners)

            goal_pos = (
                sapien.Pose(q=q) * sapien.Pose(matrix=target_poses[base_actor_id])
            ).p.tolist()
            if len(goal_rectangle_corners) == 1:
                goal_rectangle_corners = goal_rectangle_corners[0]
            else:
                grc_and_dist = []
                P = np.array(goal_pos)[:2]
                for grc in goal_rectangle_corners:
                    grc2d = np.array(grc)[..., :2]
                    A, B, C, D = grc2d
                    grc_area = np.linalg.norm(np.cross(B - A, D - A))
                    triangle_part_areas = []
                    for p0, p1, p2 in [(A, B, P), (B, C, P), (C, D, P), (A, D, P)]:
                        triangle_part_areas.append(
                            np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2
                        )
                    total_part_areas = np.sum(triangle_part_areas)
                    grc_and_dist.append((grc, np.abs(grc_area - total_part_areas)))
                grc_and_dist = sorted(grc_and_dist, key=lambda x: x[1])
                goal_rectangle_corners, area_diff = grc_and_dist[0]
                assert area_diff <= 1e-2, [x[1] for x in grc_and_dist]

            if (
                args.task == "set_table"
                and (
                    "fridge" in goal_receptacle_id
                    or "kitchen_counter" in goal_receptacle_id
                )
            ) or (args.task == "prepare_groceries" and "fridge" in goal_receptacle_id):
                articulation_id, articulation_num = get_name_num(goal_receptacle_id)

                if articulation_id in cached_urdfs:
                    robot = cached_urdfs[articulation_id]
                else:
                    urdf_dir = (
                        ASSET_DIR
                        / "scene_datasets/replica_cad_dataset/urdf"
                        / articulation_id
                    )
                    urdf_file = urdf_dir / f"{articulation_id}.urdf"
                    with open(urdf_file, "r") as f:
                        urdf_string = f.read()
                    xml = ET.fromstring(urdf_string.encode("utf-8"))
                    robot = URDF._from_xml(
                        xml, str(urdf_dir.absolute()), lazy_load_meshes=True
                    )
                    cached_urdfs[articulation_id] = robot

                if articulation_id == "fridge":
                    marker_link_name = "top_door"
                    handle_link_idx = [link.name for link in robot.links].index(
                        marker_link_name
                    )
                else:
                    marker_link_name = robot.links[link_num].name
                    handle_link_idx = link_num

                # NOTE (arth): For RCAD kitchen_counter and fridge, the above URDF object has one less joint than links.
                #       SAPIEN left-appends an extra `''` joint so the links and joints correspond, so we follow suit
                # NOTE (arth): not sure if this holds true for *all* articulations, or just these specific ones. mileage may vary
                padded_joint_names = [""] + [j.name for j in robot.joints]
                assert len(padded_joint_names) == len(
                    robot.links
                ), "Mismatch in num joints and links"

                handle_active_joint_idx = robot.actuated_joint_names.index(
                    padded_joint_names[handle_link_idx]
                )

                task_plans.append(
                    TaskPlan(
                        subtasks=[
                            PlaceSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                goal_pos=goal_pos,
                                goal_rectangle_corners=goal_rectangle_corners,
                                articulation_config=ArticulationConfig(
                                    articulation_type=articulation_id,
                                    articulation_id=f"{articulation_id}-{articulation_num}",
                                    articulation_handle_link_idx=handle_link_idx,
                                    articulation_handle_active_joint_idx=handle_active_joint_idx,
                                ),
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )
            else:
                task_plans.append(
                    TaskPlan(
                        subtasks=[
                            PlaceSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                goal_pos=goal_pos,
                                goal_rectangle_corners=goal_rectangle_corners,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )

    print(f"generated {len(task_plans)} task_plans")
    return task_plans


def gen_open_task_plans(scene_builder):
    task_plans = []
    cached_urdfs = dict()
    for init_config_name in tqdm(scene_builder._rearrange_configs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange",
                init_config_name,
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        build_config_name = Path(episode_json["scene_id"]).name

        markers = episode_json["markers"]

        for actor_id, (base_articulation_id, link_num) in zip(
            episode_json["info"]["object_labels"].keys(),
            episode_json["target_receptacles"],
        ):
            actor_id, actor_num = get_name_num(actor_id)
            articulation_id, articulation_num = get_name_num(base_articulation_id)

            relevant_markers = [
                m for m in markers if m["params"]["object"] == base_articulation_id
            ]
            assert (
                len(relevant_markers) > 0
            ), f"No handle offset maker given for {base_articulation_id}"

            if articulation_id in cached_urdfs:
                robot = cached_urdfs[articulation_id]
            else:
                urdf_dir = (
                    ASSET_DIR
                    / "scene_datasets/replica_cad_dataset/urdf"
                    / articulation_id
                )
                urdf_file = urdf_dir / f"{articulation_id}.urdf"
                with open(urdf_file, "r") as f:
                    urdf_string = f.read()
                xml = ET.fromstring(urdf_string.encode("utf-8"))
                robot = URDF._from_xml(
                    xml, str(urdf_dir.absolute()), lazy_load_meshes=True
                )
                cached_urdfs[articulation_id] = robot

            if articulation_id == "fridge":
                marker_link_name = "top_door"
                handle_link_idx = [link.name for link in robot.links].index(
                    marker_link_name
                )
            else:
                marker_link_name = robot.links[link_num].name
                handle_link_idx = link_num

            # NOTE (arth): For RCAD kitchen_counter and fridge, the above URDF object has one less joint than links.
            #       SAPIEN left-appends an extra `''` joint so the links and joints correspond, so we follow suit
            # NOTE (arth): not sure if this holds true for *all* articulations, or just these specific ones. mileage may vary
            padded_joint_names = [""] + [j.name for j in robot.joints]
            assert len(padded_joint_names) == len(
                robot.links
            ), "Mismatch in num joints and links"

            handle_active_joint_idx = robot.actuated_joint_names.index(
                padded_joint_names[handle_link_idx]
            )

            for m in relevant_markers:
                if m["params"]["link"] == marker_link_name:
                    articulation_marker = m
                    break
            else:
                raise AttributeError(
                    f"Couldn't find marker for articulation link: {base_articulation_id=} {link_num=} {marker_link_name=}"
                )

            task_plans.append(
                TaskPlan(
                    subtasks=[
                        OpenSubtask(
                            obj_id=f"{actor_id}-{actor_num}",
                            articulation_type=articulation_id,
                            articulation_id=f"{articulation_id}-{articulation_num}",
                            articulation_handle_active_joint_idx=handle_active_joint_idx,
                            articulation_handle_link_idx=handle_link_idx,
                            articulation_relative_handle_pos=articulation_marker[
                                "params"
                            ]["offset"],
                        )
                    ],
                    build_config_name=build_config_name,
                    init_config_name=init_config_name,
                )
            )

    print(f"generated {len(task_plans)} task_plans")
    return task_plans


def gen_close_task_plans(scene_builder):
    task_plans = []
    cached_urdfs = dict()
    for init_config_name in tqdm(scene_builder._rearrange_configs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange",
                init_config_name,
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        build_config_name = Path(episode_json["scene_id"]).name

        markers = episode_json["markers"]

        for actor_id, (base_articulation_id, link_num) in zip(
            episode_json["info"]["object_labels"].keys(),
            episode_json["target_receptacles"],
        ):
            actor_id, actor_num = get_name_num(actor_id)
            articulation_id, articulation_num = get_name_num(base_articulation_id)

            relevant_markers = [
                m for m in markers if m["params"]["object"] == base_articulation_id
            ]
            assert (
                len(relevant_markers) > 0
            ), f"No handle offset maker given for {base_articulation_id}"

            if articulation_id in cached_urdfs:
                robot = cached_urdfs[articulation_id]
            else:
                urdf_dir = (
                    ASSET_DIR
                    / "scene_datasets/replica_cad_dataset/urdf"
                    / articulation_id
                )
                urdf_file = urdf_dir / f"{articulation_id}.urdf"
                with open(urdf_file, "r") as f:
                    urdf_string = f.read()
                xml = ET.fromstring(urdf_string.encode("utf-8"))
                robot = URDF._from_xml(
                    xml, str(urdf_dir.absolute()), lazy_load_meshes=True
                )
                cached_urdfs[articulation_id] = robot

            if articulation_id == "fridge":
                marker_link_name = "top_door"
                handle_link_idx = [link.name for link in robot.links].index(
                    marker_link_name
                )
            else:
                marker_link_name = robot.links[link_num].name
                handle_link_idx = link_num

            # NOTE (arth): For RCAD kitchen_counter and fridge, the above URDF object has one less joint than links.
            #       SAPIEN left-appends an extra `''` joint so the links and joints correspond, so we follow suit
            # NOTE (arth): not sure if this holds true for *all* articulations, or just these specific ones. mileage may vary
            padded_joint_names = [""] + [j.name for j in robot.joints]
            assert len(padded_joint_names) == len(
                robot.links
            ), "Mismatch in num joints and links"

            handle_active_joint_idx = robot.actuated_joint_names.index(
                padded_joint_names[handle_link_idx]
            )

            for m in relevant_markers:
                if m["params"]["link"] == marker_link_name:
                    articulation_marker = m
                    break
            else:
                raise AttributeError(
                    f"Couldn't find marker for articulation link: {base_articulation_id=} {link_num=} {marker_link_name=}"
                )

            task_plans.append(
                TaskPlan(
                    subtasks=[
                        CloseSubtask(
                            articulation_type=articulation_id,
                            articulation_id=f"{articulation_id}-{articulation_num}",
                            articulation_handle_active_joint_idx=handle_active_joint_idx,
                            articulation_handle_link_idx=handle_link_idx,
                            articulation_relative_handle_pos=articulation_marker[
                                "params"
                            ]["offset"],
                            remove_obj_id=(
                                f"{actor_id}-{actor_num}"
                                if articulation_id == "kitchen_counter"
                                else None
                            ),
                        )
                    ],
                    build_config_name=build_config_name,
                    init_config_name=init_config_name,
                )
            )

    print(f"generated {len(task_plans)} task_plans")
    return task_plans


def gen_navigate_task_plans(scene_builder):
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    task_plans = []
    cached_urdfs = dict()

    def get_urdf_info(articulation_id, link_num):
        if articulation_id in cached_urdfs:
            robot = cached_urdfs[articulation_id]
        else:
            urdf_dir = (
                ASSET_DIR / "scene_datasets/replica_cad_dataset/urdf" / articulation_id
            )
            urdf_file = urdf_dir / f"{articulation_id}.urdf"
            with open(urdf_file, "r") as f:
                urdf_string = f.read()
            xml = ET.fromstring(urdf_string.encode("utf-8"))
            robot = URDF._from_xml(xml, str(urdf_dir.absolute()), lazy_load_meshes=True)
            cached_urdfs[articulation_id] = robot

        if articulation_id == "fridge":
            marker_link_name = "top_door"
            handle_link_idx = [link.name for link in robot.links].index(
                marker_link_name
            )
        else:
            marker_link_name = robot.links[link_num].name
            handle_link_idx = link_num

        # NOTE (arth): For RCAD kitchen_counter and fridge, the above URDF object has one less joint than links.
        #       SAPIEN left-appends an extra `''` joint so the links and joints correspond, so we follow suit
        # NOTE (arth): not sure if this holds true for *all* articulations, or just these specific ones. mileage may vary
        padded_joint_names = [""] + [j.name for j in robot.joints]
        assert len(padded_joint_names) == len(
            robot.links
        ), "Mismatch in num joints and links"

        handle_active_joint_idx = robot.actuated_joint_names.index(
            padded_joint_names[handle_link_idx]
        )
        return handle_link_idx, handle_active_joint_idx

    for init_config_name in tqdm(scene_builder._rearrange_configs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange",
                init_config_name,
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        build_config_name = Path(episode_json["scene_id"]).name

        init_poses = dict()
        num_per_rigid_obj = defaultdict(int)
        for obj_path, raw_pose_T in episode_json["rigid_objs"]:
            obj_name_from_path = obj_path.replace(".object_config.json", "")
            init_poses[
                f"{obj_name_from_path}-{num_per_rigid_obj[obj_name_from_path]}"
            ] = sapien.Pose(q=q) * sapien.Pose(matrix=raw_pose_T)
            num_per_rigid_obj[obj_name_from_path] += 1
        target_poses = episode_json["targets"]

        prev_goal_pos = None
        for (
            base_actor_id,
            (base_target_receptacle_id, target_link_num),
            (base_goal_receptacle_id, goal_link_num),
        ) in zip(
            episode_json["info"]["object_labels"].keys(),
            episode_json["target_receptacles"],
            episode_json["goal_receptacles"],
        ):
            actor_id, actor_num = get_name_num(base_actor_id)
            target_receptacle_id, target_receptacle_num = get_name_num(
                base_target_receptacle_id
            )
            goal_receptacle_id, goal_receptacle_num = get_name_num(
                base_goal_receptacle_id
            )

            obj_pos = init_poses[f"{actor_id}-{actor_num}"].p.tolist()
            goal_pos = (
                sapien.Pose(q=q) * sapien.Pose(matrix=target_poses[base_actor_id])
            ).p.tolist()

            if args.task == "tidy_house":
                task_plans += [
                    # prev goal -> obj
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                goal_pos=obj_pos, prev_goal_pos=prev_goal_pos
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    ),
                    # obj -> place goal
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                goal_pos=goal_pos,
                                prev_goal_pos=obj_pos,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    ),
                ]

            elif args.task == "prepare_groceries":
                # prev goal -> obj
                target_articulation_config = None
                if "fridge" in base_target_receptacle_id:
                    handle_link_idx, handle_active_joint_idx = get_urdf_info(
                        target_receptacle_id, target_link_num
                    )
                    target_articulation_config = ArticulationConfig(
                        articulation_type=target_receptacle_id,
                        articulation_id=f"{target_receptacle_id}-{target_receptacle_num}",
                        articulation_handle_link_idx=handle_link_idx,
                        articulation_handle_active_joint_idx=handle_active_joint_idx,
                    )
                task_plans.append(
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                goal_pos=obj_pos,
                                prev_goal_pos=prev_goal_pos,
                                articulation_config=target_articulation_config,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )

                # obj -> place goal
                goal_articulation_config = None
                if "fridge" in base_goal_receptacle_id:
                    handle_link_idx, handle_active_joint_idx = get_urdf_info(
                        goal_receptacle_id, goal_link_num
                    )
                    goal_articulation_config = ArticulationConfig(
                        articulation_type=goal_receptacle_id,
                        articulation_id=f"{goal_receptacle_id}-{goal_receptacle_num}",
                        articulation_handle_link_idx=handle_link_idx,
                        articulation_handle_active_joint_idx=handle_active_joint_idx,
                    )
                task_plans.append(
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                goal_pos=goal_pos,
                                prev_goal_pos=obj_pos,
                                articulation_config=goal_articulation_config,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    )
                )

            elif args.task == "set_table":
                target_handle_link_idx, target_handle_active_joint_idx = get_urdf_info(
                    target_receptacle_id, target_link_num
                )
                target_articulation_config = ArticulationConfig(
                    articulation_type=target_receptacle_id,
                    articulation_id=f"{target_receptacle_id}-{target_receptacle_num}",
                    articulation_handle_link_idx=target_handle_link_idx,
                    articulation_handle_active_joint_idx=target_handle_active_joint_idx,
                )

                task_plans += [
                    # nav to articulation for open
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                articulation_config=target_articulation_config,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    ),
                    # nav to receptacle for place
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                obj_id=f"{actor_id}-{actor_num}",
                                goal_pos=goal_pos,
                                articulation_config=target_articulation_config,
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    ),
                    # nav to opened articulation for close
                    TaskPlan(
                        subtasks=[
                            NavigateSubtask(
                                articulation_config=target_articulation_config,
                                remove_obj_id=f"{actor_id}-{actor_num}",
                            )
                        ],
                        build_config_name=build_config_name,
                        init_config_name=init_config_name,
                    ),
                ]

            else:
                raise NotImplementedError(args.task)

            prev_goal_pos = goal_pos

    print(f"generated {len(task_plans)} task_plans")
    return task_plans


def parse_args(args=None) -> GenTaskPlanArgs:
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
    return GenTaskPlanArgs(**parser.parse_args(args).__dict__)


def main():
    task_name = "ReplicaCAD" + args.task.title().replace("_", "")
    dataset = task_name + args.split.capitalize()
    scene_builder: SceneBuilder = {
        "ReplicaCADPrepareGroceriesTrain": ReplicaCADPrepareGroceriesTrainSceneBuilder,
        "ReplicaCADPrepareGroceriesVal": ReplicaCADPrepareGroceriesValSceneBuilder,
        "ReplicaCADSetTableTrain": ReplicaCADSetTableTrainSceneBuilder,
        "ReplicaCADSetTableVal": ReplicaCADSetTableValSceneBuilder,
        "ReplicaCADTidyHouseTrain": ReplicaCADTidyHouseTrainSceneBuilder,
        "ReplicaCADTidyHouseVal": ReplicaCADTidyHouseValSceneBuilder,
    }[dataset](None)

    if args.subtask == "sequential":
        task_plan_order = dict(
            tidy_house=["pick", "place"] * 5,
            prepare_groceries=["pick", "place"] * 3,
            set_table=["open", "pick", "place", "close"] * 2,
        )[args.task]

        subtask_to_plan_data: Dict[str, PlanData] = dict()
        for subtask_name in set(task_plan_order):
            fp = args.root / args.task / subtask_name / args.split / "all.json"
            print("Loading", str(fp) + "...")
            subtask_to_plan_data[subtask_name] = plan_data_from_file(fp)

        num_plans_per_subtask = [
            len(data.plans) for data in subtask_to_plan_data.values()
        ]
        num_times_each_subtask_in_sequential_plan = [
            sum([int(x == subtask_name) for x in task_plan_order])
            for subtask_name in set(task_plan_order)
        ]
        assert all_equal(num_plans_per_subtask) and all_equal(
            num_times_each_subtask_in_sequential_plan
        ), f"{num_plans_per_subtask}, {num_times_each_subtask_in_sequential_plan}"
        num_plans_to_make = num_plans_per_subtask[0] // (
            num_times_each_subtask_in_sequential_plan[0]
        )
        assert (
            num_plans_per_subtask[0] / num_times_each_subtask_in_sequential_plan[0]
            == num_plans_to_make
        )

        subtask_to_pointer = dict(
            (subtask_name, 0) for subtask_name in set(task_plan_order)
        )

        plans = []
        for tp_num in tqdm(range(num_plans_to_make)):
            subtasks = []
            subtask_build_config_names = []
            subtask_init_config_names = []
            for subtask_num, subtask_name in enumerate(task_plan_order):
                nav_subtask = NavigateSubtask()
                nav_subtask.uid = (
                    f"{args.task}-{args.subtask}-{args.split}-{tp_num}-{subtask_num*2}"
                )
                nav_subtask.composite_subtask_uids = [nav_subtask.uid]
                subtasks.append(nav_subtask)

                s_ptr = subtask_to_pointer[subtask_name]
                s_plan_data = subtask_to_plan_data[subtask_name].plans[s_ptr]

                subtask_build_config_names.append(s_plan_data.build_config_name)
                subtask_init_config_names.append(s_plan_data.init_config_name)
                next_subtask = s_plan_data.subtasks[0]

                next_subtask.uid = f"{args.task}-{args.subtask}-{args.split}-{tp_num}-{subtask_num*2+1}"
                next_subtask.composite_subtask_uids = [next_subtask.uid]

                if next_subtask.type == "close":
                    next_subtask.remove_obj_id = None

                subtasks.append(next_subtask)
                subtask_to_pointer[subtask_name] += 1
            plans.append(
                TaskPlan(
                    subtasks=subtasks,
                    build_config_name=subtask_build_config_names[0],
                    init_config_name=subtask_init_config_names[0],
                )
            )

        assert all(
            [
                subtask_to_pointer[subtask_name] == num_plans_per_subtask[0]
                for subtask_name in set(task_plan_order)
            ]
        ), f"{subtask_to_pointer}, {num_plans_per_subtask[0]}"
        plan_data = PlanData(dataset=dataset, plans=plans)

        out_fp = args.root / args.task / args.subtask / args.split / "all.json"
        os.makedirs(out_fp.parent, exist_ok=True)
        with open(out_fp, "w+") as f:
            json.dump(
                asdict(
                    plan_data,
                ),
                f,
            )

        return

    all_task_plans: List[TaskPlan] = dict(
        pick=gen_pick_task_plans,
        place=gen_place_task_plans,
        open=gen_open_task_plans,
        close=gen_close_task_plans,
        navigate=gen_navigate_task_plans,
    )[args.subtask](scene_builder)

    for tp_num, tp in enumerate(all_task_plans):
        for subtask_num, subtask in enumerate(tp.subtasks):
            subtask.uid = (
                f"{args.task}-{args.subtask}-{args.split}-{tp_num}-{subtask_num}"
            )
            subtask.composite_subtask_uids = [subtask.uid]

    plan_data_by_targ_id = dict(all=PlanData(dataset=dataset, plans=all_task_plans))

    if args.subtask != "navigate":
        if args.subtask in ["pick", "place"]:
            targ_attribute_id = "obj_id"
        elif args.subtask in ["open", "close"]:
            targ_attribute_id = "articulation_id"

        all_targ_ids = set()
        for tp in all_task_plans:
            all_targ_ids.add(tp.subtasks[0].__dict__[targ_attribute_id].split("-")[0])

        plan_data_by_targ_id.update(
            dict(
                (
                    tid,
                    PlanData(
                        dataset=dataset,
                        plans=[
                            tp
                            for tp in all_task_plans
                            if tp.subtasks[0].__dict__[targ_attribute_id].split("-")[0]
                            == tid
                        ],
                    ),
                )
                for tid in all_targ_ids
            )
        )

    for tid, plan_data in plan_data_by_targ_id.items():
        out_fp = args.root / args.task / args.subtask / args.split / f"{tid}.json"
        os.makedirs(out_fp.parent, exist_ok=True)
        with open(out_fp, "w+") as f:
            json.dump(
                asdict(
                    plan_data,
                ),
                f,
            )


if __name__ == "__main__":
    args = parse_args()
    main()
