from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import trimesh

import numpy as np
import torch

from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_invert,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose

from .planner import NavigateSubtask, NavigateSubtaskConfig, TaskPlan
from .sequential_task import GOAL_POSE_Q
from .subtask import SubtaskTrainEnv
from rlinf.envs.maniskillhab.utils.array import tensor_intersection, tensor_intersection_idx


@register_env("NavigateSubtaskTrain-v0", max_episode_steps=1000)
class NavigateSubtaskTrainEnv(SubtaskTrainEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    navigate_cfg = NavigateSubtaskConfig(
        horizon=1000,
        ee_rest_thresh=0.05,
        robot_cumulative_force_limit=torch.inf,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        dist_fn: Literal["euclidean", "geodesic"] = "geodesic",
        use_rot_rew: bool = True,
        restrict_articulation_target_area: bool = True,
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], NavigateSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {NavigateSubtask.__name__} long"

        self.subtask_cfg = self.navigate_cfg
        self.use_geodesic = dist_fn == "geodesic"
        self.use_rot_rew = use_rot_rew
        self.restrict_articulation_target_area = restrict_articulation_target_area

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _load_scene(self, options):
        super()._load_scene(options)
        self.premade_goal_list: List[Actor] = [
            self._make_goal(
                radius=0.15,
                name="goal_0",
                goal_type="sphere",
            )
        ]

        if self.use_geodesic:
            bcis = common.to_tensor(self.build_config_idxs, device=self.device)
            unique_bcis = bcis.unique()
            build_configs = [
                self.scene_builder.build_configs[bci] for bci in unique_bcis
            ]
            self.env_idx_to_floor_map = torch.searchsorted(unique_bcis, bcis)

            # NOTE (arth): we precompute navigable floor amps and all-pairs distances
            #   to efficiently compute (approx) geodesic distances at runtime
            floor_map_verts = [
                common.to_tensor(
                    trimesh.load(
                        Path(ASSET_DIR)
                        / "scene_datasets/replica_cad_dataset/configs/scenes"
                        / (
                            Path(bc).stem
                            + f".{str(self.robot_uids)}.navigable_positions_simplified.obj"
                        )
                    ).vertices,
                    device=self.device,
                )
                for bc in build_configs
            ]
            floor_map_all_pairs_dists = [
                common.to_tensor(
                    np.load(
                        Path(ASSET_DIR)
                        / "scene_datasets/replica_cad_dataset/configs/scenes"
                        / (
                            Path(bc).stem
                            + f".{str(self.robot_uids)}.navigable_positions_simplified_all_pairs_dist.npy"
                        )
                    ),
                    device=self.device,
                )
                for bc in build_configs
            ]
            max_map_verts = max([x.size(0) for x in floor_map_verts])
            self.floor_map_verts = torch.full(
                (len(build_configs), max_map_verts, 2),
                100,
                device=self.device,
                dtype=torch.float,
            )
            self.floor_map_all_pairs_dists = torch.full(
                (len(build_configs), max_map_verts, max_map_verts),
                100,
                device=self.device,
                dtype=torch.float,
            )
            for i in range(len(build_configs)):
                verts = floor_map_verts[i]
                dists = floor_map_all_pairs_dists[i]
                self.floor_map_verts[i, : verts.size(0)] = verts
                self.floor_map_all_pairs_dists[i, : dists.size(0), : dists.size(1)] = (
                    dists
                )

    def _apply_premade_spawns(self, env_idx, options: Dict):
        with torch.device(self.device):
            current_subtask = self.task_plan[0]
            batched_spawn_data = defaultdict(list)
            spawn_selection_idxs = options.get(
                "spawn_selection_idxs", [None] * env_idx.numel()
            )
            for env_num, subtask_uid, spawn_selection_idx in zip(
                env_idx,
                [
                    current_subtask.composite_subtask_uids[env_num]
                    for env_num in env_idx
                ],
                spawn_selection_idxs,
            ):
                spawn_data: Dict[str, torch.Tensor] = self.spawn_data[subtask_uid]
                for k, v in spawn_data.items():
                    if spawn_selection_idx is None:
                        spawn_selection_idx = torch.randint(
                            low=0, high=len(v), size=(1,)
                        )
                        self.spawn_selection_idxs[env_num] = spawn_selection_idx.item()
                    elif isinstance(spawn_selection_idx, int):
                        self.spawn_selection_idxs[env_num] = spawn_selection_idx
                        spawn_selection_idx = [spawn_selection_idx]
                    batched_spawn_data[k].append(v[spawn_selection_idx])
            for k, v in batched_spawn_data.items():
                if (
                    self.subtask_articulations[0] is not None
                    and k == "articulation_qpos"
                ):
                    articulation_qpos = torch.zeros(
                        (env_idx.numel(), self.subtask_articulations[0].max_dof),
                        device=self.device,
                        dtype=torch.float,
                    )
                    for i in range(env_idx.numel()):
                        articulation_qpos[i, : v[i].size(1)] = v[i].squeeze(0)
                    batched_spawn_data[k] = articulation_qpos
                else:
                    batched_spawn_data[k] = torch.cat(v, dim=0)
            if "robot_pos" in batched_spawn_data:
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=batched_spawn_data["robot_pos"])
                )
            if "robot_qpos" in batched_spawn_data:
                self.agent.robot.set_qpos(batched_spawn_data["robot_qpos"])
            subtask_obj = self.subtask_objs[0]
            if subtask_obj is not None and "obj_raw_pose_wrt_tcp" in batched_spawn_data:
                # NOTE (arth): for the navigate subtask, the subtask_obj may not exist in all
                #       envs we assume batched_spawn_data has shape (len(env_idx), ...), and
                #       select only the dims applicable to the subtask_obj
                obj_reset_idxs = tensor_intersection_idx(
                    env_idx, subtask_obj._scene_idxs
                )
                if self.gpu_sim_enabled:
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                subtask_obj.set_pose(
                    Pose.create(
                        self.agent.tcp.pose.raw_pose[
                            tensor_intersection(env_idx, subtask_obj._scene_idxs)
                        ]
                    )  # NOTE (arth): use tcp.pose for spawning for slightly better accuracy
                    * Pose.create(
                        batched_spawn_data["obj_raw_pose_wrt_tcp"][obj_reset_idxs]
                    )
                )
            subtask_articulation = self.subtask_articulations[0]
            if (
                subtask_articulation is not None
                and "articulation_qpos" in batched_spawn_data
            ):
                articulation_reset_idxs = tensor_intersection_idx(
                    env_idx, subtask_articulation._scene_idxs
                )
                subtask_articulation.set_qpos(
                    batched_spawn_data["articulation_qpos"][articulation_reset_idxs]
                )
                subtask_articulation.set_qvel(
                    torch.zeros_like(
                        batched_spawn_data["articulation_qpos"][articulation_reset_idxs]
                    )
                )
                if self.gpu_sim_enabled and len(env_idx) == self.num_envs:
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene.px.step()
                    self.scene._gpu_fetch_all()
            if self.merged_link is not None:
                merged_link_reset_idxs = tensor_intersection_idx(
                    env_idx, self.merged_link._scene_idxs
                )
                if merged_link_reset_idxs.numel():
                    if self.gpu_sim_enabled:
                        self.scene._gpu_apply_all()
                        self.scene._gpu_fetch_all()
                    goal_pose = Pose.create(
                        self.subtask_goals[-1].pose.raw_pose[env_idx].clone()
                    )
                    goal_pose.p[merged_link_reset_idxs] = self.merged_link.pose.p[
                        tensor_intersection_idx(self.merged_link._scene_idxs, env_idx)
                    ]
                    self.subtask_goals[-1].set_pose(goal_pose)

    # NOTE (arth): sometimes will need to nav w/ object, sometimes not
    #       override _merge_navigate_subtasks to allow obj in only some envs
    def _merge_navigate_subtasks(
        self,
        env_idx: torch.Tensor,
        last_subtask0,
        subtask_num: int,
        parallel_subtasks: List[NavigateSubtask],
    ):
        obj_ids, obj_sis = [], []
        art_ids, art_sis = [], []
        (
            replace_goal_with_link_ids_num,
            replace_goal_with_link_sis,
            replace_goal_with_link_nav_xrange,
            replace_goal_with_link_nav_yrange,
        ) = ([], [], [], [])
        remove_obj_ids, remove_obj_sis = [], []
        for i, subtask in enumerate(parallel_subtasks):
            if subtask.obj_id is not None:
                obj_sis.append(i)
                obj_ids.append(subtask.obj_id)
            if subtask.articulation_config is not None:
                art_sis.append(i)
                art_ids.append(subtask.articulation_config.articulation_id)
            if subtask.goal_pos is None or (
                subtask.obj_id is None and subtask.articulation_config is not None
            ):
                assert subtask.articulation_config is not None
                replace_goal_with_link_sis.append(i)
                replace_goal_with_link_ids_num.append(
                    (
                        subtask.articulation_config.articulation_id,
                        (
                            subtask.articulation_config.articulation_handle_link_idx
                            if subtask.articulation_config.articulation_type != "fridge"
                            else None
                        ),
                    )
                )
                replace_goal_with_link_nav_xrange.append(
                    dict(fridge=[0.933, 1.833], kitchen_counter=[0.3, 1.5])[
                        subtask.articulation_config.articulation_type
                    ]
                )
                replace_goal_with_link_nav_yrange.append(
                    dict(fridge=[-0.6, 0.6], kitchen_counter=[-0.6, 0.6])[
                        subtask.articulation_config.articulation_type
                    ]
                )
            if subtask.remove_obj_id is not None:
                remove_obj_sis.append(i)
                remove_obj_ids.append(subtask.remove_obj_id)

        if obj_ids:
            merged_obj = Actor.create_from_entities(
                [
                    self._get_actor_entity(actor_id=f"env-{i}_{oid}", env_num=i)
                    for i, oid in zip(obj_sis, obj_ids)
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(obj_sis, dtype=torch.int),
            )
            merged_obj.name = merged_obj_name = f"obj_{subtask_num}"

            self.agent_finger1_link = Link.create(
                [self.agent.finger1_link._objs[i] for i in obj_sis],
                self.scene,
                obj_sis,
            )
            self.agent_finger1_link.name = "agent_finger1_link_for_grasp"
            self.agent_finger2_link = Link.create(
                [self.agent.finger2_link._objs[i] for i in obj_sis],
                self.scene,
                obj_sis,
            )
            self.agent_finger2_link.name = "agent_finger2_link_for_grasp"
        else:
            merged_obj = None
            merged_obj_name = None
            self.agent_finger1_link = None
            self.agent_finger2_link = None
        self.subtask_objs.append(merged_obj)

        self.subtask_goals.append(self.premade_goal_list[subtask_num])
        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(
                q=GOAL_POSE_Q,
                p=[
                    (
                        parallel_subtasks[env_num].goal_pos
                        if parallel_subtasks[env_num].goal_pos is not None
                        else [0, 0, 0]
                    )
                    for env_num in env_idx
                ],
            )
        )

        if art_ids:
            merged_articulation = Articulation.create_from_physx_articulations(
                [
                    self._get_articulation_entity(f"env-{env_num}_{aid}", env_num)
                    for env_num, aid in zip(art_sis, art_ids)
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(art_sis, dtype=torch.int),
                _merged=True,
            )
            merged_articulation.name = f"articulation-{subtask_num}"
        else:
            merged_articulation = None
        self.subtask_articulations.append(merged_articulation)

        if replace_goal_with_link_ids_num:
            self.merged_link = Link.create(
                [
                    self._get_link_entity(
                        f"env-{env_num}_{link_aid}", env_num, link_idx
                    )
                    for env_num, (link_aid, link_idx) in zip(
                        replace_goal_with_link_sis, replace_goal_with_link_ids_num
                    )
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(replace_goal_with_link_sis, dtype=torch.int),
            )
            self.nav_merged_link_xrange = torch.tensor(
                replace_goal_with_link_nav_xrange, dtype=torch.float
            )
            self.nav_merged_link_yrange = torch.tensor(
                replace_goal_with_link_nav_yrange, dtype=torch.float
            )
        else:
            self.merged_link = None

        if remove_obj_ids:
            Actor.create_from_entities(
                [
                    self._get_actor_entity(actor_id=f"env-{i}_{roid}", env_num=i)
                    for i, roid in zip(remove_obj_sis, remove_obj_ids)
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(remove_obj_sis, dtype=torch.int),
            ).set_pose(Pose.create_from_pq(p=[-10_000, -10_000, -9000]))

        self.task_plan.append(
            NavigateSubtask(
                obj_id=merged_obj_name,
                goal_pos=self.subtask_goals[-1].pose.p,
            )
        )

    def _get_link_entity(
        self, articulation_id: str, env_num: int, link_num: Optional[int] = None
    ):
        ms_articulation = self.scene_builder.articulations[articulation_id]
        link = (
            ms_articulation.links[link_num]
            if link_num is not None
            else ms_articulation.root
        )
        return link._objs[link._scene_idxs.tolist().index(env_num)]

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def _compute_geodesic_disance(self, env_idx):
        agent_pos = self.agent.base_link.pose.p[env_idx, :2]
        goal_pos = self.subtask_goals[-1].pose.p[env_idx, :2]

        verts = self.floor_map_verts[self.env_idx_to_floor_map[env_idx]]
        dists = self.floor_map_all_pairs_dists[self.env_idx_to_floor_map[env_idx]]

        _, a_closest_vert = torch.min(
            torch.norm(agent_pos.unsqueeze(1) - verts, dim=2), dim=1
        )
        g_dist_to_vert, g_closest_vert = torch.min(
            torch.norm(goal_pos.unsqueeze(1) - verts, dim=2), dim=1
        )
        a_vert_to_g_vert_dist = dists[
            torch.arange(len(dists)), a_closest_vert, g_closest_vert
        ]

        # NOTE (arth): omit a_dist_to_vert since mesh has some gaps between vertices which agent might end up in
        return a_vert_to_g_vert_dist + g_dist_to_vert

    def _compute_distance(self, env_idx=None):
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        if self.use_geodesic:
            return self._compute_geodesic_disance(env_idx)
        else:
            return torch.norm(
                self.agent.base_link.pose.p[env_idx, :2]
                - self.subtask_goals[0].pose.p[env_idx, :2],
                dim=1,
            )

    def _is_navigated_close(self, env_idx: torch.Tensor, goal: Actor, _):
        assert (
            env_idx.numel() == self.num_envs
        ), f"{self.__name__} should have nav in every env"
        navigated_close = (
            torch.norm(
                goal.pose.p[env_idx, :2] - self.agent.base_link.pose.p[env_idx, :2],
                dim=1,
            )
            <= self.navigate_cfg.navigated_successfully_dist
        )

        if self.merged_link is not None and self.restrict_articulation_target_area:
            relative_pos_world = (
                self.agent.base_link.pose.p[self.merged_link._scene_idxs]
                - self.merged_link.pose.p
            )

            relative_pos_local = quaternion_apply(
                quaternion_invert(self.merged_link.pose.q),
                relative_pos_world,
            )

            navigated_close[self.merged_link._scene_idxs] &= (
                (self.nav_merged_link_xrange[:, 0] <= relative_pos_local[:, 0])
                & (relative_pos_local[:, 0] <= self.nav_merged_link_xrange[:, 1])
                & (self.nav_merged_link_yrange[:, 0] <= relative_pos_local[:, 2])
                & (relative_pos_local[:, 2] <= self.nav_merged_link_yrange[:, 1])
            )

        return navigated_close

    def _is_grasping_partial_env_obj(
        self, obj: Actor, env_idx, min_force=0.5, max_angle=85
    ):
        with torch.device(self.device):
            is_grasped = torch.zeros(self.num_envs, dtype=torch.bool)

            l_contact_forces = self.scene.get_pairwise_contact_forces(
                self.agent_finger1_link, obj
            )
            r_contact_forces = self.scene.get_pairwise_contact_forces(
                self.agent_finger2_link, obj
            )
            lforce = torch.linalg.norm(l_contact_forces, axis=1)
            rforce = torch.linalg.norm(r_contact_forces, axis=1)

            # direction to open the gripper
            ldirection = -self.agent_finger1_link.pose.to_transformation_matrix()[
                ..., :3, 1
            ]
            rdirection = self.agent_finger2_link.pose.to_transformation_matrix()[
                ..., :3, 1
            ]
            langle = common.compute_angle_between(ldirection, l_contact_forces)
            rangle = common.compute_angle_between(rdirection, r_contact_forces)
            lflag = torch.logical_and(
                lforce >= min_force, torch.rad2deg(langle) <= max_angle
            )
            rflag = torch.logical_and(
                rforce >= min_force, torch.rad2deg(rangle) <= max_angle
            )

            is_grasped[obj._scene_idxs] = torch.logical_and(lflag, rflag)
            return is_grasped[env_idx]

    def evaluate(self):
        info = super().evaluate()
        info["distance_from_goal"] = self._compute_distance()
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj = self.subtask_objs[0]

            begin_navigating = torch.ones(self.num_envs, dtype=torch.bool)
            if obj is not None:
                if len(obj._scene_idxs) != self.num_envs:
                    should_grasp = torch.zeros(self.num_envs, dtype=torch.bool)
                    should_grasp[obj._scene_idxs] = True
                    begin_navigating[should_grasp & ~info["is_grasped"]] = False
                    # info["should_grasp"] = should_grasp
                else:
                    begin_navigating[~info["is_grasped"]] = False
            begin_navigating_rew = 2 * begin_navigating
            reward += begin_navigating_rew
            # info["begin_navigating_rew"] = 2 * begin_navigating

            if torch.any(begin_navigating):
                done_moving = info["oriented_correctly"] & info["navigated_close"]
                done_navigating = info["navigated_close"]
                still_navigating = ~done_navigating

                done_moving &= begin_navigating
                done_navigating &= begin_navigating
                still_navigating &= begin_navigating

                # nav dist reward
                navigating_rew = 8 * (
                    1 - torch.tanh(info["distance_from_goal"][still_navigating] / 10)
                )
                reward[still_navigating] += navigating_rew

                # move forward when not done
                ego_base_lin_vel = quaternion_apply(
                    quaternion_invert(self.agent.base_link.pose.q[still_navigating]),
                    self.agent.base_link.linear_velocity[still_navigating],
                )
                still_navigating_vel_rew = 2 * torch.tanh(
                    2 * ego_base_lin_vel[:, 0].clip(min=0)
                )
                reward[still_navigating] += still_navigating_vel_rew

                # when done nav, give full from still_nav + 2
                reward[done_navigating] += 12

                if self.use_rot_rew:
                    # when done nav, orient towards obj
                    done_navigating_not_oriented = (
                        info["navigated_close"] & ~info["oriented_correctly"]
                    )
                    goal_pose_wrt_base = (
                        self.agent.base_link.pose[done_navigating_not_oriented].inv()
                        * self.subtask_goals[-1].pose[done_navigating_not_oriented]
                    )
                    targ = goal_pose_wrt_base.p[..., :2]
                    uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(-1).expand(
                        *targ.shape
                    )
                    rots = torch.sign(uc_targ[..., 1]) * torch.arccos(uc_targ[..., 0])
                    oriented_correctly_rew = 2 * (1 - torch.tanh(torch.abs(rots) / 2))
                    reward[done_navigating_not_oriented] += oriented_correctly_rew
                    reward[done_moving] += 2

                # stop moving base when done
                bqvel = torch.norm(self.agent.robot.qvel[done_moving, :3], dim=1)
                done_moving_vel_rew = 2 * (1 - torch.tanh(bqvel / 3))
                reward[done_moving] += done_moving_vel_rew

                # robot rest when done moving and ee at goal pos
                qvel = self.agent.robot.qvel[done_moving & info["ee_rest"], :-2]
                static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                reward[done_moving & info["ee_rest"]] += static_rew

                # x = torch.zeros_like(reward)
                # x[still_navigating] = navigating_rew
                # info["navigating_rew"] = x.clone()

                # x = torch.zeros_like(reward)
                # x[still_navigating] = still_navigating_vel_rew
                # info["still_moving_vel_rew"] = x.clone()

                # x = torch.zeros_like(reward)
                # x[done_navigating_not_oriented] = rots
                # info["rots"] = x.clone()

                # x = torch.zeros_like(reward)
                # x[done_navigating_not_oriented] = oriented_correctly_rew
                # info["oriented_correctly_rew"] = x.clone()

                # x = torch.zeros_like(reward)
                # x[done_moving] = done_moving_vel_rew
                # info["done_moving_vel_rew"] = x.clone()

                # x = torch.zeros_like(reward)
                # x[done_moving & info["ee_rest"]] = static_rew
                # info["static_rew"] = x.clone()

            # collisions
            step_no_col_rew = 5 * (
                1
                - torch.tanh(
                    3 * (torch.clamp(0.005 * info["robot_force"], min=0.2) - 0.2)
                )
            )
            reward += step_no_col_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 2 * (1 - torch.tanh(arm_to_resting_diff / 5))
            reward += arm_resting_orientation_rew

            # enforce ee at rest
            ee_to_rest_dist = torch.norm(
                self.agent.tcp_pose.p - self.ee_rest_world_pose.p, dim=1
            )
            ee_rest_rew = 2 * (1 - torch.tanh(3 * ee_to_rest_dist))
            reward += ee_rest_rew

            #

            # info["step_no_col_rew"] = step_no_col_rew
            # info["arm_resting_orientation_rew"] = arm_resting_orientation_rew
            # info["ee_rest_rew"] = ee_rest_rew

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 28.0 if self.use_rot_rew else 26.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
