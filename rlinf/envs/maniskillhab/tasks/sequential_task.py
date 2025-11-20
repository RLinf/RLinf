import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.random
import transforms3d

import sapien

from mani_skill.agents.robots import Fetch
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_invert,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation, Pose
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import vectorize_pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from .planner import (
    ArticulationConfig,
    CloseSubtask,
    CloseSubtaskConfig,
    NavigateSubtask,
    NavigateSubtaskConfig,
    OpenSubtask,
    OpenSubtaskConfig,
    PickSubtask,
    PickSubtaskConfig,
    PlaceSubtask,
    PlaceSubtaskConfig,
    Subtask,
    SubtaskConfig,
    TaskPlan,
)
from rlinf.envs.maniskillhab.utils.array import (
    all_equal,
    all_same_type,
    tensor_intersection,
    tensor_intersection_idx,
)


UNIQUE_SUCCESS_SUBTASK_TYPE = 100
GOAL_POSE_Q = transforms3d.quaternions.axangle2quat(
    np.array([0, 1, 0]), theta=np.deg2rad(90)
)


@register_env("SequentialTask-v0")
class SequentialTaskEnv(SceneManipulationEnv):
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

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Fetch

    EE_REST_POS_WRT_BASE = (0.5, 0, 1.25)
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
    )
    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        obj_goal_thresh=0.15,
        ee_rest_thresh=0.05,
    )
    navigate_cfg = NavigateSubtaskConfig(
        horizon=500,
        ee_rest_thresh=0.05,
        navigated_successfully_dist=2,
    )
    open_cfg = OpenSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_open_thresh_frac=dict(
            default=0.9,
            fridge=0.75,
            kitchen_counter=0.9,
        ),
    )
    close_cfg = CloseSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_close_thresh_frac=0.01,
    )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                temp_buffer_capacity=2**24,
                max_rigid_contact_count=2**23,
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
            ),
        )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        require_build_configs_repeated_equally_across_envs=True,
        randomize_build_configs_per_env=False,
        add_event_tracker_info=False,
        invisible_goals_in_human_render=False,
        task_cfgs=dict(),
        **kwargs,
    ):
        self.task_cfgs: Dict[str, SubtaskConfig] = dict(
            pick=self.pick_cfg,
            place=self.place_cfg,
            navigate=self.navigate_cfg,
            open=self.open_cfg,
            close=self.close_cfg,
        )

        task_cfg_update_dict = task_cfgs
        for k, v in task_cfg_update_dict.items():
            self.task_cfgs[k].update(v)

        assert all_equal(
            [len(plan.subtasks) for plan in task_plans]
        ), "All parallel task plans must be the same length"
        assert all(
            [
                all_same_type(parallel_subtasks)
                for parallel_subtasks in zip(*[plan.subtasks for plan in task_plans])
            ]
        ), "All parallel task plans must have same subtask types in same order"

        if randomize_build_configs_per_env:
            assert (
                not require_build_configs_repeated_equally_across_envs
            ), f"Received {randomize_build_configs_per_env=} but cannot randomize build configs per env with {require_build_configs_repeated_equally_across_envs=}"
        self._require_build_configs_repeated_equally_across_envs = (
            require_build_configs_repeated_equally_across_envs
        )
        self._randomize_build_configs_per_env = randomize_build_configs_per_env
        self._add_event_tracker_info = add_event_tracker_info
        self._invisible_goals_in_human_render = invisible_goals_in_human_render

        self.base_task_plans = dict(
            (tuple([subtask.uid for subtask in tp.subtasks]), tp) for tp in task_plans
        )
        self.bc_to_task_plans: Dict[str, List[TaskPlan]] = defaultdict(list)
        for tp in task_plans:
            self.bc_to_task_plans[tp.build_config_name].append(tp)

        self._init_config_names = set([tp.init_config_name for tp in task_plans])

        self.tp0 = task_plans[0]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # PROCESS TASKS
    # -------------------------------------------------------------------------------------------------

    def _merge_pick_subtasks(
        self, subtask_num: int, parallel_subtasks: List[PickSubtask]
    ):
        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(None)

        # NOTE (arth): this is a bit tricky, since prepare_groceries sometimes has an articulation config
        #   (pick from fridge), but sometimes does not (pick from countertop). however, when running tasks
        #   sequentially, the prepare_groceries parallel_subtasks will either all have or all not have
        #   articulation_config. this is unlike tasks like tidy_house, which never has articulation_config
        #   or set_table which always has articulation_config.
        #   later, we expect that all subtask_x has len(_objs) == num_envs, so we can't merge for prepare_groceries
        #   as-is. for now, we ignore the prepare_groceries case (since fridge is opened by ao_config anyways)
        #   but in the future we'll need to support a case with e.g. a modified set_table with pick from table
        if all(
            [subtask.articulation_config is not None for subtask in parallel_subtasks]
        ):
            merged_articulation_name = f"articulation-{subtask_num}"
            merged_articulation = (
                self._create_merged_articulation_from_articulation_ids(
                    [
                        subtask.articulation_config.articulation_id
                        for subtask in parallel_subtasks
                    ],
                    name=merged_articulation_name,
                    merging_different_articulations=True,
                )
            )
            self.subtask_articulations.append(merged_articulation)
        else:
            self.subtask_articulations.append(None)

        self.task_plan.append(
            PickSubtask(
                obj_id=merged_obj_name,
                # NOTE (arth): pick subtask might have different kinds of articulations
                #       merged into one Articulation view (e.g. fridge vs kitchen_counter)
                #       in this case, ArticulationConfig attributes like handle_link_idx
                #       don't make much sense (and aren't needed by the pick task anyways)
                articulation_config=None,
            )
        )

    def _merge_place_subtasks(
        self,
        env_idx: torch.Tensor,
        subtask_num: int,
        parallel_subtasks: List[PlaceSubtask],
    ):
        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_goal_pos = common.to_tensor(
            [subtask.goal_pos for subtask in parallel_subtasks]
        )
        merged_goal_rectangle_corners = common.to_tensor(
            [subtask.goal_rectangle_corners for subtask in parallel_subtasks]
        )

        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(q=GOAL_POSE_Q, p=merged_goal_pos[env_idx])
        )

        # NOTE (arth): see notes above regarding merged articulation config
        if all(
            [subtask.articulation_config is not None for subtask in parallel_subtasks]
        ):
            merged_articulation_name = f"articulation-{subtask_num}"
            merged_articulation = (
                self._create_merged_articulation_from_articulation_ids(
                    [
                        subtask.articulation_config.articulation_id
                        for subtask in parallel_subtasks
                    ],
                    name=merged_articulation_name,
                    merging_different_articulations=True,
                )
            )
            self.subtask_articulations.append(merged_articulation)
        else:
            self.subtask_articulations.append(None)

        self.task_plan.append(
            PlaceSubtask(
                obj_id=merged_obj_name,
                goal_pos=merged_goal_pos,
                goal_rectangle_corners=merged_goal_rectangle_corners,
                validate_goal_rectangle_corners=False,
                articulation_config=None,
            )
        )
        self.check_progressive_success_subtask_nums.append(subtask_num)

    def _merge_navigate_subtasks(
        self,
        env_idx: torch.Tensor,
        last_subtask0: Subtask,
        subtask_num: int,
        parallel_subtasks: List[NavigateSubtask],
    ):
        self.subtask_goals.append(None)
        self.subtask_articulations.append(None)

        if isinstance(last_subtask0, PickSubtask):
            last_subtask_obj = self.subtask_objs[-1]
            self.subtask_objs.append(last_subtask_obj)
            self.task_plan.append(
                NavigateSubtask(
                    obj_id=last_subtask_obj.name,
                )
            )
        else:
            self.subtask_objs.append(None)
            self.task_plan.append(NavigateSubtask())

    def _merge_open_subtasks(
        self, subtask_num: int, parallel_subtasks: List[OpenSubtask]
    ):
        subtask0 = parallel_subtasks[0]

        # NOTE (arth): current MS3 requires all parallel articulations be the same
        assert all_equal([subtask.articulation_type for subtask in parallel_subtasks])
        assert all_equal(
            [subtask.articulation_handle_link_idx for subtask in parallel_subtasks]
        )
        assert all_equal(
            [
                subtask.articulation_handle_active_joint_idx
                for subtask in parallel_subtasks
            ]
        )

        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_articulation_name = f"articulation-{subtask_num}"
        merged_articulation_relative_handle_pose = Pose.create_from_pq(
            p=[
                subtask.articulation_relative_handle_pos
                for subtask in parallel_subtasks
            ]
        )
        merged_articulation = self._create_merged_articulation_from_subtasks(
            parallel_subtasks, name=merged_articulation_name
        )
        self.subtask_articulations.append(merged_articulation)

        self.task_plan.append(
            OpenSubtask(
                obj_id=merged_obj_name,
                articulation_type=subtask0.articulation_type,
                articulation_id=merged_articulation_name,
                articulation_handle_link_idx=subtask0.articulation_handle_link_idx,
                articulation_handle_active_joint_idx=subtask0.articulation_handle_active_joint_idx,
                articulation_relative_handle_pos=merged_articulation_relative_handle_pose,
            )
        )

    def _merge_close_subtasks(
        self, subtask_num: int, parallel_subtasks: List[CloseSubtask]
    ):
        subtask0 = parallel_subtasks[0]

        # NOTE (arth): current MS3 requires all parallel articulations be the same
        assert all_equal([subtask.articulation_type for subtask in parallel_subtasks])
        assert all_equal(
            [subtask.articulation_handle_link_idx for subtask in parallel_subtasks]
        )
        assert all_equal(
            [
                subtask.articulation_handle_active_joint_idx
                for subtask in parallel_subtasks
            ]
        )

        self.subtask_objs.append(None)
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_articulation_name = f"articulation-{subtask_num}"
        merged_articulation_relative_handle_pose = Pose.create_from_pq(
            p=[
                subtask.articulation_relative_handle_pos
                for subtask in parallel_subtasks
            ]
        )
        merged_articulation = self._create_merged_articulation_from_subtasks(
            parallel_subtasks, name=merged_articulation_name
        )
        self.subtask_articulations.append(merged_articulation)

        # NOTE (arth): currently a band-aid solution, teleport bowl inside kitchen drawer away
        merged_removed_obj_name = None
        if subtask0.remove_obj_id is not None:
            merged_removed_obj_name = f"remove_obj_{subtask_num}"
            self._create_merged_actor_from_obj_ids(
                [subtask.remove_obj_id for subtask in parallel_subtasks],
                name=f"remove_obj_{subtask_num}",
            ).set_pose(
                Pose.create_from_pq(p=[-10_000, -10_000, -9000])
            )  # assume one entity per scene

        self.task_plan.append(
            CloseSubtask(
                articulation_type=subtask0.articulation_type,
                articulation_id=merged_articulation_name,
                articulation_handle_link_idx=subtask0.articulation_handle_link_idx,
                articulation_handle_active_joint_idx=subtask0.articulation_handle_active_joint_idx,
                articulation_relative_handle_pos=merged_articulation_relative_handle_pose,
                remove_obj_id=merged_removed_obj_name,
            )
        )

    def process_task_plan(
        self,
        env_idx: torch.Tensor,
        sampled_subtask_lists: List[List[Subtask]],
    ):

        self.subtask_objs: List[Actor] = []
        self.subtask_goals: List[Actor] = []
        self.subtask_articulations: List[Articulation] = []
        self.check_progressive_success_subtask_nums: List[int] = []

        # build new merged task_plan and merge actors of parallel task plants
        self.task_plan: List[Subtask] = []
        last_subtask0 = None
        for subtask_num, parallel_subtasks in enumerate(zip(*sampled_subtask_lists)):
            composite_subtask_uids = [subtask.uid for subtask in parallel_subtasks]
            subtask0: Subtask = parallel_subtasks[0]

            if isinstance(subtask0, PickSubtask):
                self._merge_pick_subtasks(subtask_num, parallel_subtasks)
            elif isinstance(subtask0, PlaceSubtask):
                self._merge_place_subtasks(env_idx, subtask_num, parallel_subtasks)
            elif isinstance(subtask0, NavigateSubtask):
                self._merge_navigate_subtasks(
                    env_idx, last_subtask0, subtask_num, parallel_subtasks
                )
            elif isinstance(subtask0, OpenSubtask):
                self._merge_open_subtasks(subtask_num, parallel_subtasks)
            elif isinstance(subtask0, CloseSubtask):
                self._merge_close_subtasks(subtask_num, parallel_subtasks)
            else:
                raise AttributeError(
                    f"{subtask0.type} {type(subtask0)} not yet supported"
                )

            last_subtask0 = subtask0

            self.task_plan[-1].composite_subtask_uids = composite_subtask_uids

        # add navigation goals for each Navigate Subtask depending on following subtask
        last_subtask = None
        for i, (subtask_obj, subtask_goal, subtask_articulation, subtask) in enumerate(
            zip(
                self.subtask_objs,
                self.subtask_goals,
                self.subtask_articulations,
                self.task_plan,
            )
        ):
            if isinstance(last_subtask, NavigateSubtask):
                if isinstance(subtask, PickSubtask):
                    self.subtask_goals[i - 1] = subtask_obj
                elif isinstance(subtask, PlaceSubtask):
                    self.subtask_goals[i - 1] = subtask_goal
                elif isinstance(subtask, OpenSubtask) or isinstance(
                    subtask, CloseSubtask
                ):
                    self.subtask_goals[i - 1] = (
                        subtask_articulation.links[subtask.articulation_handle_link_idx]
                        if subtask.articulation_type == "kitchen_counter"
                        else subtask_articulation
                    )
                    last_subtask.articulation_config = ArticulationConfig(
                        articulation_id=subtask.articulation_id,
                        articulation_type=subtask.articulation_type,
                        articulation_handle_link_idx=subtask.articulation_handle_link_idx,
                        articulation_handle_active_joint_idx=subtask.articulation_handle_active_joint_idx,
                    )
            last_subtask = subtask

        assert len(self.subtask_objs) == len(self.task_plan)
        assert len(self.subtask_goals) == len(self.task_plan)
        assert len(self.subtask_articulations) == len(self.task_plan)

        self.task_horizons = torch.tensor(
            [self.task_cfgs[subtask.type].horizon for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )
        self.task_ids = torch.tensor(
            [self.task_cfgs[subtask.type].task_id for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )

    def _get_actor_entity(self, actor_id: str, env_num: int):
        actor = self.scene_builder.movable_objects[actor_id]
        return actor._objs[actor._scene_idxs.tolist().index(env_num)]

    def _create_merged_actor_from_obj_ids(
        self,
        obj_ids: List[str],
        name: str = None,
    ):
        merged_obj = Actor.create_from_entities(
            [
                self._get_actor_entity(actor_id=f"env-{i}_{oid}", env_num=i)
                for i, oid in enumerate(obj_ids)
            ],
            scene=self.scene,
            scene_idxs=torch.arange(self.num_envs, dtype=int),
        )
        if name is not None:
            merged_obj.name = name
        return merged_obj

    def _create_merged_actor_from_subtasks(
        self,
        parallel_subtasks: List[Union[PickSubtask, PlaceSubtask]],
        name: str = None,
    ):
        return self._create_merged_actor_from_obj_ids(
            [subtask.obj_id for subtask in parallel_subtasks], name
        )

    def _get_articulation_entity(self, articulation_id: str, env_num: int):
        ms_articulation = self.scene_builder.articulations[articulation_id]
        return ms_articulation._objs[
            ms_articulation._scene_idxs.tolist().index(env_num)
        ]

    def _create_merged_articulation_from_articulation_ids(
        self,
        articulation_ids: List[str],
        name: str = None,
        merging_different_articulations: bool = False,
    ):
        scene_idx_to_physx_articulation_objs = [None for _ in range(self.num_envs)]
        for env_num, aid in enumerate(articulation_ids):
            scene_idx_to_physx_articulation_objs[env_num] = (
                self._get_articulation_entity(f"env-{env_num}_{aid}", env_num)
            )
        merged_articulation = Articulation.create_from_physx_articulations(
            scene_idx_to_physx_articulation_objs,
            scene=self.scene,
            scene_idxs=torch.arange(self.num_envs),
            _merged=merging_different_articulations,
        )
        merged_articulation.name = name
        return merged_articulation

    def _create_merged_articulation_from_subtasks(
        self,
        parallel_subtasks: List[Union[OpenSubtask, CloseSubtask]],
        name: str = None,
        merging_different_articulations: bool = False,
    ):
        return self._create_merged_articulation_from_articulation_ids(
            [subtask.articulation_id for subtask in parallel_subtasks],
            name,
            merging_different_articulations,
        )

    def _make_goal(
        self,
        pos: Union[Tuple[float, float, float], List[Tuple[float, float, float]]] = None,
        radius=0.15,
        name="goal_site",
        goal_type="sphere",
        color=[0, 1, 0, 1],
    ):
        if pos is not None:
            if len(pos) == self.num_envs:
                initial_pose = Pose.create_from_pq(p=pos)
            else:
                initial_pose = sapien.Pose(p=pos)
        else:
            initial_pose = sapien.Pose()

        if self._invisible_goals_in_human_render:
            color[-1] = 0

        if goal_type == "sphere":
            goal = actors.build_sphere(
                self.scene,
                radius=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        elif goal_type == "cube":
            goal = actors.build_cube(
                self.scene,
                half_size=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        elif goal_type == "cylinder":
            goal = actors.build_cylinder(
                self.scene,
                radius=radius,
                half_length=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        self._hidden_objects.append(goal)
        return goal

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # RESET/RECONFIGURE HANDLING
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):
        force_rew_ignore_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
        ]
        self.force_articulation_link_ids = [
            link.name
            for link in self.agent.robot.get_links()
            if link not in force_rew_ignore_links
        ]
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super()._after_reconfigure(options)

    def _load_scene(self, options):
        self.premade_goal_list: List[Actor] = []
        for subtask_num, subtask in enumerate(self.tp0.subtasks):
            if isinstance(subtask, PlaceSubtask):
                goal = self._make_goal(
                    radius=self.place_cfg.obj_goal_thresh,
                    name=f"goal_{subtask_num}",
                    goal_type=(
                        "cylinder"
                        if self.place_cfg.goal_type == "zone"
                        else self.place_cfg.goal_type
                    ),
                )
            elif isinstance(subtask, OpenSubtask) or isinstance(subtask, CloseSubtask):
                goal = self._make_goal(
                    radius=0.05,
                    name=f"goal_{subtask_num}",
                    goal_type="sphere",
                )
            else:
                goal = None
            self.premade_goal_list.append(goal)

        self.build_config_idx_to_task_plans: Dict[int, List[TaskPlan]] = dict()
        for bc in self.bc_to_task_plans.keys():
            self.build_config_idx_to_task_plans[
                self.scene_builder.build_config_names_to_idxs[bc]
            ] = self.bc_to_task_plans[bc]
            break  # mjwei TODO: here for debug we only need the first TaskPlan. In this way, the code can init fast and improve debug efficiency

        num_bcis = len(self.build_config_idx_to_task_plans.keys())

        assert (
            not self._require_build_configs_repeated_equally_across_envs
            or self.num_envs % num_bcis == 0
        ), f"These task plans cover {num_bcis} build configs, but received {self.num_envs} envs. Either change the task plan list, change num_envs, or set require_build_configs_repeated_equally_across_envs=False. Note if require_build_configs_repeated_equally_across_envs=False and num_envs % num_build_configs != 0, then a) if num_envs > num_build_configs, then some build configs might be built in more parallel envs than others (meaning associated task plans will be sampled more frequently), and b) if num_envs < num_build_configs, then some build configs might not be built at all (meaning associated task plans will not be used)."

        # if num_bcis < self.num_envs, repeat bcis and truncate at self.num_envs
        self.build_config_idxs: List[int] = options.get(
            "build_config_idxs",
            (
                self._episode_rng.choice(
                    list(self.build_config_idx_to_task_plans.keys()),
                    size=self.num_envs,
                    replace=True,
                ).tolist()
                if self._randomize_build_configs_per_env
                else np.repeat(
                    sorted(list(self.build_config_idx_to_task_plans.keys())),
                    np.ceil(self.num_envs / num_bcis),
                )[: self.num_envs].tolist()
            ),
        )
        self.num_task_plans_per_bci = torch.tensor(
            [
                len(self.build_config_idx_to_task_plans[bci])
                for bci in self.build_config_idxs
            ],
            device=self.device,
        )
        self.scene_builder.build(self.build_config_idxs, self._init_config_names)
        self.ee_rest_pos_wrt_base = Pose.create_from_pq(
            p=self.EE_REST_POS_WRT_BASE, device=self.device
        )
        self.subtask_pointer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.subtask_steps_left = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.ee_rest_goal = self._make_goal(
            radius=0.05,
            name="ee_rest_goal",
            goal_type="sphere",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options):
        with torch.device(self.device):
            self.robot_cumulative_force[env_idx] = 0

            if env_idx.numel() == self.num_envs:
                self.task_plan_idxs: torch.Tensor = options.get("task_plan_idxs", None)
            if self.task_plan_idxs is None or env_idx.numel() < self.num_envs:
                if self.task_plan_idxs is None:
                    self.task_plan_idxs = torch.zeros(self.num_envs, dtype=torch.int)
                low = torch.zeros(env_idx.numel(), dtype=torch.int)
                high = self.num_task_plans_per_bci[env_idx]
                size = (env_idx.numel(),)
                self.task_plan_idxs[env_idx] = (
                    torch.randint(2**63 - 1, size=size) % (high - low).int() + low.int()
                ).int()
            else:
                self.task_plan_idxs = self.task_plan_idxs.int()
            sampled_task_plans = [
                self.build_config_idx_to_task_plans[bci][tpi]
                for bci, tpi in zip(self.build_config_idxs, self.task_plan_idxs)
            ]
            self.init_config_idxs = [
                self.scene_builder.init_config_names_to_idxs[tp.init_config_name]
                for tp in sampled_task_plans
            ]
            super()._initialize_episode(env_idx, options)
            self.process_task_plan(
                env_idx,
                sampled_subtask_lists=[tp.subtasks for tp in sampled_task_plans],
            )

            self.subtask_pointer[env_idx] = 0
            self.subtask_steps_left[env_idx] = self.task_cfgs[
                self.task_plan[0].type
            ].horizon

            self.resting_qpos = torch.tensor(self.agent.keyframes["rest"].qpos[3:-2])

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # STATE RESET
    # -------------------------------------------------------------------------------------------------

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        state_dict["task_plan_idxs"] = self.task_plan_idxs.clone()
        state_dict["build_config_idxs"] = copy.deepcopy(self.build_config_idxs)
        state_dict["init_config_idxs"] = copy.deepcopy(self.init_config_idxs)

        state_dict["subtask_pointer"] = self.subtask_pointer.clone()
        state_dict["subtask_steps_left"] = self.subtask_steps_left.clone()
        state_dict["robot_cumulative_force"] = self.robot_cumulative_force.clone()

        return state_dict

    def set_state_dict(self, state_dict: Dict):
        task_plan_idxs = common.to_tensor(state_dict.get("task_plan_idxs"))
        build_config_idxs = state_dict.get("build_config_idxs")
        init_config_idxs = state_dict.get("init_config_idxs")

        assert torch.all(
            torch.tensor(self.build_config_idxs) == torch.tensor(build_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently built {self.build_config_idxs=}, state dict {build_config_idxs=}"

        self._initialize_episode(
            torch.arange(self.num_envs), options=dict(task_plan_idxs=task_plan_idxs)
        )

        assert torch.all(
            torch.tensor(self.init_config_idxs) == torch.tensor(init_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently init'd {self.init_config_idxs=}, state dict {init_config_idxs=}"

        self.subtask_pointer = state_dict.get("subtask_pointer")
        self.subtask_steps_left = state_dict.get("subtask_steps_left")
        self.robot_cumulative_force = state_dict.get("robot_cumulative_force")

        super().set_state_dict(state_dict)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # SUBTASK STATUS CHECKERS/UPDATERS
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):

        robot_force = (
            self.agent.robot.get_net_contact_forces(self.force_articulation_link_ids)
            .norm(dim=-1)
            .sum(dim=-1)
        )
        self.robot_cumulative_force += robot_force

        # NOTE (arth): update ee_rest_world_pose every step since robot moves
        self.ee_rest_world_pose: Pose = (
            self.agent.base_link.pose * self.ee_rest_pos_wrt_base
        )
        self.handle_world_poses: List[Union[Pose, None]] = []
        for subtask, articulation in zip(self.task_plan, self.subtask_articulations):
            if isinstance(subtask, OpenSubtask) or isinstance(subtask, CloseSubtask):
                self.handle_world_poses.append(
                    articulation.links[subtask.articulation_handle_link_idx].pose
                    * subtask.articulation_relative_handle_pos
                )
            else:
                self.handle_world_poses.append(None)

        subtask_success, success_checkers = self._subtask_check_success()
        progressive_task_success, progressive_task_checkers = (
            self._progressive_task_check_success()
        )

        success_checkers[
            "cumulative_force_within_limit"
        ] |= self.subtask_pointer >= len(self.task_plan)

        move_to_next_subtask = (
            subtask_success
            & success_checkers["cumulative_force_within_limit"]
            & progressive_task_success
        )
        self.subtask_pointer[move_to_next_subtask] += 1
        success = (
            self.subtask_pointer >= len(self.task_plan)
        ) & progressive_task_success
        # set robot_cumulative_force to 0 if evaluating new subtask
        self.robot_cumulative_force[
            move_to_next_subtask & (self.subtask_pointer < len(self.task_plan))
        ] = 0

        self.subtask_steps_left -= 1
        update_subtask_horizon = subtask_success & progressive_task_success & ~success
        self.subtask_steps_left[update_subtask_horizon] = self.task_horizons[
            self.subtask_pointer[update_subtask_horizon]
        ]

        fail = (
            ((self.subtask_steps_left <= 0) & ~success)
            | (~success_checkers["cumulative_force_within_limit"])
            | ~progressive_task_success
        )

        subtask_type = torch.full_like(
            self.subtask_pointer, UNIQUE_SUCCESS_SUBTASK_TYPE
        )
        subtask_type[~success] = self.task_ids[self.subtask_pointer[~success]]

        return dict(
            success=success,
            fail=fail,
            subtask=self.subtask_pointer,
            subtask_type=subtask_type,
            subtasks_steps_left=self.subtask_steps_left,
            robot_force=robot_force,
            robot_cumulative_force=self.robot_cumulative_force,
            **success_checkers,
            **progressive_task_checkers,
        )

    def _progressive_task_check_success(self):
        # check if prior tasks are still in completion state
        # NOTE (arth): while Habitat also checked things like holding vs not,
        #       in this setting only really place requires persisting checkers
        progressive_task_success = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        progressive_task_checkers = dict()
        for subtask_num in self.check_progressive_success_subtask_nums:
            subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer > subtask_num)[0]
            if isinstance(subtask, PlaceSubtask):
                subtask_progressive_success, subtask_progressive_checkers = (
                    self._place_check_success(
                        self.subtask_objs[subtask_num],
                        self.subtask_goals[subtask_num],
                        subtask.goal_rectangle_corners,
                        env_idx,
                        check_progressive_completion=True,
                    )
                )
            else:
                raise NotImplementedError(
                    f"{subtask.type} {type(subtask)} progressive completion checking not supported"
                )
            progressive_task_success[env_idx] &= subtask_progressive_success
            for k, v in subtask_progressive_checkers.items():
                new_k = f"{k}_progressive_{subtask_num}"
                if new_k not in progressive_task_checkers:
                    progressive_task_checkers[new_k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=v.dtype
                    )
                progressive_task_checkers[new_k][env_idx] = v

        return progressive_task_success, progressive_task_checkers

    def _subtask_check_success(self):
        subtask_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        success_checkers = dict()

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._pick_check_success(
                    self.subtask_objs[subtask_num],
                    env_idx,
                )
            elif isinstance(subtask, PlaceSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._place_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    subtask.goal_rectangle_corners,
                    env_idx,
                )
            elif isinstance(subtask, NavigateSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._navigate_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    (
                        subtask.articulation_config.articulation_type
                        if subtask.articulation_config is not None
                        else None
                    ),
                    env_idx,
                )
            elif isinstance(subtask, OpenSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._open_check_success(
                    self.subtask_articulations[subtask_num],
                    subtask.articulation_type,
                    subtask.articulation_handle_active_joint_idx,
                    subtask.articulation_handle_link_idx,
                    env_idx,
                )
            elif isinstance(subtask, CloseSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._close_check_success(
                    self.subtask_articulations[subtask_num],
                    subtask.articulation_handle_active_joint_idx,
                    subtask.articulation_handle_link_idx,
                    env_idx,
                )
            else:
                raise NotImplementedError(
                    f"{subtask.type} {type(subtask)} not supported"
                )

            for k, v in subtask_success_checkers.items():
                if k not in success_checkers:
                    success_checkers[k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=v.dtype
                    )
                success_checkers[k][env_idx] = v

        return subtask_success, success_checkers

    def _pick_check_success(
        self,
        obj: Actor,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.pick_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
        )
        robot_rest = torch.all(
            robot_rest_dist < self.pick_cfg.robot_resting_qpos_tolerance_grasping, dim=1
        )
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.pick_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)[
                    env_idx
                ],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)[
                    env_idx
                ],
                dim=1,
            )
        return (
            is_grasped
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _place_check_success(
        self,
        obj: Actor,
        obj_goal: Actor,
        goal_rectangle_corners: torch.Tensor,
        env_idx: torch.Tensor,
        check_progressive_completion=False,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        if self.place_cfg.goal_type == "zone":
            # (0 <= AM•AB <= AB•AB) and (0 <= AM•AD <=  AD•AD)
            As, Bs, Ds = (
                goal_rectangle_corners[env_idx, 0, :2],
                goal_rectangle_corners[env_idx, 1, :2],
                goal_rectangle_corners[env_idx, 3, :2],
            )
            Ms = obj.pose.p[env_idx, :2]

            AM = Ms - As
            AB = Bs - As
            AD = Ds - As

            AM_dot_AB = torch.sum(AM * AB, dim=1)
            AB_dot_AB = torch.sum(AB * AB, dim=1)
            AM_dot_AD = torch.sum(AM * AD, dim=1)
            AD_dot_AD = torch.sum(AD * AD, dim=1)

            xy_correct = (
                (0 <= AM_dot_AB)
                & (AM_dot_AB <= AB_dot_AB)
                & (0 <= AM_dot_AD)
                & (AM_dot_AD <= AD_dot_AD)
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.place_cfg.goal_type == "cylinder":
            xy_correct = (
                torch.norm(
                    obj.pose.p[env_idx, :2] - obj_goal.pose.p[env_idx, :2],
                    dim=1,
                )
                <= self.place_cfg.obj_goal_thresh
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.place_cfg.goal_type == "sphere":
            obj_at_goal = (
                torch.norm(
                    obj.pose.p[env_idx] - obj_goal.pose.p[env_idx],
                    dim=1,
                )
                <= self.place_cfg.obj_goal_thresh
            )
        else:
            raise NotImplementedError(
                f"{self.place_cfg.goal_type=} is not yet supported"
            )
        if check_progressive_completion:
            return obj_at_goal, dict(obj_at_goal=obj_at_goal)
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.place_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.place_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.place_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            obj_at_goal=obj_at_goal,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)[
                    env_idx
                ],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)[
                    env_idx
                ],
                dim=1,
            )
        return (
            ~is_grasped
            & obj_at_goal
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _is_grasping_partial_env_obj(obj, env_idx, max_angle=85):
        raise NotImplementedError()

    def _is_navigated_close(
        self,
        env_idx: torch.Tensor,
        goal: Actor,
        articulation_type: Optional[str] = None,
    ):
        navigated_close = (
            torch.norm(
                goal.pose.p[env_idx, :2] - self.agent.base_link.pose.p[env_idx, :2],
                dim=1,
            )
            <= self.navigate_cfg.navigated_successfully_dist
        )

        if isinstance(goal, Articulation) or isinstance(goal, Link):
            # NOTE (arth): assume nav to same articulation, since we check each parallel subtask at a time
            relative_pos_world = (
                self.agent.base_link.pose.p[env_idx] - goal.pose.p[env_idx]
            )

            relative_pos_local = quaternion_apply(
                quaternion_invert(goal.pose.q[env_idx]),
                relative_pos_world,
            )

            xrange = dict(fridge=[0.933, 1.833], kitchen_counter=[0.3, 1.5])[
                articulation_type
            ]
            yrange = dict(fridge=[-0.6, 0.6], kitchen_counter=[-0.6, 0.6])[
                articulation_type
            ]

            navigated_close &= (
                (xrange[0] <= relative_pos_local[:, 0])
                & (relative_pos_local[:, 0] <= xrange[1])
                & (yrange[0] <= relative_pos_local[:, 2])
                & (relative_pos_local[:, 2] <= yrange[1])
            )

        return navigated_close

    def _navigate_check_success(
        self,
        obj: Union[Actor, None],
        goal: Actor,
        articulation_type: str,
        env_idx: torch.Tensor,
    ):
        if obj is None:
            is_grasped = torch.zeros_like(env_idx, dtype=torch.bool)
        elif len(obj._scene_idxs) != self.num_envs:
            # NOTE (arth): this is so nav subtask train env can handle grasping when obj
            #   is only in some parallel envs -- not the cleanest implementation
            is_grasped = self._is_grasping_partial_env_obj(obj, env_idx, max_angle=30)
        else:
            is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]

        goal_pose_wrt_base = self.agent.base_link.pose.inv() * goal.pose
        targ = goal_pose_wrt_base.p[..., :2][env_idx]
        uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(-1).expand(*targ.shape)
        rots = torch.sign(uc_targ[..., 1]) * torch.arccos(uc_targ[..., 0])
        oriented_correctly = (
            torch.abs(rots) <= self.navigate_cfg.navigated_successfully_rot
        )

        navigated_close = self._is_navigated_close(env_idx, goal, articulation_type)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]

        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.navigate_cfg.robot_cumulative_force_limit
        )

        if self.navigate_cfg.ignore_arm_checkers:
            return (
                oriented_correctly
                & navigated_close
                & is_static
                & cumulative_force_within_limit,
                dict(
                    is_grasped=is_grasped,
                    oriented_correctly=oriented_correctly,
                    navigated_close=navigated_close,
                    is_static=is_static,
                    cumulative_force_within_limit=cumulative_force_within_limit,
                ),
            )

        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.navigate_cfg.ee_rest_thresh
        )
        if obj is None:
            robot_rest_dist = torch.abs(
                self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
            )
            robot_rest = torch.all(
                robot_rest_dist < self.navigate_cfg.robot_resting_qpos_tolerance, dim=1
            ) & (
                torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0])
                < 0.01
            )
        else:
            robot_rest_dist = torch.abs(
                self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
            )
            robot_rest = torch.all(
                robot_rest_dist
                < self.navigate_cfg.robot_resting_qpos_tolerance_grasping,
                dim=1,
            )
            if len(obj._scene_idxs) != self.num_envs:
                subtask_envs_with_obj = tensor_intersection_idx(
                    env_idx, obj._scene_idxs
                )
                robot_rest[subtask_envs_with_obj] &= torch.all(
                    robot_rest_dist[subtask_envs_with_obj]
                    < self.navigate_cfg.robot_resting_qpos_tolerance,
                    dim=1,
                )

        navigate_success = (
            oriented_correctly & navigated_close & ee_rest & robot_rest & is_static
        )
        if obj is not None:
            if len(obj._scene_idxs) != self.num_envs:
                subtask_envs_with_obj = tensor_intersection_idx(
                    env_idx, obj._scene_idxs
                )
                navigate_success[subtask_envs_with_obj] &= is_grasped[
                    subtask_envs_with_obj
                ]
            else:
                navigate_success &= is_grasped
        return (
            navigate_success & cumulative_force_within_limit,
            dict(
                is_grasped=is_grasped,
                oriented_correctly=oriented_correctly,
                navigated_close=navigated_close,
                ee_rest=ee_rest,
                robot_rest=robot_rest,
                is_static=is_static,
                cumulative_force_within_limit=cumulative_force_within_limit,
            ),
        )

    def _open_check_success(
        self,
        articulation: Articulation,
        articulation_type: str,
        active_joint_idx: int,
        link_idx: int,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(articulation.links[link_idx], max_angle=30)[
            env_idx
        ]
        articulation_open = articulation.qpos[env_idx, active_joint_idx] > (
            (
                articulation.qlimits[env_idx, active_joint_idx, 1]
                - articulation.qlimits[env_idx, active_joint_idx, 0]
            )
            * self.open_cfg.joint_qpos_open_thresh_frac[articulation_type]
            + articulation.qlimits[env_idx, active_joint_idx, 0]
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.open_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.open_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.open_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            articulation_open=articulation_open,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["handle_active_joint_qpos"] = articulation.qpos[
                env_idx, active_joint_idx
            ]
            subtask_checkers["handle_active_joint_qmin"] = articulation.qlimits[
                env_idx, active_joint_idx, 0
            ]
            subtask_checkers["handle_active_joint_qmax"] = articulation.qlimits[
                env_idx, active_joint_idx, 1
            ]
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            )
        return (
            articulation_open
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _close_check_success(
        self,
        articulation: Articulation,
        active_joint_idx: int,
        link_idx: int,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(articulation.links[link_idx], max_angle=30)[
            env_idx
        ]
        articulation_closed = articulation.qpos[env_idx, active_joint_idx] < (
            (
                articulation.qlimits[env_idx, active_joint_idx, 1]
                - articulation.qlimits[env_idx, active_joint_idx, 0]
            )
            * self.close_cfg.joint_qpos_close_thresh_frac
            + articulation.qlimits[env_idx, active_joint_idx, 0]
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.close_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.close_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.close_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            articulation_closed=articulation_closed,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["handle_active_joint_qpos"] = articulation.qpos[
                env_idx, active_joint_idx
            ]
            subtask_checkers["handle_active_joint_qmin"] = articulation.qlimits[
                env_idx, active_joint_idx, 0
            ]
            subtask_checkers["handle_active_joint_qmax"] = articulation.qlimits[
                env_idx, active_joint_idx, 1
            ]
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            )
        return (
            articulation_closed
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------

    def _get_obs_agent(self):
        agent_state = super()._get_obs_agent()
        agent_state["qpos"] = agent_state["qpos"][..., 3:]
        agent_state["qvel"] = agent_state["qvel"][..., 3:]
        return agent_state

    # NOTE (arth): for now, define keys that will always be added to obs. leave it to
    #       wrappers or task-specific envs to mask out unnecessary vals
    #       - subtasks that don't need that obs will set some default value
    #       - subtasks which need that obs will set value depending on subtask params
    def _get_obs_extra(self, info: Dict):
        base_pose_inv = self.agent.base_link.pose.inv()

        # all subtasks will have same computation for
        #       - tcp_pose_wrt_base :   tcp always there and is same link
        tcp_pose_wrt_base = vectorize_pose(base_pose_inv * self.agent.tcp.pose)

        #       - obj_pose_wrt_base :   different objs per subtask (or no obj)
        #       - goal_pos_wrt_base :   different goals per subtask (or no goal)
        obj_pose_wrt_base = torch.zeros(
            self.num_envs, 7, device=self.device, dtype=torch.float
        )
        goal_pos_wrt_base = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            subtask = self.task_plan[subtask_num]
            if self.subtask_objs[subtask_num] is not None:
                if len(self.subtask_objs[subtask_num]._scene_idxs) != self.num_envs:
                    env_scene_idx = tensor_intersection(
                        env_idx, self.subtask_objs[subtask_num]._scene_idxs
                    )
                    obj_pose_wrt_base[env_scene_idx] = vectorize_pose(
                        base_pose_inv[env_scene_idx]
                        * self.subtask_objs[subtask_num].pose
                    )
                else:
                    obj_pose_wrt_base[env_idx] = vectorize_pose(
                        base_pose_inv * self.subtask_objs[subtask_num].pose
                    )[env_idx]
            if self.subtask_goals[subtask_num] is not None:
                if isinstance(subtask, OpenSubtask) or isinstance(
                    subtask, CloseSubtask
                ):
                    goal_pos_wrt_base[env_idx] = (
                        base_pose_inv * self.handle_world_poses[subtask_num]
                    ).p[env_idx]
                else:
                    goal_pos_wrt_base[env_idx] = (
                        base_pose_inv * self.subtask_goals[subtask_num].pose
                    ).p[env_idx]

        # already computed during evaluation is
        #       - is_grasped    :   part of success criteria (or set default)
        is_grasped = info["is_grasped"]

        return dict(
            tcp_pose_wrt_base=tcp_pose_wrt_base,
            obj_pose_wrt_base=obj_pose_wrt_base,
            goal_pos_wrt_base=goal_pos_wrt_base,
            is_grasped=is_grasped,
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD (Ignored here)
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): this env does not have dense rewards since rewards are used for training subtasks.
    #       If need to train a subtask, extend this class to define a subtask
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.subtask_pointer

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # CAMERAS, SENSORS, AND RENDERING
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): also included the old "cameras" mode from MS2 since HAB renders this way
    # -------------------------------------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        if self.render_mode == "human":
            room_camera_config = CameraConfig(
                "render_camera",
                sapien_utils.look_at([4, -3.5, 3.5], [1.5, -3.5, 0]),
                1920,
                1080,
                1,
                0.01,
                10,
            )
            return room_camera_config
        # this camera follows the robot around (though might be in walls if the space is cramped)
        robot_camera_pose = sapien_utils.look_at([-0.2, 0.5, 1], [0.2, -0.2, 0])
        robot_camera_config = CameraConfig(
            "render_camera",
            robot_camera_pose,
            512,
            512,
            1.75,
            0.01,
            10,
            mount=self.agent.torso_lift_link,
        )
        return robot_camera_config

    def set_moving_goal_poses_for_render(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        for goal, pose in zip(self.subtask_goals, self.handle_world_poses):
            if pose is not None:
                goal.set_pose(pose)

    def render(self):
        self.set_moving_goal_poses_for_render()
        return super().render()

    # -------------------------------------------------------------------------------------------------
