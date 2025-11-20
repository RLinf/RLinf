from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

import sapien.physx as physx

from mani_skill.utils.structs import Pose

from .planner import Subtask, TaskPlan
from .sequential_task import SequentialTaskEnv


class SubtaskTrainEnv(SequentialTaskEnv):
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

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        # additional spawn randomization, shouldn't need to change
        spawn_data_fp=None,
        # colliison tracking
        robot_force_mult=0.001,
        robot_force_penalty_min=0.2,
        # additional randomization
        target_randomization=False,
        **kwargs,
    ):
        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], Subtask
        ), f"Task plans for {self.__class__.__name__} must be one {Subtask.__name__} long"

        # spawn vals
        self.spawn_data_fp = Path(spawn_data_fp)
        assert self.spawn_data_fp.exists(), f"could not find {self.spawn_data_fp}"

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min

        # additional target obj randomization
        self.target_randomization = target_randomization

        self.subtask_cfg = getattr(self, "subtask_cfg", None)
        assert (
            self.subtask_cfg is not None
        ), "Need to designate self.subtask_cfg (in extending env)"

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # RECONFIGURE AND INIT
    # -------------------------------------------------------------------------------------------------


    def _after_reconfigure(self, options):
        self.spawn_data = torch.load(self.spawn_data_fp, map_location=self.device)
        self.spawn_selection_idxs = [None] * self.num_envs
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx, options: Dict):
        super()._initialize_episode(env_idx, options)
        self._reset_stats(env_idx)
        self._apply_premade_spawns(env_idx, options)

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
            }
        else:
            # only update info for the environments that are being reset.
            self.consecutive_grasp[env_idx] = 0
            self.episode_stats["is_src_obj_grasped"][env_idx] = False
            self.episode_stats["consecutive_grasp"][env_idx] = False
        self.extra_stats = {}

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
                if len(v) != env_idx.numel():
                    continue
                if k == "articulation_qpos":
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
            if "obj_raw_pose" in batched_spawn_data:
                self.subtask_objs[0].set_pose(
                    Pose.create(batched_spawn_data["obj_raw_pose"])
                )
            if "obj_raw_pose_wrt_tcp" in batched_spawn_data:
                if self.gpu_sim_enabled:
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.subtask_objs[0].set_pose(
                    Pose.create(
                        self.agent.tcp.pose.raw_pose[env_idx]
                    )  # NOTE (arth): use tcp.pose for spawning for slightly better accuracy
                    * Pose.create(batched_spawn_data["obj_raw_pose_wrt_tcp"])
                )
            if "articulation_qpos" in batched_spawn_data:
                self.subtask_articulations[0].set_qpos(
                    batched_spawn_data["articulation_qpos"]
                )
                self.subtask_articulations[0].set_qvel(
                    self.subtask_articulations[0].qvel[env_idx] * 0
                )
                if self.gpu_sim_enabled and len(env_idx) == self.num_envs:
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene.px.step()
                    self.scene._gpu_fetch_all()

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        with torch.device(self.device):
            infos = super().evaluate()

            # set to zero in case we use continuous task
            #   this way, if the termination signal is ignored, env will
            #   still reevaluate success each step
            self.subtask_pointer = torch.zeros_like(self.subtask_pointer)
            return infos

    # -------------------------------------------------------------------------------------------------
