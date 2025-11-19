from typing import Any, Dict, List

import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link

from .planner import OpenSubtask, OpenSubtaskConfig, TaskPlan
from .subtask import SubtaskTrainEnv


@register_env("OpenSubtaskTrain-v0", max_episode_steps=200)
class OpenSubtaskTrainEnv(SubtaskTrainEnv):
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

    open_cfg = OpenSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_open_thresh_frac=dict(
            default=0.9,
            fridge=0.75,
            kitchen_counter=0.9,
        ),
        robot_cumulative_force_limit=10_000,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        randomly_slightly_open_articulation=False,
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], OpenSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {OpenSubtask.__name__} long"

        self.subtask_cfg = self.open_cfg
        self.randomly_slightly_open_articulation = randomly_slightly_open_articulation

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.open_subtask: OpenSubtask = self.task_plan[0]
        self.articulation: Articulation = self.subtask_articulations[0]
        self.link: Link = self.articulation.links[
            self.open_subtask.articulation_handle_link_idx
        ]
        self.qmax = self.articulation.qlimits[
            :, self.open_subtask.articulation_handle_active_joint_idx, 1
        ]
        self.qmin = self.articulation.qlimits[
            :, self.open_subtask.articulation_handle_active_joint_idx, 0
        ]
        open_thresh_frac = self.open_cfg.joint_qpos_open_thresh_frac[
            self.open_subtask.articulation_type
        ]
        self.target_qpos = (self.qmax - self.qmin) * open_thresh_frac + self.qmin

        self.ideal_qvel = 2 if self.open_subtask.articulation_type == "fridge" else 1

        if self.randomly_slightly_open_articulation:
            open_art = (
                torch.rand(
                    (len(env_idx),),
                    device=self.device,
                )
                < 0.1
            )
            new_qpos = self.articulation.qpos[env_idx].clone()
            new_qpos[
                open_art, self.open_subtask.articulation_handle_active_joint_idx
            ] = (self.target_qpos[open_art] - self.qmin[open_art]) * 0.2
            self.articulation.set_qpos(new_qpos)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            # -----------------------------------------------------------------------------------------
            # CHECKERS
            # -----------------------------------------------------------------------------------------
            # reaching checkers
            tcp_pos = self.agent.tcp_pose.p
            handle_pos = self.handle_world_poses[0].p
            tcp_to_handle_dist = torch.norm(tcp_pos - handle_pos, dim=1)

            # open checkers
            articulation_joint_qpos = self.articulation.qpos[
                :, self.open_subtask.articulation_handle_active_joint_idx
            ]
            frac_to_open_left = torch.clamp(
                (self.target_qpos - articulation_joint_qpos)
                / (self.target_qpos - self.qmin),
                min=0,
            )
            is_open = articulation_joint_qpos >= self.target_qpos

            # open speed checkers
            articulation_joint_qvel = self.articulation.qvel[
                :, self.open_subtask.articulation_handle_active_joint_idx
            ]

            # arm back to normal checkers
            arm_torso_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )

            # ee rest checkers
            ee_to_rest_dist = torch.norm(tcp_pos - self.ee_rest_world_pose.p, dim=1)

            # static checkers
            robot_qvel = self.agent.robot.qvel[..., :-2]

            # -----------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------
            # REWARD
            # -----------------------------------------------------------------------------------------
            # reaching reward to grasp link
            reaching_reward = 3 * (1 - torch.tanh(tcp_to_handle_dist))
            # if grasped, extra reward
            reaching_reward[info["is_grasped"]] = 5
            # if open, most reward
            reaching_reward[is_open] = 6

            # reward for opening cabinet
            open_reward = 6 * (1 - frac_to_open_left)
            open_reward[articulation_joint_qpos > (self.qmin + 0.01)] += 3

            # reward for qvel <= positive value (not hard condition for success)
            slow_open_reward = 3 * (articulation_joint_qvel <= self.ideal_qvel).float()

            # when articulation open, return to rest
            arm_torso_resting_orientation_reward = 2 * (
                1 - torch.tanh(arm_torso_to_resting_diff)
            )
            arm_torso_resting_orientation_reward[~is_open] = 0

            torso_resting_orientation_reward = torch.abs(
                (self.agent.robot.qpos[..., 3] - self.agent.robot.qlimits[..., 3, 0])
                / (
                    self.agent.robot.qlimits[..., 3, 1]
                    - self.agent.robot.qlimits[..., 3, 0]
                )
            )
            torso_resting_orientation_reward[~is_open] = 0

            ee_rest_reward = 2 * (1 - torch.tanh(ee_to_rest_dist))
            ee_rest_reward[~is_open] = 0

            # when articulation open and ee rest, static reward
            static_reward = 1 - torch.tanh(torch.norm(robot_qvel, dim=1))
            static_reward[~is_open | ~info["ee_rest"] | ~info["robot_rest"]] = 0

            # give reward for entering success
            success_reward = 3 * info["success"]

            reward = (
                reaching_reward
                + open_reward
                + slow_open_reward
                + arm_torso_resting_orientation_reward
                + torso_resting_orientation_reward
                + ee_rest_reward
                + static_reward
                + success_reward
            )

            return reward

            # -----------------------------------------------------------------------------------------

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 27
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
