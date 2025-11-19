from typing import Any, Dict, List

import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link

from .planner import CloseSubtask, CloseSubtaskConfig, TaskPlan
from .subtask import SubtaskTrainEnv


@register_env("CloseSubtaskTrain-v0", max_episode_steps=200)
class CloseSubtaskTrainEnv(SubtaskTrainEnv):
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

    close_cfg = CloseSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_close_thresh_frac=0.01,
        robot_cumulative_force_limit=10_000,
    )
    closed_with_ee_near_handle_thresh: float = 0.3

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], CloseSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {CloseSubtask.__name__} long"

        self.subtask_cfg = self.close_cfg

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def _load_scene(self, options):
        self.closed_with_ee_near_handle = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return super()._load_scene(options)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.close_subtask: CloseSubtask = self.task_plan[0]
        self.articulation: Articulation = self.subtask_articulations[0]
        self.link: Link = self.articulation.links[self.close_subtask.articulation_handle_link_idx]
        self.qmax = self.articulation.qlimits[:, self.close_subtask.articulation_handle_active_joint_idx, 1]
        self.qmin = self.articulation.qlimits[:, self.close_subtask.articulation_handle_active_joint_idx, 0]
        self.target_qpos = (self.qmax - self.qmin) * self.close_cfg.joint_qpos_close_thresh_frac + self.qmin

        self.ideal_qvel = 2 if self.close_subtask.articulation_type == "fridge" else 1

        self.closed_with_ee_near_handle[env_idx] = False

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            # -----------------------------------------------------------------------------------------
            # CHECKERS
            # -----------------------------------------------------------------------------------------
            # reaching checkers
            tcp_pos = self.agent.tcp_pose.p
            handle_pos = self.handle_world_poses[0].p
            tcp_to_handle_dist = torch.norm(tcp_pos - handle_pos, dim=1)

            # close checkers
            articulation_joint_qpos = self.articulation.qpos[:, self.close_subtask.articulation_handle_active_joint_idx]
            frac_to_close_left = torch.clamp(
                (articulation_joint_qpos - self.target_qpos) / (self.qmax - self.target_qpos),
                min=0,
            )
            is_closed = articulation_joint_qpos <= self.target_qpos

            # close "correctly" checkers
            self.closed_with_ee_near_handle = is_closed & (
                self.closed_with_ee_near_handle | (tcp_to_handle_dist <= self.closed_with_ee_near_handle_thresh)
            )

            # close speed checkers
            articulation_joint_qvel = self.articulation.qvel[:, self.close_subtask.articulation_handle_active_joint_idx]

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
            reaching_reward = 7 * (1 - torch.tanh(tcp_to_handle_dist))
            # if grasped, extra reward
            reaching_reward[info["is_grasped"]] = 7.5
            # if closed, most reward
            reaching_reward[self.closed_with_ee_near_handle] = 8

            # reward for closing cabinet
            close_reward = 5 * (1 - frac_to_close_left)

            # reward for staying within reasonable qvel
            slow_close_reward = 5 * (torch.abs(articulation_joint_qvel) <= self.ideal_qvel).float()

            # when articulation close, return to rest
            arm_torso_resting_orientation_reward = 1 - ((torch.tanh(arm_torso_to_resting_diff) + torch.tanh(arm_torso_to_resting_diff / 5)) / 2)
            arm_torso_resting_orientation_reward[self.closed_with_ee_near_handle] *= 10

            torso_resting_orientation_reward = torch.abs(
                (self.agent.robot.qpos[..., 3] - self.agent.robot.qlimits[..., 3, 0])
                / (self.agent.robot.qlimits[..., 3, 1] - self.agent.robot.qlimits[..., 3, 0])
            )
            torso_resting_orientation_reward[~self.closed_with_ee_near_handle] = 0

            ee_rest_reward = 2 - (torch.tanh(ee_to_rest_dist) + torch.tanh(3 * ee_to_rest_dist))
            ee_rest_reward[~self.closed_with_ee_near_handle] = 0

            # when articulation close and ee rest, static reward
            static_reward = 2 * (1 - torch.tanh(torch.norm(robot_qvel, dim=1)))
            static_reward[~self.closed_with_ee_near_handle | ~info["ee_rest"] | ~info["robot_rest"]] = 0

            # give reward for entering success
            success_reward = 3 * (info["success"] & self.closed_with_ee_near_handle)

            reward = (
                reaching_reward
                + close_reward
                + slow_close_reward
                + arm_torso_resting_orientation_reward
                + torso_resting_orientation_reward
                + ee_rest_reward
                + static_reward
                + success_reward
            )

            return reward

            # -----------------------------------------------------------------------------------------

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 37
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
