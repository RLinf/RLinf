from typing import Any, Dict, List

import torch

from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .planner import PickSubtask, PickSubtaskConfig, TaskPlan
from .subtask import SubtaskTrainEnv


@register_env("PickSubtaskTrain-v0", max_episode_steps=200)
class PickSubtaskTrainEnv(SubtaskTrainEnv):
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

    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        robot_cumulative_force_limit=5000,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        **kwargs,
    ):
        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], PickSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {PickSubtask.__name__} long"

        self.subtask_cfg = self.pick_cfg

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            super()._initialize_episode(env_idx, options)
            if self.target_randomization:
                b = len(env_idx)

                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz += self.subtask_objs[0].pose.p
                xyz[..., 2] += 0.005

                qs = quaternion_raw_multiply(
                    randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False
                    ),
                    self.subtask_objs[0].pose.q,
                )
                self.subtask_objs[0].set_pose(Pose.create_from_pq(xyz, qs))

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (arth): reward "steps" are as follows:
            #       - reaching_reward
            #       - if not grasped
            #           - not_grasped_reward
            #       - is_grasped_reward
            #       - if grasped
            #           - grasped_rewards
            #       - if grasped and ee_at_rest
            #           - static_reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            not_grasped = ~info["is_grasped"]
            not_grasped_reward = torch.zeros_like(reward[not_grasped])

            is_grasped = info["is_grasped"]
            is_grasped_reward = torch.zeros_like(reward[is_grasped])

            robot_ee_rest_and_grasped = (
                is_grasped & info["ee_rest"] & info["robot_rest"]
            )
            robot_ee_rest_and_grasped_reward = torch.zeros_like(
                reward[robot_ee_rest_and_grasped]
            )

            # ---------------------------------------------------

            # reaching reward
            tcp_to_obj_dist = torch.norm(obj_pos - tcp_pos, dim=1)
            reaching_rew = 5 * (1 - torch.tanh(3 * tcp_to_obj_dist))
            reward += reaching_rew

            # penalty for ee moving too much when not grasping
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew

            # pick reward
            grasp_rew = 2 * info["is_grasped"]
            reward += grasp_rew

            # success reward
            success_rew = 3 * info["success"]
            reward += success_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 1 - torch.tanh(arm_to_resting_diff / 5)
            reward += arm_resting_orientation_rew

            # ---------------------------------------------------------------
            # colliisions
            step_no_col_rew = 3 * (
                1
                - torch.tanh(
                    3
                    * (
                        torch.clamp(
                            self.robot_force_mult * info["robot_force"],
                            min=self.robot_force_penalty_min,
                        )
                        - self.robot_force_penalty_min
                    )
                )
            )
            reward += step_no_col_rew

            # cumulative collision penalty
            cum_col_under_thresh_rew = (
                2
                * (
                    info["robot_cumulative_force"]
                    < self.pick_cfg.robot_cumulative_force_limit
                ).float()
            )
            reward += cum_col_under_thresh_rew
            # ---------------------------------------------------------------

            if torch.any(not_grasped):
                # penalty for torso moving up and down too much
                tqvel_z = self.agent.robot.qvel[..., 3][not_grasped]
                torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
                torso_not_moving_rew[tcp_to_obj_dist[not_grasped] < 0.3] = 1
                not_grasped_reward += torso_not_moving_rew

                # penalty for ee not over obj
                ee_over_obj_rew = 1 - torch.tanh(
                    5
                    * torch.norm(
                        obj_pos[..., :2][not_grasped] - tcp_pos[..., :2][not_grasped],
                        dim=1,
                    )
                )
                not_grasped_reward += ee_over_obj_rew

            if torch.any(is_grasped):
                # not_grasped reward has max of +2
                # so, we add +2 to grasped reward so reward only increases as task proceeds
                is_grasped_reward += 2

                # place reward
                ee_to_rest_dist = torch.norm(
                    tcp_pos[is_grasped] - rest_pos[is_grasped], dim=1
                )
                place_rew = 5 * (1 - torch.tanh(3 * ee_to_rest_dist))
                is_grasped_reward += place_rew

                # arm_to_resting_diff_again
                is_grasped_reward += arm_resting_orientation_rew[is_grasped]

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][is_grasped]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                is_grasped_reward += base_still_rew

                if torch.any(robot_ee_rest_and_grasped):
                    # increment to encourage robot and ee staying in rest
                    robot_ee_rest_and_grasped_reward += 2

                    qvel = self.agent.robot.qvel[..., :-2][robot_ee_rest_and_grasped]
                    static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                    robot_ee_rest_and_grasped_reward += static_rew

            # add rewards to specific envs
            reward[not_grasped] += not_grasped_reward
            reward[is_grasped] += is_grasped_reward
            reward[robot_ee_rest_and_grasped] += robot_ee_rest_and_grasped_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 28.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


    def evaluate(self):
        """
        rename the grasped and add a state consecutive_grasp for holding pick for more than 5time steps
        """
        with torch.device(self.device):
            infos = super().evaluate()
            # is_grasped
            is_src_obj_grasped = infos["is_grasped"]
            infos["is_src_obj_grasped"] = is_src_obj_grasped

            # if is_src_obj_grasped:
            self.consecutive_grasp += is_src_obj_grasped
            self.consecutive_grasp[is_src_obj_grasped == 0] = 0
            consecutive_grasp = self.consecutive_grasp >= 5
            infos["consecutive_grasp"] = consecutive_grasp
            return infos
    # -------------------------------------------------------------------------------------------------
