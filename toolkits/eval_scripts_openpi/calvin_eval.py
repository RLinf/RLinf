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

import collections
import pathlib

import imageio
import numpy as np
import tqdm
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env

from rlinf.envs.calvin import ENV_CFG_DIR, _get_calvin_tasks_and_reward
from toolkits.eval_scripts_openpi import setup_logger, setup_policy


# print performance
def _calvin_print_performance(logger, episode_solved_subtasks, per_subtask_success):
    # Compute avg success rate per task length
    logger.info("#####################################################")
    logger.info(f"Avg solved subtasks: {np.mean(episode_solved_subtasks)}\n")

    logger.info("Per sequence_length avg success:")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logger.info(
            f"{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks) * 100}%"
        )

    logger.info("\n Per subtask avg success:")
    for key in per_subtask_success:
        logger.info(f"{key}: \t\t\t {np.mean(per_subtask_success[key]) * 100}%")
    logger.info("#####################################################")


# main function
def main(args):
    # Setup logging
    logger = setup_logger(args.exp_name, args.log_dir)
    # env setup
    env = get_env(ENV_CFG_DIR, show_gui=False)
    task_definitions, task_instructions, task_reward = _get_calvin_tasks_and_reward(
        args.num_trials
    )
    # policy setup
    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")
    # Start evaluation
    episode_solved_subtasks = []
    per_subtask_success = collections.defaultdict(list)
    for i, (initial_state, task_sequence) in enumerate(tqdm.tqdm(task_definitions)):
        logger.info(f"Starting episode {i + 1}...")
        logger.info(f"Task sequence: {task_sequence}")
        # Reset env to initial position for task
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        # Rollout
        rollout_images = []
        solved_subtasks = 0
        for subtask in task_sequence:
            start_info = env.get_info()
            action_plan = collections.deque()
            obs = env.get_obs()
            done = False
            for _ in range(args.max_steps):
                img = obs["rgb_obs"]["rgb_static"]
                wrist_img = obs["rgb_obs"]["rgb_gripper"]
                rollout_images.append(img)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    state_ee_pos = obs["robot_obs"][:3]
                    state_ee_rot = obs["robot_obs"][3:6]
                    state_gripper = obs["robot_obs"][6:7]

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": obs["robot_obs"][:7],
                        "observation/state_ee_pos": state_ee_pos,
                        "observation/state_ee_rot": state_ee_rot,
                        "observation/state_gripper": state_gripper,
                        "prompt": str(task_instructions[subtask][0]),
                    }
                    action_chunk_result = policy.infer(element)["actions"]
                    assert len(action_chunk_result) >= args.action_chunk, (
                        f"We want to replan every {args.action_chunk} steps, but policy only predicts {len(action_chunk_result)} steps."
                    )
                    action_plan.extend(action_chunk_result[: args.action_chunk])

                action = action_plan.popleft().copy()
                # Round gripper action since env expects gripper_action in (-1, 1)
                action[-1] = 1 if action[-1] > 0 else -1

                # Step environment
                obs, _, _, current_info = env.step(action)

                # check if current step solves a task
                current_task_info = task_reward.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )
                if len(current_task_info) > 0:
                    done = True
                    solved_subtasks += 1
                    break

            per_subtask_success[subtask].append(int(done))
            if not done:
                # Subtask execution failed --> stop episode
                break

        episode_solved_subtasks.append(solved_subtasks)
        if len(episode_solved_subtasks) <= args.num_save_videos:
            # Save rollout video
            idx = len(episode_solved_subtasks)
            # Determine success: all subtasks completed
            is_success = solved_subtasks == len(task_sequence)
            suffix = "success" if is_success else "failure"
            out_path = (
                pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                / f"rollout_{idx}_{suffix}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                out_path,
                [np.asarray(x) for x in rollout_images[:: args.video_temp_subsample]],
                fps=50 // args.video_temp_subsample,
            )

        # Print current performance after each episode
        logger.info(f"Solved subtasks: {solved_subtasks}")
        _calvin_print_performance(logger, episode_solved_subtasks, per_subtask_success)

    env.close()

    # Log final performance
    logger.info(f"results/avg_num_subtasks: {np.mean(episode_solved_subtasks)}")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logger.info(
            f"results/avg_success_len_{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks)}"
        )
    for key in per_subtask_success:
        logger.info(f"results/avg_success__{key}: {np.mean(per_subtask_success[key])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="calvin_pi0",
        help="Experiment name used for naming log files and video save directories",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi0_calvin",
        help="Config name, options: 'pi0_calvin' or 'pi05_calvin', used to select the corresponding model configuration",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model weights file. If None, uses the default pretrained model",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="Total number of evaluation trials, i.e., the number of episodes to evaluate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=480,
        help="Maximum number of steps per subtask. If a subtask is not completed within this number of steps, it is considered failed",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Action chunk size: the length of action sequence predicted by the policy each time. Actions are replanned every N steps",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps to sample from the policy",
    )
    parser.add_argument(
        "--num_save_videos",
        type=int,
        default=10,
        help="Number of videos to save. Only saves rollout videos for the first N episodes to disk",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Video temporal subsampling rate. Saves every Nth frame to the video to reduce file size",
    )
    args = parser.parse_args()
    main(args)
