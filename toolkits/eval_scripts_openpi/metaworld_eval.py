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
import json
import os
import pathlib

import gymnasium as gym
import imageio
import metaworld
import numpy as np

from toolkits.eval_scripts_openpi import setup_logger, setup_policy

metaworld.register_mw_envs()
os.environ["MUJOCO_GL"] = "egl"


def load_prompt_from_json(json_path, env_name):
    with open(json_path, "r") as f:
        prompt_data = json.load(f)
    return prompt_data.get(env_name, "")


PROMPT_JSON_PATH = "rlinf/envs/metaworld/metaworld_config.json"
with open(PROMPT_JSON_PATH, "r") as f:
    config_data = json.load(f)
task_description_dict = config_data.get("TASK_DESCRIPTIONS", {})
difficulty_to_tasks = config_data.get("DIFFICULTY_TO_TASKS", {})
env_list = list(task_description_dict.keys())


def main(args):
    # Setup logging
    logger = setup_logger(args.exp_name, args.log_dir)

    # policy setup
    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")

    total_episodes = 0
    total_successes = 0
    results_per_task = {}

    for env_name in env_list:
        logger.info(f"Start evaluating: {env_name}")
        logger.info(f"任务描述 (Prompt): {task_description_dict[env_name]}")
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode="rgb_array",
            camera_id=2,
            disable_env_checker=True,
        )
        # Set camera position if necessary
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        task_successes = 0
        for trial_id in range(args.num_trials_per_task):
            frames = []
            observation, info = env.reset()
            dummy_action = [0.0] * 4
            for _ in range(15):  # wait for objects to settle
                observation, _, _, _, _ = env.step(dummy_action)

            success = 0
            action_plan = collections.deque()

            for step in range(args.max_steps):
                image = env.render()
                image = image[::-1, ::-1]
                state = observation[:4]
                batch = {
                    "observation/image": image,
                    "observation/state": state,
                    "prompt": task_description_dict[env_name],
                }
                # Plan actions only if empty
                if not action_plan:
                    action_chunk_result = policy.infer(batch)["actions"]
                    assert len(action_chunk_result) >= args.action_chunk, (
                        f"We want to replan every {args.action_chunk} steps, but policy only predicts {len(action_chunk_result)} steps."
                    )
                    action_plan.extend(action_chunk_result[: args.action_chunk])
                action = action_plan.popleft()
                observation, reward, terminated, truncated, info = env.step(action)
                frames.append(image)
                if info.get("success", 0) or terminated or truncated:
                    success = int(info.get("success", 0))
                    # break  # end on success or termination

            # If episode succeeded, accumulate
            task_successes += success
            total_successes += success
            total_episodes += 1

            # Save video only for first N episodes
            if total_episodes <= args.num_save_videos:
                suffix = "success" if success else "failure"
                out_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                    / f"{env_name}_{trial_id}_{suffix}.mp4"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    out_path,
                    [np.asarray(x) for x in frames[:: args.video_temp_subsample]],
                    fps=25 // args.video_temp_subsample,
                )

        env.close()
        task_success_rate = task_successes / args.num_trials_per_task
        results_per_task[env_name] = task_success_rate
        logger.info(
            f"Task: {env_name}, Successes: {task_successes}/{args.num_trials_per_task}, Success Rate: {task_success_rate:.2%}"
        )

    total_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("\n===============")
    logger.info("Per-Task Success Rate:")
    for env_name, sr in results_per_task.items():
        logger.info(f"{env_name}: {sr:.2%}")

    # Calculate success rate by difficulty
    logger.info("\n===============")
    logger.info("Success Rate by Difficulty:")
    difficulty_rates = {}
    for difficulty, tasks in difficulty_to_tasks.items():
        task_rates = [
            results_per_task.get(task, 0.0)
            for task in tasks
            if task in results_per_task
        ]
        if task_rates:
            avg_rate = sum(task_rates) / len(task_rates)
            difficulty_rates[difficulty] = avg_rate
            logger.info(
                f"{difficulty}: {avg_rate:.2%} (averaged over {len(task_rates)} tasks)"
            )

    # Calculate overall average across all difficulties
    if difficulty_rates:
        overall_avg = sum(difficulty_rates.values()) / len(difficulty_rates)
        logger.info(
            f"\nOverall Average Success Rate (across all difficulties): {overall_avg:.2%}"
        )

    logger.info(
        f"\nTotal Success Rate: {total_successes}/{total_episodes} = {total_success_rate:.2%}"
    )


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
        default="metaworld_32",
        help="Experiment name used for naming log files and video save directories",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi0_metaworld",
        help="Config name, options: 'pi0_metaworld' or 'pi05_metaworld', used to select the corresponding model configuration",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model weights file. If None, uses the default pretrained model. Only PyTorch models are supported for now",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=10,
        help="Number of trials per task",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=160,
        help="Maximum number of steps per episode. If a task is not completed within this number of steps, it is considered failed",
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
