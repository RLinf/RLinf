# Copyright 2026 The RLinf Authors.
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
import math
import os
import pathlib

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from toolkits.eval_scripts_dexbotic import setup_logger, setup_policy

os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


# main function
def main(args):
    # Setup logging
    logger = setup_logger(args.exp_name, args.log_dir)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"Task suite: {args.task_suite_name}")

    # Determine max steps based on task suite
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # policy setup
    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    results_per_task = {}

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            logger.info(f"\nTask: {task_description}")
            logger.info(f"Starting episode {task_episodes + 1}...")

            # Reset environment
            policy.reset()
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            replay_images = []

            for t in range(max_steps + args.num_steps_wait):
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                if not action_plan:
                    observation = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    # infer the action
                    action_chunk = policy.infer(observation)["actions"]

                    assert len(action_chunk) >= args.action_chunk, (
                        f"We want to replan every {args.action_chunk} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.action_chunk])

                action = action_plan.popleft()
                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode (only for first N episodes across all tasks)
            if total_episodes <= args.num_save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                out_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                    / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    out_path,
                    [
                        np.asarray(x)
                        for x in replay_images[:: args.video_temp_subsample]
                    ],
                    fps=30 // args.video_temp_subsample,
                )

            # Log current results
            logger.info(f"Success: {done}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        env.close()

        # Log final results for this task
        task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
        results_per_task[task_description] = task_success_rate
        logger.info(
            f"Task: {task_description}, Successes: {task_successes}/{task_episodes}, Success Rate: {task_success_rate:.2%}"
        )

    # Log final performance
    total_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("\n===============")
    logger.info("Per-Task Success Rate:")
    for task_description, sr in results_per_task.items():
        logger.info(f"{task_description}: {sr:.2%}")

    logger.info(
        f"\nTotal Success Rate: {total_successes}/{total_episodes} = {total_success_rate:.2%}"
    )
    logger.info(f"results/total_success_rate: {total_success_rate}")
    logger.info(f"results/total_episodes: {total_episodes}")
    logger.info(f"results/total_successes: {total_successes}")


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
        default="libero_spatial_dexbotic",
        help="Experiment name used for naming log files and video save directories",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the trained Dexbotic checkpoint directory",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        help="Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=50,
        help="Number of rollouts per task",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=50,
        help="Action chunk size: the length of action sequence predicted by the policy each time. Actions are replanned every N steps. Dexbotic uses 50 by default.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of diffusion denoising steps for Dexbotic action generation (default: 10).",
    )
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Number of steps to wait for objects to stabilize in sim",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (for reproducibility)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (cuda or cpu)",
    )
    args = parser.parse_args()
    main(args)
