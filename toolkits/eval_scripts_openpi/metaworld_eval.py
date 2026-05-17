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

import argparse
import collections
import json
import os
import pathlib
import time

import gymnasium as gym
import imageio
import metaworld
import numpy as np

from toolkits.eval_scripts_openpi import setup_logger, setup_policy

metaworld.register_mw_envs()
os.environ["MUJOCO_GL"] = "egl"


PROMPT_JSON_PATH = "rlinf/envs/metaworld/metaworld_config.json"
with open(PROMPT_JSON_PATH, "r") as f:
    config_data = json.load(f)
task_description_dict = config_data.get("TASK_DESCRIPTIONS", {})
difficulty_to_tasks = config_data.get("DIFFICULTY_TO_TASKS", {})
env_list = list(task_description_dict.keys())


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def format_metric(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_env(env_name):
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        camera_id=2,
        disable_env_checker=True,
    )
    env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    return env


def build_policy_input(observation, env_name, env):
    image = env.render()
    image = image[::-1, ::-1]
    state = observation[:4]
    batch = {
        "observation/image": image,
        "observation/state": state,
        "prompt": task_description_dict[env_name],
    }
    return batch, image


def save_rollout_video(frames, out_path, video_temp_subsample):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        out_path,
        [np.asarray(x) for x in frames[::video_temp_subsample]],
        fps=max(1, 25 // video_temp_subsample),
    )


def summarize_difficulty_rates(results_per_task, logger):
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

    if difficulty_rates:
        overall_avg = sum(difficulty_rates.values()) / len(difficulty_rates)
        logger.info(
            f"\nOverall Average Success Rate (across all difficulties): {overall_avg:.2%}"
        )


def run_one_episode(env, env_name, args, policy):
    frames = []
    observation, _ = env.reset()

    dummy_action = [0.0] * 4
    for _ in range(args.settle_steps):
        observation, _, _, _, _ = env.step(dummy_action)

    success = 0
    action_plan = collections.deque()
    episode_infer_ms_sum = 0.0
    episode_infer_count = 0
    episode_infer_ms_list = []
    first_success_time_s = None
    first_success_step = None
    episode_t0 = time.perf_counter()

    for step in range(args.max_steps):
        batch, image = build_policy_input(observation, env_name, env)

        if not action_plan:
            # 串行版的单次推理时间：一次 policy.infer 生成一整段 chunk。
            infer_t0 = time.perf_counter()
            action_chunk_result = policy.infer(batch)["actions"]
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0
            episode_infer_ms_sum += infer_ms
            episode_infer_count += 1
            episode_infer_ms_list.append(infer_ms)

            assert len(action_chunk_result) >= args.action_chunk, (
                f"We want to replan every {args.action_chunk} steps, but policy only predicts "
                f"{len(action_chunk_result)} steps."
            )
            action_plan.extend(action_chunk_result[: args.action_chunk])

        # The fixed-rate control cycle starts after any synchronous replanning
        # delay. This makes the baseline look like:
        #   execute chunk -> block on infer -> resume 1 / control_hz stepping.
        step_t0 = time.perf_counter()
        action = action_plan.popleft()
        observation, _, terminated, truncated, info = env.step(action)
        frames.append(image)

        if first_success_time_s is None and info.get("success", 0):
            first_success_time_s = time.perf_counter() - episode_t0
            first_success_step = step + 1

        # 为了和 RTC 版公平比较，这里也显式固定控制周期。
        elapsed = time.perf_counter() - step_t0
        sleep_s = max(0.0, (1.0 / args.control_hz) - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

        if info.get("success", 0):
            success = 1
            if args.break_on_success:
                break

        if terminated or truncated:
            break

    episode_elapsed_s = time.perf_counter() - episode_t0
    episode_metrics = {
        "success": int(success),
        "infer_ms_sum": episode_infer_ms_sum,
        "infer_count": episode_infer_count,
        "infer_ms_list": episode_infer_ms_list,
        "avg_infer_ms": safe_div(episode_infer_ms_sum, episode_infer_count),
        "episode_elapsed_s": episode_elapsed_s,
        "first_success_time_s": first_success_time_s,
        "first_success_step": first_success_step,
    }
    return success, frames, episode_metrics


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    policy = setup_policy(args)

    total_episodes = 0
    total_successes = 0
    total_infer_ms_sum = 0.0
    total_infer_count = 0
    total_episode_time_sum = 0.0
    total_success_time_sum = 0.0
    total_success_time_count = 0

    results_per_task = {}
    task_time_metrics = {}

    for env_name in env_list:
        env = make_env(env_name)

        task_successes = 0
        task_infer_ms_sum = 0.0
        task_infer_count = 0
        task_episode_time_sum = 0.0
        task_success_time_sum = 0.0
        task_success_time_count = 0

        for trial_id in range(args.num_trials_per_task):
            success, frames, episode_metrics = run_one_episode(
                env=env,
                env_name=env_name,
                args=args,
                policy=policy,
            )

            task_successes += success
            total_successes += success
            total_episodes += 1

            task_infer_ms_sum += episode_metrics["infer_ms_sum"]
            task_infer_count += episode_metrics["infer_count"]
            total_infer_ms_sum += episode_metrics["infer_ms_sum"]
            total_infer_count += episode_metrics["infer_count"]

            task_episode_time_sum += episode_metrics["episode_elapsed_s"]
            total_episode_time_sum += episode_metrics["episode_elapsed_s"]

            if episode_metrics["first_success_time_s"] is not None:
                task_success_time_sum += episode_metrics["first_success_time_s"]
                task_success_time_count += 1
                total_success_time_sum += episode_metrics["first_success_time_s"]
                total_success_time_count += 1

            if total_episodes <= args.num_save_videos:
                suffix = "success" if success else "failure"
                out_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                    / f"{env_name}_{trial_id}_{suffix}.mp4"
                )
                save_rollout_video(frames, out_path, args.video_temp_subsample)

            if args.save_episode_stats:
                stats_path = (
                    pathlib.Path(f"{args.log_dir}/{args.exp_name}/episode_stats")
                    / f"{env_name}_{trial_id}.json"
                )
                write_json(stats_path, episode_metrics)

        env.close()

        task_success_rate = task_successes / args.num_trials_per_task
        results_per_task[env_name] = task_success_rate

        task_summary = {
            "success_rate": task_success_rate,
            "avg_infer_ms_per_call": safe_div(task_infer_ms_sum, task_infer_count),
            "avg_episode_time_s_all": safe_div(
                task_episode_time_sum, args.num_trials_per_task
            ),
            "avg_success_time_s": safe_div(
                task_success_time_sum, task_success_time_count
            ),
            # 这个指标把失败 episode 的时间成本也算进去，
            # 更接近“为了得到一次成功，平均要花多少真实时间”。
            "effective_time_per_success_s": safe_div(
                task_episode_time_sum, task_successes
            ),
            "infer_count": task_infer_count,
            "num_trials": args.num_trials_per_task,
            "num_successes": task_successes,
        }
        task_time_metrics[env_name] = task_summary

        logger.info(
            f"Task: {env_name}, Successes: {task_successes}/{args.num_trials_per_task}, "
            f"Success Rate: {task_success_rate:.2%}, "
            f"avg_infer_ms/call={format_metric(task_summary['avg_infer_ms_per_call'], 1)}, "
            f"avg_episode_time_s(all)={format_metric(task_summary['avg_episode_time_s_all'])}, "
            f"avg_success_time_s={format_metric(task_summary['avg_success_time_s'])}, "
            f"effective_time_per_success_s={format_metric(task_summary['effective_time_per_success_s'])}"
        )

    total_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    overall_summary = {
        "success_rate": total_success_rate,
        "avg_infer_ms_per_call": safe_div(total_infer_ms_sum, total_infer_count),
        "avg_episode_time_s_all": safe_div(total_episode_time_sum, total_episodes),
        "avg_success_time_s": safe_div(
            total_success_time_sum, total_success_time_count
        ),
        "effective_time_per_success_s": safe_div(
            total_episode_time_sum, total_successes
        ),
        "infer_count": total_infer_count,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
    }

    logger.info(
        "[Overall] "
        f"success_rate={total_success_rate:.2%}, "
        f"avg_infer_ms/call={format_metric(overall_summary['avg_infer_ms_per_call'], 1)}, "
        f"avg_episode_time_s(all)={format_metric(overall_summary['avg_episode_time_s_all'])}, "
        f"avg_success_time_s={format_metric(overall_summary['avg_success_time_s'])}, "
        f"effective_time_per_success_s={format_metric(overall_summary['effective_time_per_success_s'])}"
    )

    summary_path = pathlib.Path(f"{args.log_dir}/{args.exp_name}/timing_summary.json")
    write_json(
        summary_path,
        {
            "method": "serial_sync",
            "per_task": task_time_metrics,
            "overall": overall_summary,
        },
    )


if __name__ == "__main__":
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
        help="Config name, options: 'pi0_metaworld' or 'pi05_metaworld'",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model weights file. Only PyTorch models are supported for now",
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
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Actions are replanned every N steps",
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
        default=0,
        help="Only saves rollout videos for the first N episodes. Default is 0 to avoid I/O overhead",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Save every Nth frame to the video",
    )
    parser.add_argument(
        "--control_hz",
        type=float,
        default=10.0,
        help="Control loop frequency in Hz. Use the same value as RTC for fair timing comparison",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=15,
        help="Initial dummy steps to let objects settle",
    )
    parser.add_argument(
        "--break_on_success",
        action="store_true",
        help="Break the episode immediately once success is achieved",
    )
    parser.add_argument(
        "--save_episode_stats",
        action="store_true",
        help="Save per-episode timing stats as json",
    )
    args = parser.parse_args()
    main(args)
