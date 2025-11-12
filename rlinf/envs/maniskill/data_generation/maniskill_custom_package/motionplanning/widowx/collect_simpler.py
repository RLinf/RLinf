# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# MIT License

# Copyright (c) 2024 simpler-env

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import multiprocessing as mp
import os
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill import DEMO_DIR  # MANISKILL_ROOT_DIR
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm

from rlinf.envs.maniskill.data_generation.maniskill_custom_package.motionplanning.widowx.motionplanner import (
    VLADataCollectWidowXArmMotionPlanningSolver,
)
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.motionplanning.widowx.solutions.widowx_simpler_mp import (
    SolvePutCarrot,
    SolvePutEggplant,
    SolvePutSpoon,
    SolveStackCube,
)

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
SIMPLER_MP_SOLUTIONS = {
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": SolveStackCube,
    "PutSpoonOnTableClothInScene-v1": SolvePutSpoon,
    "PutEggplantInBasketScene-v1": SolvePutEggplant,
    "PutCarrotOnPlateInScene-v1": SolvePutCarrot,
}


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = (
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
    )
    """The environment ID of the task you want to simulate
        f"Environment to run motion planning solver on. Available options are {list(SIMPLER_MP_SOLUTIONS.keys())}"
    """

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb+segmentation"
    """Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script."""

    num_traj: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 5
    """Number of trajectories to generate"""

    control_mode: str = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    """just can be arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "cpu"
    """Simulation backend: 'auto', 'cpu', or 'gpu'"""

    render_mode: str = "rgb_array"
    """Render mode: 'sensors' or 'rgb_array'"""

    vis: bool = False
    """Whether to open a GUI for live visualization, whether or not to open a GUI to visualize the solution live"""

    save_video: bool = False
    """Whether to save videos locally"""

    save_data: bool = False
    """Whether to save data locally"""

    shader: str = "default"
    """Shader used for rendering: 'default', 'rt', or 'rt-fast'"""
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: str = os.path.join(DEMO_DIR, "mp_collect")
    """Directory to save recorded trajectories"""

    num_procs: int = 1
    """Number of processes for parallel trajectory replay (only for CPU backend)"""
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment."""

    plan_time_step: Optional[float] = None
    seed: int = 0

    debug: bool = False


def _main(args, proc_id: int = 0, single_num: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        num_envs=1,
        control_mode=args.control_mode,  # "pd_joint_pos", "pd_joint_pos_vel", "pd_ee_delta_pose" "pd_ee_target_delta_pose"
        render_mode=args.render_mode,
        reward_mode="none" if args.reward_mode is None else args.reward_mode,
        sensor_configs={"shader_pack": args.shader},
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={"shader_pack": args.shader},
        sim_backend=args.sim_backend,
        sim_config={
            "sim_freq": 500,
            "control_freq": 5,
        },
    )

    env = RecordEpisode(
        env,
        output_dir=Path(args.record_dir) / env_id / f"{args.num_traj}" / "videos",
        save_trajectory=False,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=24,
        save_on_reset=False,
        # recording_camera_name="3rd_view_camera",
        avoid_overwriting_video=True,
        max_steps_per_video=1000,
    )
    if env_id not in SIMPLER_MP_SOLUTIONS:
        print(f"Environment {env_id} not supported, use `SolvePutCarrot` as default.")
        solve = SolvePutCarrot
    else:
        solve = SIMPLER_MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")

    if single_num == 0:
        single_num = args.num_traj
    pbar = tqdm(range(single_num), desc=f"proc_id: {proc_id}")
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    failure = 0

    np.random.seed(proc_id + 1 + args.seed)
    torch.manual_seed(proc_id + 1 + args.seed)
    idx = 0

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("solve function timed out!")

    def solve_with_timeout(
        env, seed, debug, vis, use_rrt, plan_time_step, timeout=30
    ) -> Tuple[int, Optional[VLADataCollectWidowXArmMotionPlanningSolver]]:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = solve(env, seed, debug, vis, use_rrt, plan_time_step)
            signal.alarm(0)
            return result
        except TimeoutException:
            print("solve function timed out!")
            return -1, None

    while True:
        try:
            start_solve_t = time.time()
            print("start motionplanning!")

            # normal
            env_reset_options = {"obj_set": "train"}

            # debug
            if args.debug:
                pq = len(env.unwrapped.xyz_configs) * len(env.unwrapped.quat_configs)
                episode_id = (
                    torch.randint(10000000, (env.num_envs,), device=env.device) % pq
                    + idx * pq
                )
                env_reset_options["episode_id"] = episode_id

            obs, info = env.reset(options=env_reset_options)
            res, planner = solve_with_timeout(
                env,
                seed=idx,
                debug=False,
                vis=args.vis,
                use_rrt=False,
                plan_time_step=args.plan_time_step,
                timeout=20,
            )
            print("motionplanning using time(s):", time.time() - start_solve_t)
        except Exception as e:
            print(
                f"Cannot find valid solution because of an error in motion planning solution: {e}"
            )
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
            print("<<<<<<<Failure, Motion planning can not get a solution.>>>>>>>")
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
            if success:
                print(" ******* Success! And Motion planning get a solution. ******* ")
            else:
                print(" ******* Failure! But Motion planning get a solution. ******* ")
        successes.append(success)

        if success:
            saving_path_name = f"success_proc_{proc_id}_numid_{passed}_epsid_{idx}"
            if args.save_video:
                video_path = saving_path_name + ".mp4"
                env.flush_video(name=video_path)
            if args.save_data:
                exp_dir = Path(args.record_dir) / env_id / f"{args.num_traj}" / "data"
                exp_dir.mkdir(parents=True, exist_ok=True)
                saving_path = exp_dir / (saving_path_name)
                planner.data_collector.save_data(saving_path, is_compressed=True)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "succ_rate": np.mean(successes),
                    "fail_rate": failed_motion_plans / (idx + 1),
                    "avg_eplen": np.mean(solution_episode_lengths),
                    "max_eplen": np.max(solution_episode_lengths, initial=0),
                    "min_eplen": np.min(solution_episode_lengths, initial=0),
                }
            )
            passed += 1
            if passed == single_num:
                break
        else:
            if args.save_video:
                saving_path_name = f"failure_proc_{proc_id}_numid_{failure}_epsid_{idx}"
                env.flush_video(name=saving_path_name + ".mp4", save=True)
            failure += 1
        idx += 1
    env.close()
    return


def main(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    if (
        args.sim_backend != "gpu"
        and args.num_procs > 1
        and args.num_procs <= args.num_traj
    ):
        if args.num_traj < args.num_procs:
            raise ValueError(
                "Number of trajectories should be greater than or equal to number of processes"
            )
        total_num_traj = args.num_traj
        single_num_traj = total_num_traj // args.num_procs
        proc_args = [
            (deepcopy(args), i, single_num_traj) for i in range(args.num_procs)
        ]
        pool = mp.Pool(args.num_procs)
        pool.starmap(_main, proc_args)
        pool.close()
        pool.join()
    else:
        _main(args)


if __name__ == "__main__":
    start = time.time()
    mp.set_start_method("spawn")
    parsed_args = tyro.cli(Args)
    main(parsed_args)
    print(f"Total time taken: {time.time() - start}")
    if parsed_args.save_data:
        exp_dir = (
            Path(parsed_args.record_dir)
            / parsed_args.env_id
            / f"{parsed_args.num_traj}"
            / "data"
        )
        print(f"Data saved to {exp_dir} ")
