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

from collections import defaultdict
import json
import signal
import time
import numpy as np

import torch
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c

import gymnasium as gym
import numpy as np
from PIL import Image
from mani_skill.envs.sapien_env import BaseEnv
import tyro
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments. Example command to run:
    """

    env_id: str = "PutCarrotOnPlateInScene-v1"
    """The environment ID to run"""

    num_envs: int = 1
    """Number of environments to run in parallel"""
    
    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    max_episode_len: int = 80
    """Max episode length"""
    
    record_dir: str = "/nvme_data/tonghe/RL4VLA/datasets/octo_collect"
    """The directory to save videos and results"""

    seed: int = 0
    """Seed the model and environment. Default seed is 0"""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    save_data: bool = True
    """Whether to save collect data"""

    save_successful_videos_only: bool = True
    """Whether to save videos only for successful episodes"""


def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)

    env: BaseEnv = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend="gpu",
        sim_config={
            "sim_freq": 500,
            "control_freq": 5,
        },
        max_episode_steps=args.max_episode_len,
        sensor_configs={"shader_pack": "default"},
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'
    
    from simpler_env.policies.octo.octo_model import OctoInference
    model = OctoInference(model_type="octo-small", policy_setup="widowx_bridge", init_rng=args.seed, action_scale=1)

    eval_metrics = defaultdict(list)
    eps_count = 0
    valid_count = 0

    print(f"Running Real2Sim Evaluation of model Octo-Small on environment {args.env_id}")
    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()

    while valid_count < args.num_episodes:
        seed = args.seed + eps_count

        # data dump
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(args.num_envs)]

        # env and policy reset
        env_reset_options = {
            "episode_id": torch.arange(args.num_envs) + eps_count
        }
        obs, info = env.reset(seed=seed, options=env_reset_options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8) # on cuda:0
        instruction = env.unwrapped.get_language_instruction()
        model.reset(instruction)
        
        print("instruction[0]:", instruction[0])

        # data dump: instruction
        for idx in range(args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        elapsed_steps = 0
        predicted_terminated, truncated = False, False
        while not (predicted_terminated or truncated):
        # inference
            start_time = time.time()
            raw_action, action = model.step(obs_image, instruction)
            action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

            timers["inference"] += time.time() - start_time

            # step
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            # print("delta action:", action)
            obs_image_new = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
            info = {k: v.cpu().numpy() for k, v in info.items()}
            truncated = bool(truncated.any())  # note that all envs truncate and terminate at the same time.

            timers["env.step"] += time.time() - start_time

            # print info
            info_dict = {k: v.mean().tolist() for k, v in info.items()}
            print(f"step {elapsed_steps}: {info_dict}")

            # data dump: image, action, info
            for i in range(args.num_envs):
                log_image = obs_image[i].cpu().numpy()
                log_action = action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in info.items()}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # add count
            obs_image = obs_image_new
            elapsed_steps += 1

        # data dump: last image
        for i in range(args.num_envs):
            log_image = obs_image[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        if args.save_video:
            exp_dir = Path(args.record_dir) / args.env_id / f"{args.num_episodes}" / "videos"
            exp_dir.mkdir(parents=True, exist_ok=True)

            for i in range(args.num_envs):
                images = datas[i]["image"]
                infos = datas[i]["info"]
                assert len(images) == len(infos) + 1

                if args.info_on_video:
                    for j in range(len(infos)):
                        images[j + 1] = visualization.put_info_on_image(images[j + 1], infos[j])

                success = np.sum([d["success"] for d in infos]) >= 6
                if args.save_successful_videos_only:
                    if not success:
                        continue
                images_to_video(images, str(exp_dir), f"video_{eps_count + i}_success={success}",
                                fps=10, verbose=False)

        # save data
        exp_dir = Path(args.record_dir) / args.env_id / f"{args.num_episodes}" / "data"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.num_envs):
            if np.sum([d["success"] for d in datas[i]["info"]]) < 6:
                continue
            res = datas[i].copy()
            res["image"] = [Image.fromarray(im).convert("RGB") for im in res["image"]]
            np.save(exp_dir / f"data_{eps_count + i:0>4d}.npy", res)
            valid_count += 1
        # metrics log and print
        for k, v in info.items():
            for i in range(args.num_envs):
                info_i = np.array([inf[k] for inf in datas[i]["info"]]).sum() >= 6
                eval_metrics[k].append(int(info_i))
            print(f"{k}: {np.mean(eval_metrics[k])}")

        eps_count += args.num_envs
        
        print(f"Episode {eps_count} finished. Valid count: {valid_count}/{args.num_episodes}")

    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]

    print("\nTiming Info:")
    for key, value in timers.items():
        print(f"{key}: {value:.2f} seconds")

    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["total_episodes"] = eps_count
    mean_metrics["total_steos"] = eps_count * args.max_episode_len
    mean_metrics["time/episodes_per_second"] = eps_count / timers["total"]

    exp_dir = Path(args.record_dir) / args.env_id / f"{args.num_episodes}" / "videos"
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / f"eval_metrics.json"
    json.dump(mean_metrics, open(metrics_path, "w"), indent=4)
    print(f"Evaluation complete. Results saved to {exp_dir}. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
