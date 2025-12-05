import os
from termcolor import colored
import cv2
import h5py
import argparse
import numpy as np

from liberopro.liberopro.envs import OffScreenRenderEnv
from liberopro.liberopro import benchmark, get_libero_path


def render_task(task, bddl_file, init_states, demo_file):
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = OffScreenRenderEnv(**env_args)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    images = [obs["agentview_image"]]

    with h5py.File(demo_file, "r") as f:
        states = f["data/demo_0/states"][()]
        obs = env.set_init_state(states[-1])

    images.append(obs["agentview_image"])
    images = np.concatenate(images, axis=1)
    cv2.imwrite(
        f"benchmark_tasks/{task.problem}-{task.language}.png", images[::-1, :, ::-1]
    )
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--bddl_file", type=str)
    parser.add_argument("--demo_file", type=str)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name
    task_id = args.task_id
    bddl_file = args.bddl_file
    demo_file = args.demo_file

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    os.makedirs("benchmark_tasks", exist_ok=True)

    task = benchmark_instance.get_task(task_id)
    init_states = benchmark_instance.get_task_init_states(task_id)

    env = OffScreenRenderEnv(**env_args)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    images = [obs["agentview_image"]]

    with h5py.File(demo_file, "r") as f:
        states = f["data/demo_0/states"][()]
        obs = env.set_init_state(states[-1])

    images.append(obs["agentview_image"])
    images = np.concatenate(images, axis=1)
    image_name = demo_file.split("/")[-1].replace(".hdf5", ".png")
    cv2.imwrite(f"benchmark_tasks/{image_name}", images[::-1, :, ::-1])
    env.close()


if __name__ == "__main__":
    main()
