"""
This script is to test if users can successfully load all the environments, the benchmark initial states in their machines
"""
import os
from termcolor import colored
import cv2
import h5py
import subprocess
import shutil
import numpy as np

from pathlib import Path

# import init_path
from liberoplus.liberoplus import benchmark, get_libero_path


# def render_task(task, bddl_file, init_states, demo_file):
#     env_args = {
#         "bddl_file_name": bddl_file,
#         "camera_heights": 128,
#         "camera_widths": 128
#     }

#     env = OffScreenRenderEnv(**env_args)
#     env.reset()
#     obs = env.set_init_state(init_states[0])
#     for _ in range(5):
#         obs, _, _, _ = env.step([0.] * 7)
#     images = [obs["agentview_image"]]

#     with h5py.File(demo_file, "r") as f:
#         states = f["data/demo_0/states"][()]
#         obs = env.set_init_state(states[-1])

#     images.append(obs["agentview_image"])
#     images = np.concatenate(images, axis=1)
#     cv2.imwrite(f"benchmark_tasks/{task.problem}-{task.language}.png", images[::-1, :, ::-1])
#     env.close()


def main():

    benchmark_root_path = get_libero_path("benchmark_root")
    init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")

    # Check all the files
    task_tuples = []
    demo_files = []
    for benchmark_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        num_tasks = benchmark_instance.get_num_tasks()
        # see how many tasks involved in the benchmark
        print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")

        # Check if all the task names and their bddl file names
        task_names = benchmark_instance.get_task_names()
        print("The benchmark contains the following tasks:")
        for task_id in range(num_tasks):
            task_name = task_names[task_id]
            task = benchmark_instance.get_task(task_id)
            bddl_file = os.path.join(
                bddl_files_default_path, task.problem_folder, task.bddl_file
            )
            assert os.path.exists(bddl_file), f"{bddl_file} does not exist!"
            init_states_path = os.path.join(
                init_states_default_path, task.problem_folder, task.init_states_file
            )
            assert os.path.exists(
                init_states_path
            ), f"{init_states_path} does not exist!"
            demo_file = os.path.join(
                datasets_default_path,
                benchmark_instance.get_task_demonstration(task_id),
            )
            assert os.path.exists(demo_file), f"{demo_file} does not exist!"
            init_states = benchmark_instance.get_task_init_states(task_id)
            task_tuples.append((benchmark_name, task_id, bddl_file, demo_file))
            demo_files.append(demo_file)

    print(colored("All the files exist!", "green"))
    processes = []
    if os.path.exists("benchmark_tasks"):
        shutil.rmtree("benchmark_tasks")

    for i in range(len(task_tuples)):
        command = f"python benchmark_scripts/render_single_task.py --benchmark_name {task_tuples[i][0]} --task_id {task_tuples[i][1]} --bddl_file {task_tuples[i][2]} --demo_file {task_tuples[i][3]}"
        p = subprocess.Popen(command, shell=True)
        processes.append(p)
        if i % 10 == 9:
            for p in processes:
                p.wait()
            processes = []

    count = len(list(Path("benchmark_tasks").glob("*.png")))
    print(f"Expected 130 tasks, Rendered {count} tasks successfully.")
    if count < 130:
        print(colored("Some tasks failed to render!", "red"))
        for demo_file in demo_files:
            if not os.path.exists(
                os.path.join(
                    "benchmark_tasks", demo_file.split("/")[-1].replace(".hdf5", ".png")
                )
            ):
                print(demo_file)


if __name__ == "__main__":
    main()
