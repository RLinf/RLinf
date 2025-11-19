import time
from functools import partial

import psutil
import pynvml
from tqdm import tqdm

import gymnasium as gym

import torch

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import SimConfig

from mshab.utils.bench.raytracing.utils.fetch_1cam import Fetch1Cam
from mshab.utils.bench.raytracing.utils.scene_builder_fetch_1cam import (
    ReplicaCADSceneBuilderFetch1Cam,
)


env = gym.make(
    "SceneManipulation-v1",
    num_envs=1,
    obs_mode="rgbd",  # this will render 1 rgbd image as part of obs
    reward_mode=None,
    control_mode="pd_joint_delta_pos",
    render_mode=None,
    shader_dir="rt-fast",
    robot_uids="fetch_1cam",
    sim_backend="gpu",
    # time limit
    max_episode_steps=100_000,
    # SceneManipulationEnv args
    scene_builder_cls=partial(
        ReplicaCADSceneBuilderFetch1Cam, include_staging_scenes=True
    ),
    build_config_idxs=[10],
    sim_config=SimConfig(sim_freq=120, control_freq=30),
    sensor_configs=dict(
        base_camera=dict(
            pose=sapien_utils.look_at([0.3, 0, 2], [-0.1, 0, 0.1]),
            height=128,
            width=128,
            fov=2,
            near=0.01,
            far=100,
        ),  # NOTE (arth): 128x128 rgbd sensor
    ),
)

obs, info = env.reset()

# Initialize NVML for GPU memory tracking
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

total_time = 0
num_runs = 10
steps_per_run = 300
metrics = {"sps": [], "vram_usage": [], "cpu_mem_usage": []}  # In MB  # In MB

for run in tqdm(range(num_runs)):
    running_time = 0
    env.reset()
    for step in range(steps_per_run):
        actions = torch.rand(env.action_space.shape).clip(-0.3, 0.3)

        # Time only the step call
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(
            actions
        )  # renders base_camera rgbd (from SceneManipulationEnv default_sensor_configs)
        running_time += time.time() - start

    # Calculate metrics
    sps = steps_per_run * env.num_envs / running_time
    vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # Convert to MB
    cpu_mem = psutil.Process().memory_info().rss / 1024**2  # Convert to MB

    metrics["sps"].append(sps)
    metrics["vram_usage"].append(vram)
    metrics["cpu_mem_usage"].append(cpu_mem)
    print(f"Run {run}: SPS={sps:.2f}, VRAM={vram:.1f}MB, CPU Mem={cpu_mem:.1f}MB")

# Calculate mean and 95% confidence intervals for each metric
for metric_name, values in metrics.items():
    values_tensor = torch.tensor(values)
    mean = torch.mean(values_tensor)
    std = torch.std(values_tensor)
    ci_95 = 1.96 * std / torch.sqrt(torch.tensor(len(values)))

    unit = "MB" if metric_name != "sps" else "steps/sec"
    print(f"{metric_name}: {mean:.2f} ± {ci_95:.2f} {unit} (95% CI)")

pynvml.nvmlShutdown()
