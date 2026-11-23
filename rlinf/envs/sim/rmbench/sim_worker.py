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


import json
import random
import sys
from collections import deque
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np


class PolicyClient:
    """Send RMBench camera observations to the RLinf policy."""

    def __init__(
        self,
        connection: Connection,
        short_memory: dict[str, Any] | None = None,
    ) -> None:
        self.connection = connection
        self.seed = 0
        self.stride = 1
        self.time_offsets = None
        capacity = 1
        if short_memory is not None:
            gaps = short_memory["num_frames"] - 1
            self.stride = short_memory["stride"]
            self.time_offsets = np.arange(-gaps, 1, dtype=np.float32) * (
                self.stride / short_memory["fps"]
            )
            capacity += gaps * self.stride
        self.frames = deque(maxlen=capacity)

    def reset(self, seed: int) -> None:
        """Start a new episode with the benchmark's accepted seed."""
        self.frames.clear()
        self.seed = seed

    def observe(self, observation: dict[str, Any]) -> None:
        """Save the head and both wrist images at one control tick."""
        cameras = observation["observation"]
        self.frames.append(
            (
                cameras["head_camera"]["rgb"],
                cameras["left_camera"]["rgb"],
                cameras["right_camera"]["rgb"],
            )
        )

    def predict(self, state: np.ndarray, instruction: str) -> np.ndarray:
        """Query with the current state, instruction and camera observations."""
        main, left, right = self.frames[-1]
        observation = {
            "states": np.asarray(state, dtype=np.float32)[None],
            "task_descriptions": [instruction],
        }
        if self.time_offsets is None:
            observation["main_images"] = main[None]
            observation["wrist_images"] = np.stack((left, right))[None]
        else:
            observation["history"] = self._history(main, left)
        self.connection.send((self.seed, observation))
        return self.connection.recv()

    def _history(self, main: np.ndarray, wrist: np.ndarray) -> dict[str, np.ndarray]:
        frames = list(self.frames)[:: -self.stride][::-1]
        num_frames = len(self.time_offsets)
        missing = num_frames - len(frames)
        history_main = np.zeros((1, num_frames, *main.shape), dtype=main.dtype)
        history_wrists = np.zeros((1, num_frames, 2, *wrist.shape), dtype=wrist.dtype)
        for index, (main_frame, left, right) in enumerate(frames, start=missing):
            history_main[0, index] = main_frame
            history_wrists[0, index, 0] = left
            history_wrists[0, index, 1] = right
        return {
            "main_images": history_main,
            "wrist_images": history_wrists,
            "frame_mask": (np.arange(num_frames) >= missing)[None],
            "time_offsets": self.time_offsets[None],
        }


def reset_model(model: PolicyClient) -> None:
    """Start an episode with an empty history."""
    import torch

    # RMBench calls torch.manual_seed() during setup with its accepted seed.
    model.reset(torch.initial_seed())


def eval(task_env: Any, model: PolicyClient, observation: dict[str, Any]) -> None:
    """Execute one policy chunk and capture the observations between actions."""
    model.observe(observation)
    actions = model.predict(
        observation["joint_action"]["vector"], task_env.get_instruction()
    )
    for index, action in enumerate(actions):
        task_env.take_action(action, action_type="qpos")
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break
        # The benchmark reads the last observation of a chunk before the next query.
        if model.time_offsets is not None and index + 1 < len(actions):
            model.observe(task_env.get_obs())


def run(connection: Connection, cfg: dict[str, Any]) -> None:
    """Evaluate one task using RMBench's expert screening and scoring."""
    from scripts.eval_policy import eval_policy
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

    from rlinf.utils.logging import get_logger

    task_name = cfg["task_name"]
    episodes = cfg["episodes"]
    start_seed = 100_000 * (1 + cfg["seed"])
    if episodes < 1:
        raise ValueError(f"Expected at least one episode, got {episodes}.")
    random.seed(start_seed)
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    args, _ = load_task_args(
        {"task_name": task_name, "policy_name": "rlinf.envs.sim.rmbench.sim_worker"}
    )
    args["ckpt_setting"] = "RLinf"
    args["data_type"]["endpose"] = False
    args["data_type"]["third_view"] = cfg["save_video"]
    video_size = None
    if cfg["save_video"]:
        video_dir = log_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        args["eval_video_save_dir"] = video_dir
        video_size = "320x240"

    _, successes, total_reward = eval_policy(
        task_name,
        class_decorator(task_name),
        args,
        PolicyClient(connection, cfg["short_memory"]),
        start_seed,
        test_num=episodes,
        video_size=video_size,
        instruction_type=cfg["instruction_type"],
    )

    get_logger().info(
        "RMBench %s (%s instructions, start seed %d): "
        "%d/%d successes (%.1f%%), mean_max_reward=%.6g",
        task_name,
        cfg["instruction_type"],
        start_seed,
        successes,
        episodes,
        100 * successes / episodes,
        total_reward / episodes,
    )


def main() -> None:
    """Run the benchmark in its simulator Python environment."""
    with Connection(int(sys.argv[1])) as connection:
        run(connection, json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
