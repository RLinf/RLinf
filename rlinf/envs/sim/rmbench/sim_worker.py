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
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger


class RMBenchSimulator:
    """Own an RMBench task in the simulator's Python environment."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from scripts.eval_policy_xpolicylab import (
            UnStableError,
            class_decorator,
            generate_episode_descriptions,
            load_task_args,
        )

        self.task_name = cfg["task_name"]
        self.instruction_type = cfg["instruction_type"]
        self.episodes = cfg["episodes"]
        self.max_episode_steps = cfg["max_episode_steps"]
        self.record_video = cfg["record_video"]
        self.seed = 100_000 * (1 + cfg["seed"])
        self.episode = 0
        self.active = False
        self._unstable_error = UnStableError
        self._generate_descriptions = generate_episode_descriptions
        random.seed(self.seed)
        self.args, _ = load_task_args(
            {"task_name": self.task_name, "policy_name": "RLinf"}
        )
        self.args.update(eval_mode=True, render_freq=0)
        self.args["data_type"].update(endpose=False, third_view=False)
        self.task = class_decorator(self.task_name)
        self.history = None
        if (short_memory := cfg["short_memory"]) is not None:
            gaps = short_memory["num_frames"] - 1
            self.stride = short_memory["stride"]
            self.offsets = np.arange(-gaps, 1, dtype=np.float32) * (
                self.stride / short_memory["fps"]
            )
            self.history = deque(maxlen=gaps * self.stride + 1)

    def reset(self) -> dict[str, Any]:
        """Screen seeds with the expert and recreate the accepted scene."""
        self.close()
        while True:
            seed = self.seed
            self.seed += 1
            try:
                self.task.setup_demo(
                    now_ep_num=self.episode, seed=seed, is_test=True, **self.args
                )
                expert_info = self.task.play_once()
            except self._unstable_error:
                continue
            finally:
                self.task.close_env()
            if self.task.plan_success and self.task.check_success():
                break

        self.active = True
        self.task.setup_demo(
            now_ep_num=self.episode, seed=seed, is_test=True, **self.args
        )
        descriptions = self._generate_descriptions(
            self.task_name, [expert_info["info"]], self.episodes
        )
        self.instruction = str(np.random.choice(descriptions[0][self.instruction_type]))
        self.task.set_instruction(instruction=self.instruction)
        self.episode += 1
        get_logger().info("RMBench episode %d, seed %d", self.episode, seed)
        if self.history is not None:
            self.history.clear()
        self.observation = self._observe()
        return self.observation

    def chunk_step(
        self, actions: np.ndarray
    ) -> tuple[
        list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
    ]:
        """Execute native joint targets through the episode boundary."""
        shape = (1, len(actions))
        rewards = np.zeros(shape, dtype=np.float32)
        terminations = np.zeros(shape, dtype=np.bool_)
        truncations = np.zeros(shape, dtype=np.bool_)
        if not self.active:
            return [self.observation], rewards, terminations, truncations, {}

        limit = min(self.task.step_lim, self.max_episode_steps)
        infos = {}
        observations = []
        for i, action in enumerate(actions):
            previous_reward = self.task.max_reward
            self.task.take_action(action, action_type="qpos")
            rewards[0, i] = self.task.max_reward - previous_reward
            success = self.task.eval_success
            at_limit = self.task.take_action_cnt >= limit
            if success or at_limit:
                terminations[0, i] = success
                truncations[0, i] = not success and at_limit
                infos["episode"] = {
                    "success_once": np.array([success], dtype=np.bool_),
                    "max_reward": np.array([self.task.max_reward], dtype=np.float32),
                    "episode_len": np.array(
                        [self.task.take_action_cnt], dtype=np.int64
                    ),
                }
                break
            if (self.history is not None or self.record_video) and i + 1 < len(actions):
                main, _, _, _ = self._capture()
                if self.record_video:
                    observations.append({"main_images": main[None]})

        self.observation = self._observe()
        if infos:
            self.close()
        observations.append(self.observation)
        return observations, rewards, terminations, truncations, infos

    def _capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self.task.get_obs()
        cameras = raw["observation"]
        main = cameras["head_camera"]["rgb"]
        left = cameras["left_camera"]["rgb"]
        right = cameras["right_camera"]["rgb"]
        if self.history is not None:
            self.history.append((main, left, right))
        return main, left, right, raw["joint_action"]["vector"]

    def _observe(self) -> dict[str, Any]:
        main, left, right, state = self._capture()
        observation = {
            "main_images": main[None],
            "wrist_images": np.stack((left, right))[None],
            "states": np.asarray(state, dtype=np.float32)[None],
            "task_descriptions": [self.instruction],
        }
        if self.history is not None:
            n = len(self.offsets)
            frames = list(self.history)[:: -self.stride][::-1]
            missing = n - len(frames)
            main_history = np.zeros((1, n, *main.shape), dtype=main.dtype)
            wrist_history = np.zeros((1, n, 2, *left.shape), dtype=left.dtype)
            for i, (head, left_wrist, right_wrist) in enumerate(frames, start=missing):
                main_history[0, i] = head
                wrist_history[0, i, 0] = left_wrist
                wrist_history[0, i, 1] = right_wrist
            observation["history"] = {
                "main_images": main_history,
                "wrist_images": wrist_history,
                "frame_mask": (np.arange(n) >= missing)[None],
                "time_offsets": self.offsets[None],
            }
        return observation

    def close(self) -> None:
        """Release the current scene."""
        if self.active:
            self.active = False
            self.task.close_env(
                clear_cache=(self.episode + 1) % self.args["clear_cache_freq"] == 0
            )


def run(connection: Connection, cfg: dict[str, Any]) -> None:
    """Serve the owning RMBench environment adapter."""
    simulator = RMBenchSimulator(cfg)
    try:
        while True:
            try:
                command, payload = connection.recv()
            except EOFError:
                break
            if command == "reset":
                connection.send(simulator.reset())
            elif command == "chunk_step":
                connection.send(simulator.chunk_step(payload))
            else:
                raise ValueError(f"Unknown RMBench simulator command {command!r}.")
    finally:
        simulator.close()


def main() -> None:
    """Run the simulator command loop in the benchmark's Python environment."""
    with Connection(int(sys.argv[1])) as connection:
        run(connection, json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
