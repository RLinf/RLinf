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

import multiprocessing as mp

import numpy as np

from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = "place_shoe"
    n_envs = 2
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10
    robotwin = RoboTwin(task_name, n_envs, horizon, steps)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        prev_obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            robotwin.init_process()
        )
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim) * 0.05
            actions = np.clip(actions, 0, 1)
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                robotwin.step(actions)
            )

            if step % 10 == 0:
                robotwin.reset()
            if terminated_venv[0] == 1:
                print("main", f"terminated_venv: {terminated_venv}")
            if truncated_venv[0] == 1:
                print("main", f"truncated_venv: {truncated_venv}")
            print("main", f"info_venv: {info_venv}")
        robotwin.clear()
