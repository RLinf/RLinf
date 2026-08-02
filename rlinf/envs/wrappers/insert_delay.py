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

import time

import gymnasium as gym
import torch

from rlinf.utils.delay_sampler import DelaySampler


class InsertDelay(gym.Wrapper):
    """Insert a configurable delay after each step, chunk_step and reset to
    emulate per-environment sensor / action latency. Sampled delays are
    buffered internally and can be consumed via ``insert_delay_metrics()``.
    """

    def __init__(self, env, delay_cfg):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env
        self.sampler = DelaySampler.create(delay_cfg)
        self.delays: list[float] = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.delay()
        return obs, reward, terminated, truncated, info

    def chunk_step(self, *args, **kwargs):
        result = self.env.chunk_step(*args, **kwargs)
        self.delay()
        return result

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.delay()
        return obs, info

    def delay(self):
        delay = self.sampler.sample_one()
        self.delays.append(delay)
        time.sleep(delay)

    def insert_delay_metrics(self) -> torch.Tensor:
        delays = self.delays[:]
        self.delays.clear()
        return torch.tensor(delays, dtype=torch.float32).reshape(-1).cpu()
