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

import asyncio

import gymnasium as gym
import torch

from rlinf.utils.env_helpers.delay_sampler import DelaySampler


class InsertDelay(gym.Wrapper):
    """Insert a configurable delay after each step, chunk_step and reset to
    emulate per-environment sensor / action latency.

    ``step``/``chunk_step``/``reset`` only *sample* the delay; the wait itself is
    taken by ``await wait_delay()``. The env APIs are synchronous but are driven
    from the env worker's event loop, so sleeping inside them would block every
    other coroutine in the process for the whole delay. Sampled delays accumulate
    until they are waited out, so a caller that drains them late still pays the
    full latency rather than losing it.

    Sampled delays are also buffered for metrics and consumed via
    ``insert_delay_metrics()``.
    """

    def __init__(self, env, delay_cfg):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env
        self.sampler = DelaySampler.create(delay_cfg)
        self.delays: list[float] = []
        self._pending_delay = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.sample_delay()
        return obs, reward, terminated, truncated, info

    def chunk_step(self, *args, **kwargs):
        result = self.env.chunk_step(*args, **kwargs)
        self.sample_delay()
        return result

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.sample_delay()
        return obs, info

    def sample_delay(self):
        """Sample one delay and hold it until ``wait_delay`` is awaited."""
        delay = self.sampler.sample_one()
        self.delays.append(delay)
        self._pending_delay += delay

    async def wait_delay(self):
        """Wait out every delay sampled since the last call."""
        pending, self._pending_delay = self._pending_delay, 0.0
        if pending > 0:
            await asyncio.sleep(pending)

    def insert_delay_metrics(self) -> torch.Tensor:
        delays = self.delays[:]
        self.delays.clear()
        return torch.tensor(delays, dtype=torch.float32).reshape(-1).cpu()
