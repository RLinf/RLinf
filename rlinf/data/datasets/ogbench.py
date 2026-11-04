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


import numpy as np


class OGBenchDataset:
    """An append-only offline/online replay with explicit episode boundaries.

    Capacity is fixed for a run; refusing overflow prevents chunks crossing the
    chronological discontinuity of a circular buffer.
    """

    keys = (
        "observations",
        "actions",
        "rewards",
        "masks",
        "terminals",
        "next_observations",
    )

    def __init__(self, data, seed=0, extra_capacity=0):
        self.size = len(data["observations"])
        self.capacity = self.size + extra_capacity
        self.rng = np.random.RandomState(seed)
        self.data = {}
        for key in self.keys:
            value = np.asarray(data[key], dtype=np.float32)
            if len(value) != self.size:
                raise ValueError(f"OGBench field {key} has inconsistent length")
            self.data[key] = np.empty((self.capacity, *value.shape[1:]), np.float32)
            self.data[key][: self.size] = value
        self.data["actions"][: self.size] = np.clip(
            self.data["actions"][: self.size], -1 + 1e-5, 1 - 1e-5
        )
        self.offline_size = self.size

    @classmethod
    def from_task(cls, task_name, seed=0, extra_capacity=0):
        import ogbench

        env, train, _ = ogbench.make_env_and_datasets(task_name)
        env.close()
        return cls(train, seed, extra_capacity)

    def sample(self, batch_size, horizon, discount, indices=None):
        if not 1 <= horizon <= self.size:
            raise ValueError(f"Expected 1 <= horizon <= {self.size}, got {horizon}")
        if indices is None:
            indices = self.rng.randint(self.size - horizon + 1, size=batch_size)
        indices = np.asarray(indices)
        idx = indices[:, None] + np.arange(horizon)
        result = {k: self.data[k][idx].copy() for k in self.keys}
        result["observations"] = self.data["observations"][indices].copy()
        valid = np.ones((len(indices), horizon), np.float32)
        for i in range(1, horizon):
            valid[:, i] = 1 - result["terminals"][:, i - 1]
            result["rewards"][:, i] = (
                result["rewards"][:, i - 1]
                + result["rewards"][:, i] * discount**i * valid[:, i]
            )
            result["masks"][:, i] = np.minimum(
                result["masks"][:, i - 1], result["masks"][:, i]
            ) * valid[:, i] + result["masks"][:, i - 1] * (1 - valid[:, i])
            result["terminals"][:, i] = np.maximum(
                result["terminals"][:, i - 1], result["terminals"][:, i]
            )
            result["next_observations"][:, i] = result["next_observations"][
                :, i
            ] * valid[:, i, None] + result["next_observations"][:, i - 1] * (
                1 - valid[:, i, None]
            )
        result["valid"] = valid
        return result

    def add(self, transition):
        if self.size >= self.capacity:
            raise RuntimeError(
                "OGBench replay capacity exceeded; increase online_steps capacity"
            )
        for key in self.keys:
            self.data[key][self.size] = transition[key]
        self.size += 1

    def state_dict(self):
        return {
            "rng": self.rng.get_state(),
            "offline_size": self.offline_size,
            "online": {
                k: v[self.offline_size : self.size].copy() for k, v in self.data.items()
            },
        }

    def load_state_dict(self, state):
        if state["offline_size"] != self.offline_size:
            raise ValueError("Checkpoint and current offline dataset sizes differ")
        count = len(state["online"]["observations"])
        if self.offline_size + count > self.capacity:
            raise ValueError("Checkpoint replay exceeds configured capacity")
        self.size = self.offline_size + count
        for k, v in state["online"].items():
            self.data[k][self.offline_size : self.size] = v
        self.rng.set_state(state["rng"])
