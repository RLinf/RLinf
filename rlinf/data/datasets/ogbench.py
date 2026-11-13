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
    """An immutable OGBench dataset with episode-aware action chunks."""

    keys = (
        "observations",
        "actions",
        "rewards",
        "masks",
        "terminals",
        "next_observations",
    )

    def __init__(self, data, seed=0):
        self.size = len(data["observations"])
        self.rng = np.random.RandomState(seed)
        self.data = {}
        for key in self.keys:
            value = np.asarray(data[key], dtype=np.float32)
            if len(value) != self.size:
                raise ValueError(f"OGBench field {key} has inconsistent length")
            self.data[key] = value.copy()
        self.data["actions"] = np.clip(self.data["actions"], -1 + 1e-5, 1 - 1e-5)

    @classmethod
    def from_task(cls, task_name, seed=0, dataset_dir=None):
        import ogbench

        kwargs = {"dataset_dir": dataset_dir} if dataset_dir is not None else {}
        env, train, _ = ogbench.make_env_and_datasets(task_name, **kwargs)
        env.close()
        return cls(train, seed)

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

    def state_dict(self):
        return {"rng": self.rng.get_state(), "size": self.size}

    def load_state_dict(self, state):
        if state.get("size", state.get("offline_size")) != self.size:
            raise ValueError("Checkpoint and current offline dataset sizes differ")
        if "online" in state and len(state["online"]["observations"]):
            raise ValueError("Cannot resume an online replay in offline QAM")
        self.rng.set_state(state["rng"])
