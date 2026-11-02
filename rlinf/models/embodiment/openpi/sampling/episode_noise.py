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

"""CPU JAX noise indexed by episode seed and policy query."""

from typing import Any

import numpy as np


class JaxEpisodeNoise:
    """Match OpenPI's repeated key splitting without advancing on replayed queries."""

    def __init__(self):
        self._states: dict[int, tuple[int, Any, np.ndarray]] = {}

    def __call__(
        self,
        episode_seeds: np.ndarray,
        query_indices: np.ndarray,
        action_shape: tuple[int, int],
    ) -> np.ndarray:
        """Return [B,H,D] noise; each episode follows native batch-size-one sampling."""
        import jax
        import jax.numpy as jnp

        seeds, indices = np.asarray(episode_seeds), np.asarray(query_indices)
        if (
            seeds.ndim != 1
            or seeds.shape != indices.shape
            or not len(seeds)
            or not np.issubdtype(seeds.dtype, np.integer)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(seeds < 0)
            or np.any(seeds >= 2**32)
            or np.any(indices < 0)
            or len(action_shape) != 2
            or min(action_shape) <= 0
        ):
            raise ValueError(
                "Expected integer episode seeds [B] in [0,2**32), "
                f"nonnegative query indices [B] and positive (H,D); "
                f"got {seeds}, {indices}, {action_shape}"
            )
        batch = []
        current = {}
        with jax.default_device(jax.devices("cpu")[0]):
            for seed, query in zip(seeds.tolist(), indices.tolist(), strict=True):
                state = self._states.get(seed)
                if state is None or query < state[0] or state[2].shape != action_shape:
                    previous, key = -1, jax.random.key(seed, impl="threefry2x32")
                else:
                    previous, key, values = state
                if query != previous:
                    for _ in range(previous + 1, query + 1):
                        key, sample_key = jax.random.split(key)
                    values = np.asarray(
                        jax.random.normal(
                            sample_key, (1, *action_shape), dtype=jnp.float32
                        )
                    )[0].copy()
                current[seed] = (query, key, values)
                batch.append(values)
        self._states = current
        return np.stack(batch)
