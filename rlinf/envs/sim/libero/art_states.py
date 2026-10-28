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

"""Bounded reservoir capture and task-matched state sampling for LIBERO ART."""

from collections.abc import Callable

import numpy as np


class ARTStateReservoir:
    """Reserve at most ``capacity`` snapshots per source trajectory.

    Reservoir decisions precede simulator capture, so rejected states incur no
    snapshot copies or IPC. A bank lasts one frozen-policy collection rollout.
    """

    def __init__(self, num_trajectories: int, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.seen = np.zeros(num_trajectories, dtype=np.int64)
        self.states: list[list[dict]] = [[] for _ in range(num_trajectories)]

    def capture(self, records: list[dict], capture_snapshots: Callable) -> None:
        """Capture only admitted records; each record names its local ``env_id``."""
        accepted = []
        for record in records:
            env_id = record["env_id"]
            self.seen[env_id] += 1
            slot = (
                int(self.seen[env_id] - 1)
                if self.seen[env_id] <= self.capacity
                else int(self.rng.integers(self.seen[env_id]))
            )
            if slot < self.capacity:
                accepted.append((record, slot))
        if not accepted:
            return
        snapshots = capture_snapshots([record["env_id"] for record, _ in accepted])
        for (record, slot), snapshot in zip(accepted, snapshots):
            record = {**record, "snapshot": snapshot}
            states = self.states[record["env_id"]]
            if slot == len(states):
                states.append(record)
            else:
                states[slot] = record

    def records(self) -> list[dict]:
        """Return retained records; label source outcomes after collection ends."""
        return [record for states in self.states for record in states]


def select_art_states(
    states: list[dict],
    task_ids: np.ndarray,
    rng: np.random.Generator,
    sampler: str,
    max_episode_steps: int,
    success_fraction: float = 0.7,
    failure_max_step_fraction: float = 0.5,
) -> list[dict]:
    """Sample temporal strata, optionally changing source-outcome composition.

    ``legacy`` and ``uniform`` use the supplied pool unchanged. The caller owns
    whether it contains all leader states or bounded all-trajectory states.
    """
    selected = [None] * len(task_ids)

    def draw(pool: list[dict], groups: np.ndarray) -> None:
        if not len(groups):
            return
        if not pool:
            raise ValueError("ART state sampler has no eligible states for a task.")
        pool = sorted(pool, key=lambda state: state["elapsed_steps"])
        # Tiny outcome-specific pools can contain fewer states than groups.
        strata = np.array_split(np.arange(len(pool)), len(groups))
        for group, stratum in zip(groups, strata):
            index = rng.choice(stratum if len(stratum) else len(pool))
            selected[group] = pool[index]

    for task_id in np.unique(task_ids):
        groups = np.flatnonzero(task_ids == task_id)
        pool = [state for state in states if state["task_id"] == task_id]
        if sampler == "early_failure":
            pool = [
                state
                for state in pool
                if state["source_success"]
                or state["elapsed_steps"]
                <= max_episode_steps * failure_max_step_fraction
            ]
        if sampler == "success_mixture":
            successful = [state for state in pool if state["source_success"]]
            failed = [state for state in pool if not state["source_success"]]
            if successful and failed:
                shuffled = rng.permutation(groups)
                count = round(len(groups) * success_fraction)
                draw(successful, shuffled[:count])
                draw(failed, shuffled[count:])
                continue
        draw(pool, groups)
    return selected
