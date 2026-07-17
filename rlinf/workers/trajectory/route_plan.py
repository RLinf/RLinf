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

"""Deterministic slot routing for trajectory workers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from rlinf.scheduler.worker.worker_group import WorkerGroup


@dataclass(frozen=True)
class TrajectoryRoute:
    """One disjoint shard of a component rank's local batch."""

    channel_worker_rank: int
    local_indices: tuple[int, ...]
    slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrajectoryRoutePlan:
    """Map component-local batches onto trajectory worker slot shards.

    Every component is assumed to partition the same ``total_slots`` into
    contiguous rank-local batches.  The plan partitions those global slots once
    more across trajectory workers, so a source batch can be split into one or
    more disjoint :class:`TrajectoryRoute` objects.
    """

    total_slots: int
    trajectory_world_size: int
    component_world_sizes: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.total_slots <= 0:
            raise ValueError("total_slots must be positive.")
        if self.trajectory_world_size <= 0:
            raise ValueError("trajectory_world_size must be positive.")
        for component, world_size in self.component_world_sizes.items():
            if world_size <= 0:
                raise ValueError(
                    f"World size for component {component!r} must be positive."
                )

    @classmethod
    def from_cfg(
        cls,
        cfg,
        *,
        trajectory_world_size: int,
        component_world_sizes: Mapping[str, int],
    ) -> "TrajectoryRoutePlan":
        """Build a route plan from the configured number of training env slots."""
        return cls(
            total_slots=int(cfg.env.train.total_num_envs),
            trajectory_world_size=trajectory_world_size,
            component_world_sizes=dict(component_world_sizes),
        )

    @classmethod
    def from_worker_groups(
        cls,
        cfg: Any,
        *,
        trajectory_worker_group: "WorkerGroup",
        component_worker_groups: Mapping[str, "WorkerGroup"],
    ) -> "TrajectoryRoutePlan":
        """Build a plan from the launched component worker groups.

        Worker-group membership is the source of truth for component world
        sizes; cfg supplies the global number of train-environment slots.
        """
        return cls.from_cfg(
            cfg,
            trajectory_world_size=len(trajectory_worker_group.worker_info_list),
            component_world_sizes={
                component: len(worker_group.worker_info_list)
                for component, worker_group in component_worker_groups.items()
            },
        )

    def routes_for(self, component: str, rank: int) -> list[TrajectoryRoute]:
        """Return disjoint trajectory-worker routes for a component rank."""
        source_start, source_end = self.component_slot_range(component, rank)
        routes: list[TrajectoryRoute] = []

        for worker_rank in range(self.trajectory_world_size):
            worker_start, worker_end = self._partition_range(
                worker_rank,
                self.trajectory_world_size,
            )
            overlap_start = max(source_start, worker_start)
            overlap_end = min(source_end, worker_end)
            if overlap_start >= overlap_end:
                continue

            routes.append(
                TrajectoryRoute(
                    channel_worker_rank=worker_rank,
                    local_indices=tuple(
                        range(overlap_start - source_start, overlap_end - source_start)
                    ),
                    slot_ids=tuple(range(overlap_start, overlap_end)),
                )
            )

        return routes

    def component_slot_range(self, component: str, rank: int) -> tuple[int, int]:
        """Return the half-open global slot range owned by a component rank."""
        try:
            world_size = self.component_world_sizes[component]
        except KeyError as error:
            raise ValueError(f"Unknown trajectory component: {component!r}.") from error
        if not 0 <= rank < world_size:
            raise ValueError(
                f"Rank {rank} is outside the {component!r} world size {world_size}."
            )
        return self._partition_range(rank, world_size)

    def channel_worker_slot_range(self, rank: int) -> tuple[int, int]:
        """Return the half-open global slot range owned by one trajectory worker."""
        if not 0 <= rank < self.trajectory_world_size:
            raise ValueError(
                "Rank "
                f"{rank} is outside the trajectory world size "
                f"{self.trajectory_world_size}."
            )
        return self._partition_range(rank, self.trajectory_world_size)

    def component_local_range_on_channel_worker(
        self,
        component: str,
        component_rank: int,
        channel_worker_rank: int,
    ) -> tuple[int, int] | None:
        """Return a component shard's local range on one trajectory worker."""
        worker_start, worker_end = self.channel_worker_slot_range(channel_worker_rank)
        component_start, component_end = self.component_slot_range(
            component, component_rank
        )
        overlap_start = max(worker_start, component_start)
        overlap_end = min(worker_end, component_end)
        if overlap_start >= overlap_end:
            return None
        return overlap_start - worker_start, overlap_end - worker_start

    def channel_worker_for_slot(self, slot_id: int) -> int:
        """Return the trajectory worker that owns one global slot."""
        self._validate_slot_id(slot_id)
        return min(
            ((slot_id + 1) * self.trajectory_world_size - 1) // self.total_slots,
            self.trajectory_world_size - 1,
        )

    def component_rank_for_slot(self, component: str, slot_id: int) -> int:
        """Return the rank of ``component`` that owns one global slot."""
        self._validate_slot_id(slot_id)
        try:
            world_size = self.component_world_sizes[component]
        except KeyError as error:
            raise ValueError(f"Unknown trajectory component: {component!r}.") from error
        return min(
            ((slot_id + 1) * world_size - 1) // self.total_slots,
            world_size - 1,
        )

    def _partition_range(self, rank: int, world_size: int) -> tuple[int, int]:
        return (
            self.total_slots * rank // world_size,
            self.total_slots * (rank + 1) // world_size,
        )

    def _validate_slot_id(self, slot_id: int) -> None:
        if not 0 <= slot_id < self.total_slots:
            raise ValueError(f"Slot {slot_id} is outside [0, {self.total_slots}).")
