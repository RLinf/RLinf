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
import uuid

import ray
from omegaconf import DictConfig

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.scheduler.channel.trajectory_channel.data import (
    TrajectoryData,
)
from rlinf.scheduler.channel.trajectory_channel.trajectory_worker import (
    TrajectoryWorker,
)
from rlinf.scheduler.channel.trajectory_channel.work import (
    AsyncPublishWork,
    AsyncSubscribeWork,
)
from rlinf.scheduler.collective import AsyncWork
from rlinf.scheduler.worker.worker import Worker, WorkerAddress
from rlinf.scheduler.worker.worker_group import WorkerGroup


class TrajectoryChannel:
    """P2P channel backed by a dedicated trajectory assembly worker."""

    def __init__(
        self,
        cfg: "DictConfig",
        name: str,
        trajectory_worker_group: WorkerGroup["TrajectoryWorker"],
    ):
        """Initialize a channel from its trajectory worker group."""
        self._cfg = cfg
        self._name = name
        self._trajectory_worker_group = trajectory_worker_group
        self._trajectory_workers: dict[WorkerGroup.WorkerRank, ray.ObjectRef] = {
            worker_info.rank: worker_info.worker
            for worker_info in trajectory_worker_group.worker_info_list
        }

    @classmethod
    def create(
        cls,
        name: str,
        cfg: "DictConfig",
        trajectory_worker_group: WorkerGroup["TrajectoryWorker"] | None = None,
        trajectory_node_rank: int | None = None,
    ) -> "TrajectoryChannel":
        """Create or attach to a single trajectory worker."""
        if trajectory_worker_group is None:
            if trajectory_node_rank is None:
                trajectory_node_rank = cfg.cluster.get("trajectory_node_rank", 0)
            if (
                trajectory_node_rank < 0
                or trajectory_node_rank >= cfg.cluster.num_nodes
            ):
                raise ValueError(
                    f"Invalid trajectory_node_rank: {trajectory_node_rank}. It must be in the range [0, {cfg.cluster.num_nodes - 1}]."
                )
            from rlinf.scheduler.cluster.cluster import Cluster
            from rlinf.scheduler.placement import NodePlacementStrategy

            trajectory_worker_group = TrajectoryWorker.create_group(cfg).launch(
                cluster=Cluster(),
                name=f"{name}.trajectory_worker",
                placement_strategy=NodePlacementStrategy([trajectory_node_rank]),
                max_concurrency=2**31 - 1,
            )
        return cls(cfg, name, trajectory_worker_group)

    def publish(
        self, data: TrajectoryData, async_op: bool = False
    ) -> AsyncPublishWork | None:
        """Publish a trajectory event from the current worker."""
        if not self._current_worker:
            raise RuntimeError(
                "TrajectoryChannel methods must be called from within a Worker."
            )
        worker = self._current_worker
        send_work: AsyncWork = worker.send(
            object=data,
            dst_group_name=self._trajectory_worker_group.worker_group_name,
            dst_rank=0,
            async_op=True,
        )

        async_publish_work = AsyncPublishWork(send_work)
        if async_op:
            return async_publish_work
        async_publish_work.wait()
        return None

    def start_receivers(self, *producer_groups: WorkerGroup) -> None:
        """Start one persistent receive loop for every producer worker."""
        producer_addresses = [
            WorkerAddress(group.worker_group_name, ranks=worker.rank)
            for group in producer_groups
            for worker in group.worker_info_list
        ]
        receiver_refs = [
            actor.start_receivers.remote(producer_addresses)
            for actor in self._trajectory_workers.values()
        ]
        producer_refs = [
            worker.worker.initialize_p2p.remote(
                self._trajectory_worker_group.worker_group_name, 0
            )
            for group in producer_groups
            for worker in group.worker_info_list
        ]
        ray.get(receiver_refs + producer_refs)

    def subscribe(
        self, query_key: str = "default", async_op: bool = False
    ) -> AsyncSubscribeWork | Trajectory:
        """Subscribe to the next assembled item in a queue."""
        if not self._current_worker:
            raise RuntimeError(
                "TrajectoryChannel methods must be called from within a Worker."
            )
        worker = self._current_worker
        trajectory_worker_actor = self._trajectory_workers[0]
        query_id = uuid.uuid4().int
        subscribe_ref: ray.ObjectRef = trajectory_worker_actor.subscribe.remote(
            worker.worker_address, query_key, query_id
        )
        recv_work: AsyncWork = worker.recv(
            src_group_name=self._trajectory_worker_group.worker_group_name,
            src_rank=0,
            async_op=True,
        )

        async_subscribe_work = AsyncSubscribeWork(subscribe_ref, recv_work, query_id)
        return async_subscribe_work if async_op else async_subscribe_work.wait()

    def try_subscribe(
        self, query_key: str = "default", async_op: bool = False
    ) -> AsyncSubscribeWork | Trajectory:
        """Subscribe without waiting when no assembled item is available.

        Raises:
            asyncio.QueueEmpty: If the trajectory worker's queue is empty.
        """
        if not self._current_worker:
            raise RuntimeError(
                "TrajectoryChannel methods must be called from within a Worker."
            )
        worker = self._current_worker
        trajectory_worker_actor = self._trajectory_workers[0]
        query_id = uuid.uuid4().int
        subscribe_ref: ray.ObjectRef = trajectory_worker_actor.try_subscribe.remote(
            worker.worker_address, query_key, query_id
        )
        if not ray.get(subscribe_ref):
            raise asyncio.QueueEmpty

        recv_work: AsyncWork = worker.recv(
            src_group_name=self._trajectory_worker_group.worker_group_name,
            src_rank=0,
            async_op=True,
        )
        async_subscribe_work = AsyncSubscribeWork(subscribe_ref, recv_work, query_id)
        return async_subscribe_work if async_op else async_subscribe_work.wait()

    def __getstate__(self) -> dict:
        """Serialize without process-local worker state."""
        state = self.__dict__.copy()
        state.pop("_current_worker", None)
        return state

    def __setstate__(self, state: dict):
        """Restore process-local worker state after deserialization."""
        self.__dict__.update(state)
        self._current_worker = Worker.current_worker
