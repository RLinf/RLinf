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

"""Trajectory channel worker orchestration."""

import asyncio
from collections import defaultdict

from omegaconf import DictConfig

from rlinf.scheduler.channel.trajectory_channel.assembler import (
    TrajectoryEventAssembler,
)
from rlinf.scheduler.channel.trajectory_channel.collectors import (
    create_trajectory_collector,
)
from rlinf.scheduler.channel.trajectory_channel.data import TrajectoryData
from rlinf.scheduler.worker.worker import Worker, WorkerAddress


class TrajectoryWorker(Worker):
    """Receive trajectory events and serve collector outputs."""

    def __init__(self, cfg: DictConfig, max_size: int = 0):
        """Initialize event assembly, collection, and output queues."""
        super().__init__()
        if max_size != 0:
            raise ValueError("Trajectory output queues must be unbounded.")
        self._cfg = cfg
        self._receiver_tasks: dict[WorkerAddress, asyncio.Task] = {}
        self._receiver_error: BaseException | None = None
        self._output_queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_size)
        )

        from rlinf.scheduler.cluster import Cluster
        from rlinf.utils.metric_utils import compute_split_num
        from rlinf.utils.placement import HybridComponentPlacement

        placement = HybridComponentPlacement(cfg, Cluster())
        env_world_size = placement.get_world_size("env")
        actor_world_size = placement.get_world_size("actor")
        self.source_count = env_world_size * cfg.rollout.pipeline_stage_num
        self.chunk_count = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.output_count = (
            compute_split_num(self.source_count, actor_world_size) * actor_world_size
        )
        if self.output_count % self.source_count:
            raise ValueError(
                "Trajectory routing requires each rollout source to have an equal "
                "number of actor shards."
            )
        self.shards_per_source = self.output_count // self.source_count

        self._assembler = TrajectoryEventAssembler(
            source_batch_size=cfg.env.train.total_num_envs // self.source_count
        )
        self._collector = create_trajectory_collector(
            cfg,
            source_count=self.source_count,
            chunk_count=self.chunk_count,
            shards_per_source=self.shards_per_source,
            actor_world_size=actor_world_size,
        )

    async def start_receivers(self, producer_addresses: list[WorkerAddress]) -> None:
        """Start one long-running receive loop for every producer."""
        if self._receiver_tasks:
            raise RuntimeError("Trajectory receiver loops have already been started.")
        if len(set(producer_addresses)) != len(producer_addresses):
            raise ValueError("Trajectory producer addresses must be unique.")

        for address in producer_addresses:
            task = asyncio.create_task(self._consume(address))
            task.add_done_callback(self._on_receiver_done)
            self._receiver_tasks[address] = task

        # Let every task submit its first recv before producers begin publishing.
        await asyncio.sleep(0)

    async def _consume(self, worker_address: WorkerAddress) -> None:
        """Receive trajectory events from one producer in send order."""
        while True:
            data = await self.recv(
                src_group_name=worker_address.root_group_name,
                src_rank=worker_address.rank,
                async_op=True,
            ).async_wait()
            self._apply_event(data)

    def _on_receiver_done(self, task: asyncio.Task) -> None:
        """Retain unexpected receiver failures for subsequent subscribers."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self._receiver_error is None:
            self._receiver_error = error
            self.log_error(f"Trajectory receiver failed: {error}")

    def _apply_event(self, data: TrajectoryData) -> None:
        """Assemble one event and enqueue any completed collector outputs."""
        for chunk in self._assembler.push(data):
            outputs = self._collector.collect(chunk)
            self._assembler.acknowledge(chunk.key)
            for output in outputs:
                self._output_queues[output.queue_key].put_nowait(output.data)

    async def subscribe(
        self, worker_address: WorkerAddress, queue_key: str, query_id: int
    ) -> None:
        """Send the next queued item to a subscriber."""
        if self._receiver_error is not None:
            raise RuntimeError(
                "A trajectory receiver has failed."
            ) from self._receiver_error
        data = await self._output_queues[queue_key].get()
        await self.send(
            object=data,
            dst_group_name=worker_address.root_group_name,
            dst_rank=worker_address.rank,
            piggyback_payload=query_id,
            async_op=True,
        ).async_wait()

    async def try_subscribe(
        self, worker_address: WorkerAddress, queue_key: str, query_id: int
    ) -> bool:
        """Submit a send if an assembled item is immediately available."""
        if self._receiver_error is not None:
            raise RuntimeError(
                "A trajectory receiver has failed."
            ) from self._receiver_error
        try:
            data = self._output_queues[queue_key].get_nowait()
        except asyncio.QueueEmpty:
            return False

        self.send(
            object=data,
            dst_group_name=worker_address.root_group_name,
            dst_rank=worker_address.rank,
            piggyback_payload=query_id,
            async_op=True,
        )
        return True
