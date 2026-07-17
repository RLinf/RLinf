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
from typing import Any

from rlinf.scheduler.channel.channel_worker import (
    DEFAULT_KEY,
    ChannelWorker,
    WeightedItem,
)
from rlinf.scheduler.worker.worker import WorkerAddress
from rlinf.workers.trajectory.compression import (
    TrajectoryCompressionConfig,
    compress_trajectory,
)
from rlinf.workers.trajectory.data import (
    Observations,
    TrajectoryData,
    TrajectoryEnvelope,
)
from rlinf.workers.trajectory.storage import TrajectoryStorage, TrajectoryStorageConfig
from rlinf.workers.trajectory.transport import TransportEndpoint


class TrajectoryChannelWorker(ChannelWorker):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize=maxsize)
        self._storage_config: TrajectoryStorageConfig | None = None
        self._slot_to_local_index: dict[int, int] = {}
        self._actor_slot_indices: dict[int, tuple[int, ...]] = {}
        self._storages: dict[tuple[int, int], TrajectoryStorage] = {}
        self._actor_channel_name = "Actor"
        self._storage_queue: asyncio.Queue[TrajectoryData] = asyncio.Queue()
        self._storage_task: asyncio.Task[None] | None = None
        self._compression_config = TrajectoryCompressionConfig()
        self._trajectory_encode_queue: asyncio.Queue[tuple[Any, Any]] = asyncio.Queue()
        self._trajectory_encode_task: asyncio.Task[None] | None = None
        self._transport_endpoints: dict[str, TransportEndpoint] = {}

    def configure_storage(
        self,
        config: TrajectoryStorageConfig,
        slot_ids: tuple[int, ...],
        actor_slot_indices: dict[int, tuple[int, ...]],
        actor_channel_name: str = "Actor",
        compression_config: TrajectoryCompressionConfig = TrajectoryCompressionConfig(),
    ) -> None:
        """Configure this worker's fixed local trajectory shard."""
        if config.num_envs != len(slot_ids):
            raise ValueError("Storage slot count must match storage config.num_envs.")
        self._storage_config = config
        self._slot_to_local_index = {
            slot_id: local_index for local_index, slot_id in enumerate(slot_ids)
        }
        if set().union(*map(set, actor_slot_indices.values())) != set(
            range(len(slot_ids))
        ):
            raise ValueError("Actor slot indices must partition the local shard.")
        self._actor_slot_indices = actor_slot_indices
        self._actor_channel_name = actor_channel_name
        self._compression_config = compression_config
        self._storages = {}
        self._transport_endpoints = {}

    async def put(
        self,
        src_addr: WorkerAddress,
        nowait: bool = False,
        transport_key: str | None = None,
    ) -> None:
        """Receive a live message and schedule its storage-side handling."""
        item, (key, weight) = self.recv(src_addr.root_group_name, src_addr.rank_path)
        self._bootstrap_transport(src_addr, item, transport_key)
        await self._enqueue_live(item, weight, key, nowait)
        if self._stores_live_item(key, item):
            self._schedule_storage(item)

    async def put_frame(
        self,
        src_addr: WorkerAddress,
        key: Any,
        weight: int,
        transport_key: str,
        nowait: bool = False,
    ) -> None:
        """Receive one fixed tensor frame for live relay and trajectory storage."""
        item = self._receive_frame(src_addr, transport_key)
        await self._enqueue_live(item, weight, key, nowait)
        if self._stores_live_item(key, item):
            self._schedule_storage(item)

    async def put_via_ray(
        self, item: Any, weight: int, key: Any = DEFAULT_KEY, nowait: bool = False
    ) -> None:
        """Receive one driver-originated record through Ray transport."""
        await self._enqueue_live(item, weight, key, nowait)
        if self._stores_live_item(key, item):
            self._schedule_storage(item)

    async def record_via_ray(self, item: Any) -> None:
        """Schedule a storage-only record without creating a live queue entry."""
        self._schedule_storage(item)

    async def record(
        self, src_addr: WorkerAddress, transport_key: str | None = None
    ) -> None:
        """Receive a storage-only record through worker P2P transport."""
        item = self.recv(src_addr.root_group_name, src_addr.rank_path)
        self._bootstrap_transport(src_addr, item, transport_key)
        self._schedule_storage(item)

    async def record_frame(self, src_addr: WorkerAddress, transport_key: str) -> None:
        """Receive one fixed tensor frame for storage only."""
        self._schedule_storage(self._receive_frame(src_addr, transport_key))

    async def get(
        self,
        dst_addr: WorkerAddress,
        query_id: int,
        key: Any = DEFAULT_KEY,
        nowait: bool = False,
    ) -> None:
        """Return one record through the standard channel transport."""
        await super().get(dst_addr, query_id, key, nowait)

    async def get_via_ray(self, key: Any = DEFAULT_KEY, nowait: bool = False) -> Any:
        """Return one driver-originated record through Ray transport."""
        return await super().get_via_ray(key, nowait)

    async def _enqueue_live(
        self, item: Any, weight: int, key: Any, nowait: bool
    ) -> None:
        """Publish the live payload before doing any trajectory assembly work."""
        self.create_queue(key, self.maxsize())
        queued_item = WeightedItem(weight=weight, item=item)
        if nowait:
            self._queue_map[key].put_nowait(queued_item)
        else:
            await self._queue_map[key].put(queued_item)

    def _bootstrap_transport(
        self,
        src_addr: WorkerAddress,
        item: Any,
        transport_key: str | None,
    ) -> None:
        if isinstance(item, TrajectoryData):
            if transport_key is None:
                raise ValueError("TrajectoryData bootstrap requires a transport key.")
            self._transport_endpoint(src_addr, transport_key).bootstrap(item)

    def _receive_frame(
        self, src_addr: WorkerAddress, transport_key: str
    ) -> TrajectoryData:
        endpoint = self._transport_endpoint(src_addr, transport_key)
        buffers = endpoint.receive_frame_buffers()
        self.recv_tensor_frame(
            buffers,
            src_addr.root_group_name,
            src_addr.rank_path,
        )
        return endpoint.decode(buffers[0], buffers[1:])

    def _transport_endpoint(
        self, src_addr: WorkerAddress, transport_key: str
    ) -> TransportEndpoint:
        key = f"{src_addr.root_group_name}:{src_addr.rank_path}:{transport_key}"
        endpoint = self._transport_endpoints.get(key)
        if endpoint is None:
            endpoint = TransportEndpoint()
            self._transport_endpoints[key] = endpoint
        return endpoint

    def _schedule_storage(self, item: Any) -> None:
        """Queue storage work without delaying a live channel operation."""
        if not isinstance(item, TrajectoryData):
            if not isinstance(item, TrajectoryEnvelope):
                return
            item = item.record
        if (
            isinstance(item, Observations)
            and self._storage_config is not None
            and item.current_step == self._storage_config.max_steps_per_rollout_epoch
        ):
            # The final live observation is consumed by Rollout to calculate its
            # bootstrap value. Env later records it as EnvBootstrap together
            # with the bootstrap-adjusted final reward.
            return

        if self._storage_task is None or self._storage_task.done():
            self._storage_task = asyncio.create_task(self._run_storage_loop())
        self._storage_queue.put_nowait(item)

    @staticmethod
    def _stores_live_item(key: Any, item: Any) -> bool:
        """Exclude Env-to-Reward requests from trajectory storage writes."""
        return not (
            isinstance(key, tuple)
            and key[0] == "Reward"
            and isinstance(item, Observations)
            and item.reward_inputs is not None
        )

    async def _run_storage_loop(self) -> None:
        """Serialize storage writes while live relay continues independently."""
        while True:
            item = await self._storage_queue.get()
            try:
                await self._ingest(item)
            except Exception as error:
                self.log_error(f"Trajectory storage write failed: {error}")
            finally:
                self._storage_queue.task_done()

    async def _ingest(self, item: TrajectoryData) -> None:
        if self._storage_config is None:
            raise RuntimeError("Trajectory storage has not been configured.")
        if item.slot_ids is None:
            raise ValueError("Trajectory records must carry routed slot_ids.")
        local_slots = self._local_slots(item.slot_ids)
        storage_key = (item.global_step, item.stage_id)
        storage = self._storages.get(storage_key)
        if storage is None:
            storage = TrajectoryStorage.from_config(self._storage_config)
            self._storages[storage_key] = storage
        storage.write(item, local_slots)
        storage.apply_terminal_bootstrap_reward(
            item.current_epoch,
            gamma=storage.gamma,
            auto_reset=storage.auto_reset,
            bootstrap_type=storage.bootstrap_type,
        )
        if not storage.complete():
            return

        for actor_rank, actor_slots in self._actor_slot_indices.items():
            actor_key = (self._actor_channel_name, DEFAULT_KEY, actor_rank)
            self.create_queue(actor_key, self.maxsize())
            trajectory = storage.to_trajectory(actor_slots)
            if self._compression_config.enabled:
                self._schedule_actor_trajectory(actor_key, trajectory)
            else:
                await self._queue_map[actor_key].put(
                    WeightedItem(weight=0, item=trajectory)
                )
        del self._storages[storage_key]

    def _schedule_actor_trajectory(self, key: Any, trajectory: Any) -> None:
        """Encode completed Actor output without blocking storage ingestion."""
        if self._trajectory_encode_task is None or self._trajectory_encode_task.done():
            self._trajectory_encode_task = asyncio.create_task(
                self._run_trajectory_encode_loop()
            )
        self._trajectory_encode_queue.put_nowait((key, trajectory))

    async def _run_trajectory_encode_loop(self) -> None:
        """Serialize compression to preserve Actor queue order per worker."""
        while True:
            key, trajectory = await self._trajectory_encode_queue.get()
            try:
                encoded = await asyncio.to_thread(
                    compress_trajectory, trajectory, self._compression_config
                )
                await self._queue_map[key].put(WeightedItem(weight=0, item=encoded))
            finally:
                self._trajectory_encode_queue.task_done()

    def _local_slots(self, slot_ids: tuple[int, ...]) -> list[int]:
        try:
            return [self._slot_to_local_index[slot_id] for slot_id in slot_ids]
        except KeyError as error:
            raise ValueError(
                f"Slot {error.args[0]} does not belong to this trajectory worker."
            ) from error
