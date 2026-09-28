# Copyright 2025 The RLinf Authors.
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
import gc
from dataclasses import dataclass, field
from typing import Any

from ..worker import Worker, WorkerAddress
from .channel import DEFAULT_KEY
from .hooks import (
    ChannelContext,
    iter_collected,
    resolve_collector,
    resolve_dispatcher,
)


@dataclass(order=True)
class WeightedItem:
    """A class that holds an item with a weight for priority queueing."""

    weight: int
    item: Any = field(compare=False)


class PeekQueue(asyncio.Queue):
    """A queue that allows peeking at the next item without removing it."""

    def __init__(self, maxsize=0):
        """Initialize the PeekQueue.

        Args:
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        super().__init__(maxsize)

    async def peek(self):
        """Peek at the next item in the queue without removing it."""
        while self.empty():
            getter = self._get_loop().create_future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # Just in case getter is not done yet.
                try:
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                except ValueError:
                    # The getter could be removed from self._getters by a
                    # previous put_nowait call.
                    pass
                if not self.empty() and not getter.cancelled():
                    # We were woken up by put_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._getters)
                raise
        item = self._queue[0]
        return item

    def peek_all(self):
        """Peek at all items in the queue without removing them."""
        return list(self._queue)


class LocalChannel:
    """A local channel that holds the data in the current process, which cannot be connected by other workers."""

    def __init__(self, maxsize: int = 0):
        """Initialize the LocalChannel with a maximum size for the queue.

        Args:
            maxsize (int): The maximum size of the default channel queue. Defaults to 0 (unbounded).

        """
        self._queue_map: dict[str, PeekQueue] = {}

        self._queue_map[DEFAULT_KEY] = PeekQueue(maxsize=maxsize)

    def create_queue(self, key: Any, maxsize: int = 0):
        """Create a new queue in the channel. No effect if a queue with the same name already exists.

        Args:
            key (Any): The key of the queue to create.
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        if key in self._queue_map:
            return
        self._queue_map[key] = PeekQueue(maxsize=maxsize)

    def qsize(self, key: Any = DEFAULT_KEY) -> int:
        """Get the size of the channel queue.

        Args:
            key (Any): The key of the queue to check.

        """
        if key not in self._queue_map:
            return 0
        return self._queue_map[key].qsize()

    def empty(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is empty.

        Args:
            key (Any): The key of the queue to check.

        """
        if key not in self._queue_map:
            return True
        return self._queue_map[key].empty()

    def full(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is full.

        Args:
            key (Any): The key of the queue to check.

        """
        if key not in self._queue_map:
            return False
        return self._queue_map[key].full()

    def maxsize(self, key: Any = DEFAULT_KEY) -> int:
        """Get the maximum size of the channel queue.

        Args:
            key (Any): The key of the queue to check.

        """
        if key not in self._queue_map:
            return self._queue_map[DEFAULT_KEY].maxsize
        return self._queue_map[key].maxsize

    def put(
        self,
        item: Any,
        weight: int,
        key: Any = DEFAULT_KEY,
        nowait: bool = False,
    ):
        """Put an item into the channel queue.

        Args:
            item (Any): The item to be put into the queue.
            weight (int): The weight of the item to be put into the queue.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        self.create_queue(key, maxsize=self.maxsize())
        item = WeightedItem(weight=weight, item=item)
        if nowait:
            self._queue_map[key].put_nowait(item)
        else:
            while self._queue_map[key].full():
                continue
            self._queue_map[key].put_nowait(item)

    def get(
        self,
        key: Any = DEFAULT_KEY,
        nowait: bool = False,
    ) -> Any:
        """Get an item from the channel queue.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        self.create_queue(key, maxsize=self.maxsize())
        if nowait:
            weighted_item: WeightedItem = self._queue_map[key].get_nowait()
        else:
            while self._queue_map[key].empty():
                continue
            weighted_item: WeightedItem = self._queue_map[key].get_nowait()
        return weighted_item.item

    async def get_batch(
        self,
        target_weight: int,
        key: Any = DEFAULT_KEY,
    ) -> list[Any]:
        """Get a batch of items from the channel queue based on the batch weight.

        Args:
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.

        """
        self.create_queue(key, maxsize=self.maxsize())
        batch = []
        current_weight = 0
        items: list[WeightedItem] = self._queue_map[key].peek_all()
        for item in items:
            if current_weight + item.weight > target_weight:
                break
            current_weight += item.weight
            item: WeightedItem = self._queue_map[key].get_nowait()
            batch.append(item.item)
            if current_weight >= target_weight:
                break

        return batch

    def peek_all(self, key: str = DEFAULT_KEY) -> list[Any]:
        """Get all items from the channel queue without removing them.

        Args:
            key (str): The key to get the items from. A unique identifier for a specific set of items.

        Returns:
            List[Any]: A list of all items in the queue.

        """
        self.create_queue(key, maxsize=self.maxsize())
        return self._queue_map[key].peek_all()


class ChannelWorker(Worker):
    """The actual worker that holds the channel."""

    MEM_CLEAN_THRESHOLD = 0.4
    MEM_CLEAN_PERIOD_SECONDS = 5

    def __init__(
        self,
        maxsize: int = 0,
        collector: Any = None,
        dispatcher: Any = None,
        context: ChannelContext | None = None,
    ):
        """Initialize the ChannelWorker with a maximum size for the queue.

        Args:
            maxsize (int): The maximum size of the default channel queue. Defaults to 0 (unbounded).
            collector (Any): Collector spec transforming items on the way in. See
                :func:`~rlinf.scheduler.channel.hooks.resolve_collector`. Defaults to pass-through.
            dispatcher (Any): Dispatcher spec choosing each item's consumer. See
                :func:`~rlinf.scheduler.channel.hooks.resolve_dispatcher`. Defaults to one shared queue per key.
            context (ChannelContext): Description of the channel handed to both hooks' ``setup``.

        """
        super().__init__()
        self._queue_map: dict[str, PeekQueue] = {}
        self._queue_map[DEFAULT_KEY] = PeekQueue(maxsize=maxsize)
        self._key_to_channel_rank: dict[Any, int] = {}

        self._context = context or ChannelContext(name=self.worker_address.get_name())
        self._collector = resolve_collector(collector)
        self._dispatcher = resolve_dispatcher(dispatcher)
        self._collector.setup(self._context)
        self._dispatcher.setup(self._context)
        # Private per-consumer queues, only used when the dispatcher deals.
        self._deals = self._dispatcher.deals
        self._consumer_queues: dict[str, dict[str, PeekQueue]] = {}

        self._mem_cleaner_task = asyncio.create_task(self._mem_cleaner())

    def _consumer_queue(self, key: Any, consumer: str) -> PeekQueue:
        """Return the private queue holding ``key`` items dealt to ``consumer``."""
        queues = self._consumer_queues.setdefault(key, {})
        queue = queues.get(consumer)
        if queue is None:
            queue = PeekQueue(maxsize=self.maxsize(key))
            queues[consumer] = queue
        return queue

    def _queue_for_get(self, key: Any, consumer: str) -> PeekQueue:
        """Return the queue a consumer should read ``key`` from."""
        if not self._deals:
            self.create_queue(key, self.maxsize())
            return self._queue_map[key]
        observe = getattr(self._dispatcher, "observe", None)
        if observe is not None:
            observe(consumer)
        return self._consumer_queue(key, consumer)

    async def _enqueue(self, key: Any, item: Any, weight: int, nowait: bool = False):
        """Route one collected item to its queue."""
        consumer = self._dispatcher.route(item, key) if self._deals else None
        weighted_item = WeightedItem(weight=weight, item=item)
        if consumer is None:
            self.create_queue(key, self.maxsize())
            queue = self._queue_map[key]
        else:
            queue = self._consumer_queue(key, consumer)
        if nowait:
            queue.put_nowait(weighted_item)
        else:
            await queue.put(weighted_item)

    def _unrouted_queue(self, key: Any) -> PeekQueue | None:
        """Return the shared queue populated before consumer discovery."""
        queue = self._queue_map.get(key)
        return queue if queue is not None and not queue.empty() else None

    async def _take(self, key: Any, consumer: str, nowait: bool) -> WeightedItem:
        """Take one item for ``consumer``, stealing from a peer if allowed."""
        queue = self._queue_for_get(key, consumer)
        if self._deals and queue.empty():
            unrouted = self._unrouted_queue(key)
            if unrouted is not None:
                return unrouted.get_nowait()
        if nowait:
            return queue.get_nowait()
        if self._deals and queue.empty():
            depths = {
                name: peer.qsize()
                for name, peer in self._consumer_queues.get(key, {}).items()
            }
            donor = self._dispatcher.rebalance(key, consumer, depths)
            if donor is not None and donor in self._consumer_queues.get(key, {}):
                return self._consumer_queues[key][donor].get_nowait()
        return await queue.get()

    async def _mem_cleaner(self):
        """A background task that cleans up memory when triggered."""
        mem_util_after_clean = 1.0
        current_mem_util = 1.0
        mem_clean_threshold = ChannelWorker.MEM_CLEAN_THRESHOLD
        while True:
            await asyncio.sleep(ChannelWorker.MEM_CLEAN_PERIOD_SECONDS)
            if self.has_accelerator and Worker.torch_platform.is_initialized():
                memory_reserved = Worker.torch_platform.memory_reserved()
                memory_allocated = Worker.torch_platform.memory_allocated()
                current_mem_util = (
                    memory_allocated / memory_reserved if memory_reserved > 0 else 1.0
                )
                if current_mem_util < mem_clean_threshold:
                    gc.collect()
                    Worker.torch_platform.synchronize()
                    Worker.torch_platform.empty_cache()
                    memory_reserved = Worker.torch_platform.memory_reserved()
                    memory_allocated = Worker.torch_platform.memory_allocated()
                    mem_util_after_clean = (
                        memory_allocated / memory_reserved
                        if memory_reserved > 0
                        else 1.0
                    )
                    if mem_util_after_clean < mem_clean_threshold:
                        mem_clean_threshold = mem_util_after_clean
                        self.log_debug(
                            f"ChannelWorker memory cleaned but still below threshold. Updated MEM_CLEAN_THRESHOLD to {mem_clean_threshold:.2f}"
                        )
                    else:
                        mem_clean_threshold = ChannelWorker.MEM_CLEAN_THRESHOLD

                    self.log_debug(
                        f"ChannelWorker memory after cleanup {Worker.torch_platform.memory_allocated()}, {Worker.torch_platform.memory_reserved()}"
                    )

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current device memory usage of the ChannelWorker.

        Returns:
            Tuple[int, int]: A tuple containing the allocated and reserved memory in bytes.

        """
        if self.has_accelerator and Worker.torch_platform.is_initialized():
            allocated = Worker.torch_platform.memory_allocated()
            reserved = Worker.torch_platform.memory_reserved()
            return allocated, reserved
        return 0, 0

    def create_queue(self, key: Any, maxsize: int = 0):
        """Create a new queue in the channel. No effect if a queue with the same name already exists.

        Args:
            key (Any): The key of the queue to create.
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        if key in self._queue_map:
            return
        self._queue_map[key] = PeekQueue(maxsize=maxsize)

    def qsize(self, key: Any = DEFAULT_KEY, consumer: str | None = None) -> int:
        """Get the size of the channel queue.

        Args:
            key (Any): The key to check the queue size for.
            consumer (str): Report only this consumer's share when the dispatcher
                deals. Defaults to the total outstanding across all consumers.

        """
        if consumer is not None:
            return self._consumer_queues.get(key, {}).get(consumer, PeekQueue()).qsize()
        total = self._queue_map[key].qsize() if key in self._queue_map else 0
        for queue in self._consumer_queues.get(key, {}).values():
            total += queue.qsize()
        return total

    def empty(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is empty.

        Args:
            key (Any): The key to check the queue emptiness for.

        """
        return self.qsize(key) == 0

    def full(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is full.

        Args:
            key (Any): The key to check the queue fullness for.

        """
        if key not in self._queue_map:
            return False
        return self._queue_map[key].full()

    def maxsize(self, key: Any = DEFAULT_KEY) -> int:
        """Get the maximum size of the channel queue.

        Args:
            key (Any): The key to check the maximum size for.

        """
        if key not in self._queue_map:
            return self._queue_map[DEFAULT_KEY].maxsize
        return self._queue_map[key].maxsize

    async def put(
        self,
        src_addr: WorkerAddress,
        nowait: bool = False,
    ):
        """Put an item into the channel queue.

        Args:
            src_addr (WorkerAddress): The address of the source worker.
            When a key is given, the channel will put the item in the queue associated with that key.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        item, (key, weight) = self.recv(src_addr.root_group_name, src_addr.rank_path)
        for out_key, out_item in iter_collected(self._collector, item, key):
            await self._enqueue(out_key, out_item, weight, nowait=nowait)

    async def put_via_ray(
        self,
        item: Any,
        weight: int,
        key: Any = DEFAULT_KEY,
        nowait: bool = False,
    ):
        """Put an item into the channel queue via Ray's communication. Useful when there is no worker.

        Args:
            item (Any): The item to be put into the queue.
            weight (int): The weight of the item to be put into the queue.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will put the item in the queue associated with that key.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        for out_key, out_item in iter_collected(self._collector, item, key):
            await self._enqueue(out_key, out_item, weight, nowait=nowait)

    async def get(
        self,
        dst_addr: WorkerAddress,
        query_id: int,
        key: Any = DEFAULT_KEY,
        nowait: bool = False,
    ) -> Any:
        """Get an item from the channel queue.

        Args:
            dst_addr (WorkerAddress): The address of the destination worker.
            query_id (int): The ID of this get query.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        consumer = dst_addr.get_name()
        if nowait:
            try:
                weighted_item: WeightedItem = await self._take(key, consumer, True)
            except asyncio.QueueEmpty:
                query_id = asyncio.QueueEmpty
                weighted_item = WeightedItem(weight=0, item=None)
        else:
            weighted_item: WeightedItem = await self._take(key, consumer, False)
        self.send(
            weighted_item.item,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )

    async def get_via_ray(self, key: Any = DEFAULT_KEY, nowait: bool = False) -> Any:
        """Get an item from the channel queue via Ray's communication. Useful when there is no worker.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        self.create_queue(key, self.maxsize())
        if nowait:
            weighted_item: WeightedItem = self._queue_map[key].get_nowait()
        else:
            weighted_item: WeightedItem = await self._queue_map[key].get()
        return weighted_item.item

    async def get_batch(
        self,
        dst_addr: WorkerAddress,
        query_id: int,
        target_weight: int,
        key: str = DEFAULT_KEY,
    ) -> list[Any]:
        """Get a batch of items from the channel queue based on the batch weight.

        Args:
            dst_addr (WorkerAddress): The address of the destination worker.
            query_id (int): The ID of this get query.
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.

        """
        queue = self._queue_for_get(key, dst_addr.get_name())
        batch = []
        current_weight = 0
        while True:
            next_item: WeightedItem = await queue.peek()
            if next_item is None or current_weight + next_item.weight > target_weight:
                break
            current_weight += next_item.weight
            item = await queue.get()
            batch.append(item.item)
            if current_weight >= target_weight:
                break

        self.send(
            batch,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            async_op=True,
            piggyback_payload=query_id,
        )

    async def get_batch_via_ray(
        self, target_weight: int, key: Any = DEFAULT_KEY
    ) -> list[Any]:
        """Get a batch of items from the channel queue via Ray's communication based on the batch weight.

        Args:
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.

        """
        self.create_queue(key, self.maxsize())
        batch = []
        current_weight = 0
        while True:
            next_item: WeightedItem = await self._queue_map[key].peek()
            if next_item is None or current_weight + next_item.weight > target_weight:
                break
            current_weight += next_item.weight
            item = await self._queue_map[key].get()
            batch.append(item.item)
            if current_weight >= target_weight:
                break
        return batch

    def peek_all(self, key: Any = DEFAULT_KEY) -> list[Any]:
        """Get all items from the channel queue without removing them.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.

        Returns:
            List[Any]: A list of all items in the queue.

        """
        self.create_queue(key, self.maxsize())
        return self._queue_map[key].peek_all()

    async def ensure_key_replica(self, key: Any, src_node_rank: int = -1) -> int:
        """Assign (or fetch) the replica rank that should host the given key.

        If the key is new, choose the replica whose rank matches the source node rank
        (given NodePlacementStrategy launches workers in node order). If out of range,
        fall back to rank 0.
        """
        # Fallback to rank 0 if out of range
        default_rank = src_node_rank if 0 <= src_node_rank < self._world_size else 0
        return self._key_to_channel_rank.setdefault(key, default_rank)
