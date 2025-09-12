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
import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..cluster import Cluster
from ..worker import Worker, WorkerAddress
from .channel import DEFAULT_QUEUE_NAME


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


class WorkerGPULock(asyncio.Lock):
    """Represents an asyncio lock associated with a worker address."""

    def __init__(self):
        """Initialize the lock."""
        super().__init__()
        self._worker_address = None

    async def acquire(self, worker_address: WorkerAddress):
        """Acquire the lock for the specified worker address.

        Args:
            worker_address (WorkerAddress): The worker address to lock.
        """
        if self._worker_address == worker_address:
            return True
        if not self._locked and (
            self._waiters is None or all(w.cancelled() for w in self._waiters)
        ):
            self._locked = True
            return True

        if self._waiters is None:
            self._waiters = collections.deque()
        fut = self._get_loop().create_future()
        self._waiters.append(fut)

        try:
            try:
                await fut
            finally:
                self._waiters.remove(fut)
        except asyncio.exceptions.CancelledError:
            if not self._locked:
                self._wake_up_first()
            raise

        self._locked = True
        assert self._worker_address is None, (
            f"Lock is already acquired by another worker {self._worker_address.get_name()}."
        )
        self._worker_address = worker_address
        return True

    async def release(self, worker_address: WorkerAddress):
        """Release the lock for the specified worker address.

        Args:
            worker_address (WorkerAddress): The worker address to unlock.
        """
        if self._worker_address is not None and worker_address != self._worker_address:
            raise RuntimeError(
                f"The lock is owned by worker {self._worker_address.get_name()}, but {worker_address.get_name()} is trying to release it."
            )
        if self._locked:
            self._locked = False
            self._worker_address = None
            self._wake_up_first()
        else:
            raise RuntimeError("Lock is not acquired.")


class LocalChannel:
    """A local channel that holds the data in the current process, which cannot be connected by other workers."""

    def __init__(self, maxsize: int = 0):
        """Initialize the LocalChannel with a maximum size for the queue.

        Args:
            maxsize (int): The maximum size of the default channel queue. Defaults to 0 (unbounded).

        """
        self._queue_map: Dict[str, PeekQueue] = {}

        self._queue_map[DEFAULT_QUEUE_NAME] = PeekQueue(maxsize=maxsize)

    def create_queue(self, queue_name: str, maxsize: int = 0):
        """Create a new queue in the channel. No effect if a queue with the same name already exists.

        Args:
            queue_name (str): The name of the queue to create.
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        if queue_name in self._queue_map:
            return
        self._queue_map[queue_name] = PeekQueue(maxsize=maxsize)

    def qsize(self, queue_name: str = DEFAULT_QUEUE_NAME) -> int:
        """Get the size of the channel queue.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].qsize()

    def empty(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is empty.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].empty()

    def full(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is full.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].full()

    def maxsize(self, queue_name: str = DEFAULT_QUEUE_NAME) -> int:
        """Get the maximum size of the channel queue.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].maxsize

    def put(
        self,
        item: Any,
        weight: int,
        queue_name: str = DEFAULT_QUEUE_NAME,
        nowait: bool = False,
    ):
        """Put an item into the channel queue.

        Args:
            item (Any): The item to be put into the queue.
            weight (int): The weight of the item to be put into the queue.
            queue_name (str): The name of the queue to put the item into. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        item = WeightedItem(weight=weight, item=item)
        if nowait:
            self._queue_map[queue_name].put_nowait(item)
        else:
            while self._queue_map[queue_name].full():
                continue
            self._queue_map[queue_name].put_nowait(item)

    def get(
        self,
        queue_name: str = DEFAULT_QUEUE_NAME,
        nowait: bool = False,
    ) -> Any:
        """Get an item from the channel queue.

        Args:
            queue_name (str): The name of the queue to get the item from. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        if nowait:
            weighted_item: WeightedItem = self._queue_map[queue_name].get_nowait()
        else:
            while self._queue_map[queue_name].empty():
                continue
            weighted_item: WeightedItem = self._queue_map[queue_name].get_nowait()
        return weighted_item.item

    async def get_batch(
        self,
        target_weight: int,
        queue_name: str = DEFAULT_QUEUE_NAME,
    ) -> List[Any]:
        """Get a batch of items from the channel queue based on the batch weight.

        Args:
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            queue_name (str): The name of the queue to get the batch from. Defaults to DEFAULT_QUEUE_NAME.

        """
        batch = []
        current_weight = 0
        items: List[WeightedItem] = self._queue_map[queue_name].peek_all()
        for item in items:
            if current_weight + item.weight > target_weight:
                break
            current_weight += item.weight
            item: WeightedItem = self._queue_map[queue_name].get_nowait()
            batch.append(item.item)
            if current_weight >= target_weight:
                break

        return batch

    def get_all(self, queue_name: str = DEFAULT_QUEUE_NAME) -> List[Any]:
        """Get all items from the channel queue without removing them.

        Args:
            queue_name (str): The name of the queue to get the items from. Defaults to DEFAULT_QUEUE_NAME.

        Returns:
            List[Any]: A list of all items in the queue.

        """
        return self._queue_map[queue_name].peek_all()


class ChannelWorker(Worker):
    """The actual worker that holds the channel."""

    def __init__(self, maxsize: int = 0):
        """Initialize the ChannelWorker with a maximum size for the queue.

        Args:
            maxsize (int): The maximum size of the default channel queue. Defaults to 0 (unbounded).

        """
        super().__init__()
        self._queue_map: Dict[str, PeekQueue] = {}
        cluster = Cluster()
        self._gpu_locks = [
            WorkerGPULock()
            for _ in range(cluster.num_nodes * cluster.num_gpus_per_node)
        ]

        self._queue_map[DEFAULT_QUEUE_NAME] = PeekQueue(maxsize=maxsize)

    def create_queue(self, queue_name: str, maxsize: int = 0):
        """Create a new queue in the channel. No effect if a queue with the same name already exists.

        Args:
            queue_name (str): The name of the queue to create.
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        if queue_name in self._queue_map:
            return
        self._queue_map[queue_name] = PeekQueue(maxsize=maxsize)

    def qsize(self, queue_name: str = DEFAULT_QUEUE_NAME) -> int:
        """Get the size of the channel queue.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].qsize()

    def empty(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is empty.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].empty()

    def full(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is full.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].full()

    def maxsize(self, queue_name: str = DEFAULT_QUEUE_NAME) -> int:
        """Get the maximum size of the channel queue.

        Args:
            queue_name (str): The name of the queue to check.

        """
        return self._queue_map[queue_name].maxsize

    async def acquire_gpus(self, worker_address: WorkerAddress, gpu_ids: List[int]):
        """Lock the specified GPU IDs.

        Args:
            worker_address (WorkerAddress): The address of the worker requesting the lock.
            gpu_ids (List[int]): The list of GPU IDs to lock.
        """
        await asyncio.gather(
            *(self._gpu_locks[gpu_id].acquire(worker_address) for gpu_id in gpu_ids)
        )

    async def release_gpus(self, worker_address: WorkerAddress, gpu_ids: List[int]):
        """Release the specified GPU IDs.

        Args:
            worker_address (WorkerAddress): The address of the worker releasing the lock.
            gpu_ids (List[int]): The list of GPU IDs to release.
        """
        await asyncio.gather(
            *(self._gpu_locks[gpu_id].release(worker_address) for gpu_id in gpu_ids)
        )

    async def put(
        self,
        src_addr: WorkerAddress,
        weight: int,
        queue_name: str = DEFAULT_QUEUE_NAME,
        nowait: bool = False,
    ):
        """Put an item into the channel queue.

        Args:
            src_addr (WorkerAddress): The address of the source worker.
            weight (int): The weight of the item to be put into the queue.
            queue_name (str): The name of the queue to put the item into. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        item = self.recv(src_addr.root_group_name, src_addr.rank_path)
        item = WeightedItem(weight=weight, item=item)
        if nowait:
            self._queue_map[queue_name].put_nowait(item)
        else:
            await self._queue_map[queue_name].put(item)

    async def put_via_ray(
        self,
        item: Any,
        weight: int,
        queue_name: str = DEFAULT_QUEUE_NAME,
        nowait: bool = False,
    ):
        """Put an item into the channel queue via Ray's communication. Useful when there is no worker.

        Args:
            item (Any): The item to be put into the queue.
            weight (int): The weight of the item to be put into the queue.
            queue_name (str): The name of the queue to put the item into. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueFull if the queue is full. Defaults to False.

        """
        weighted_item = WeightedItem(weight=weight, item=item)
        if nowait:
            self._queue_map[queue_name].put_nowait(weighted_item)
        else:
            await self._queue_map[queue_name].put(weighted_item)

    async def get(
        self,
        dst_addr: WorkerAddress,
        queue_name: str = DEFAULT_QUEUE_NAME,
        nowait: bool = False,
    ) -> Any:
        """Get an item from the channel queue.

        Args:
            dst_addr (WorkerAddress): The address of the destination worker.
            queue_name (str): The name of the queue to get the item from. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        if nowait:
            weighted_item: WeightedItem = self._queue_map[queue_name].get_nowait()
        else:
            weighted_item: WeightedItem = await self._queue_map[queue_name].get()
        self.send(
            weighted_item.item,
            dst_addr.root_group_name,
            dst_addr.rank_path,
            async_op=True,
        )

    async def get_via_ray(
        self, queue_name: str = DEFAULT_QUEUE_NAME, nowait: bool = False
    ) -> Any:
        """Get an item from the channel queue via Ray's communication. Useful when there is no worker.

        Args:
            queue_name (str): The name of the queue to get the item from. Defaults to DEFAULT_QUEUE_NAME.
            nowait (bool): If True, directly raise asyncio.QueueEmpty if the queue is empty. Defaults to False.

        """
        if nowait:
            weighted_item: WeightedItem = self._queue_map[queue_name].get_nowait()
        else:
            weighted_item: WeightedItem = await self._queue_map[queue_name].get()
        return weighted_item.item

    async def get_batch(
        self,
        dst_addr: WorkerAddress,
        target_weight: int,
        queue_name: str = DEFAULT_QUEUE_NAME,
    ) -> List[Any]:
        """Get a batch of items from the channel queue based on the batch weight.

        Args:
            dst_addr (WorkerAddress): The address of the destination worker.
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            queue_name (str): The name of the queue to get the batch from. Defaults to DEFAULT_QUEUE_NAME.

        """
        batch = []
        current_weight = 0
        while True:
            next_item: WeightedItem = await self._queue_map[queue_name].peek()
            if next_item is None or current_weight + next_item.weight > target_weight:
                break
            current_weight += next_item.weight
            item = await self._queue_map[queue_name].get()
            batch.append(item.item)
            if current_weight >= target_weight:
                break

        self.send(batch, dst_addr.root_group_name, dst_addr.rank_path, async_op=True)

    async def get_batch_via_ray(
        self, target_weight: int, queue_name: str = DEFAULT_QUEUE_NAME
    ) -> List[Any]:
        """Get a batch of items from the channel queue via Ray's communication based on the batch weight.

        Args:
            target_weight (int): The target weight for the batch. The batch will contain items until the total weight reaches this value.
            queue_name (str): The name of the queue to get the batch from. Defaults to DEFAULT_QUEUE_NAME.

        """
        batch = []
        current_weight = 0
        while True:
            next_item: WeightedItem = await self._queue_map[queue_name].peek()
            if next_item is None or current_weight + next_item.weight > target_weight:
                break
            current_weight += next_item.weight
            item = await self._queue_map[queue_name].get()
            batch.append(item.item)
            if current_weight >= target_weight:
                break
        return batch

    def get_all(self, queue_name: str = DEFAULT_QUEUE_NAME) -> List[Any]:
        """Get all items from the channel queue without removing them.

        Args:
            queue_name (str): The name of the queue to get the items from. Defaults to DEFAULT_QUEUE_NAME.

        Returns:
            List[Any]: A list of all items in the queue.

        """
        return self._queue_map[queue_name].peek_all()
