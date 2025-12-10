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
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

import ray
import ray.actor

from ..cluster import Cluster
from ..collective import AsyncChannelCommWork, AsyncChannelWork, AsyncWork
from ..manager import WorkerAddress
from ..placement import NodePlacementStrategy
from ..worker import Worker, WorkerGroup

if TYPE_CHECKING:
    from .channel_worker import LocalChannel

DEFAULT_KEY = "default_queue"


class Channel:
    """A FIFO queue-like channel for inter-worker communication.

    **Creation**: Channel can be created both inside and outside of worker contexts.
    The recommended practice is to create channels outside of worker contexts using `Channel.create()`, and then pass them into workers as needed.
    You can also create channels inside worker contexts or connect to existing channels, using `self.create_channel()` or `self.connect_channel()`.

    **Interface**: Similar as the `asyncio.Queue`, the `Channel` provides interfaces like `put`, `get`, `put_no_wait`, and `get_no_wait`,
    as well as query interfaces like `qsize`, `empty`, and `full`.
    The semantics of these interfaces are identical to those of `asyncio.Queue`.

    **Features**:

    1. **Async operation**: Channel supports both synchronous and asynchronous `put` and `get` operations, similar to Worker's `send` and `recv` APIs. Both operations accept arbitrary data item as long as it's serializable. The default behavior is synchronous, and async operations can be enabled by setting the `async_op` flag. This async can be used not only in asyncio context with `await channel.get(async_op=True).async_wait()`, but also in non-asyncio contexts by generating a communication handle that can be waited later, like async torch.distributed.send().

    2. **Key-based routing**: Channel allows specifying a `key` for each data item, which can be used to identify and route messages. For example, if you wish a specific data to be get and processed by a specific worker, you can assign a unique key to that data item when putting it into the channel. The target worker can then use this key to retrieve the specific data item.This is useful in multi-turn scenarios in agent and embodied RL, where a data is processed by a fixed set of workers.

    3. **Weight and batch processing**: Channel also supports assigning weights to individual data items, allowing for more fine-grained control over how messages are processed. A `get_batch` method can be used to retrieve a batch of messages which respects the assigned weights.

    4. **Debugging**: Channel allows you to print a Channel's internal data by directly print the Channel object.

    Example::

        >>> import sys
        >>> import os
        >>> import asyncio
        >>> import torch
        >>> from rlinf.scheduler import (
        ...     Worker,
        ...     Cluster,
        ...     PackedPlacementStrategy,
        ... )
        >>>
        >>> class Producer(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def produce(self, channel: Channel):
        ...         # Synchronous put of common object
        ...         channel.put("Hello from Producer")
        ...
        ...         # Synchronous put of tensor
        ...         tensor = torch.ones(1, device=torch.cuda.current_device())
        ...         channel.put(tensor)
        ...
        ...         # Asynchronous put of common object
        ...         async_work = channel.put(
        ...             "Hello from Producer asynchronously", async_op=True
        ...         )
        ...         async_work.wait()
        ...
        ...         # Asynchronous put using asyncio
        ...         async_work = channel.put(tensor, async_op=True)
        ...
        ...         async def wait_async():
        ...             await async_work.async_wait()
        ...
        ...         asyncio.run(wait_async())
        ...
        ...         # Put object with weight
        ...         channel.put("Hello with weight", weight=1)
        ...         channel.put(tensor, weight=2)
        >>>
        >>> class Consumer(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def consume(self, channel: Channel):
        ...         tensor = channel.get()
        ...
        ...         async_work = channel.get(async_op=True)
        ...         async_result = async_work.wait()
        ...
        ...         async_work = channel.get(async_op=True)
        ...
        ...         async def wait_async():
        ...             result = await async_work.async_wait()
        ...
        ...         asyncio.run(wait_async())
        ...
        ...         # Get batch of objects based on weight
        ...         batch = channel.get_batch(target_weight=3)
        >>>
        >>> cluster = Cluster(num_nodes=1)
        >>> channel = Channel.create(name="channel")
        >>> placement = PackedPlacementStrategy(
        ...     start_hardware_rank=0, end_hardware_rank=0
        ... )
        >>> producer = Producer.create_group().launch(
        ...     cluster, name="test", placement_strategy=placement
        ... )
        >>> consumer = Consumer.create_group().launch(
        ...     cluster, name="test2", placement_strategy=placement
        ... )
        >>> r1 = producer.produce(channel)
        >>> r2 = consumer.consume(channel)
        >>> res = r1.wait()
        >>> res = r2.wait()

    """

    local_channel_map: dict[int, "LocalChannel"] = {}

    @classmethod
    def create(
        cls, name: str, node_rank: int = 0, maxsize: int = 0, local: bool = False
    ) -> "Channel":
        """Create a new channel with the specified name, node ID, and accelerator ID.

        Args:
            name (str): The name of the channel.
            node_rank (int): The global rank of the node in the cluster where the channel will be created.
            maxsize (int): The maximum size of the channel queue. Defaults to 0 (unbounded).
            local (bool): Create the channel for intra-process communication. A local channel cannot be connected by other workers, and its data cannot be shared among different processes.

        Returns:
            Channel: A new instance of the Channel class.

        """
        from .channel_worker import ChannelWorker, LocalChannel

        cluster = Cluster()
        channel = cls()
        if local:
            local_channel = LocalChannel(maxsize=maxsize)
            channel._initialize(
                name,
                None,
                None,
                Worker.current_worker,
                local_channel=local_channel,
                maxsize=maxsize,
            )
            return channel

        placement = NodePlacementStrategy(node_ranks=[node_rank])
        try:
            channel_worker_group = ChannelWorker.create_group(maxsize=maxsize).launch(
                cluster=cluster,
                name=name,
                placement_strategy=placement,
                # Set max_concurrency to a high value to avoid large number of gets blocking puts
                max_concurrency=2**31 - 1,
            )
        except ValueError:
            Worker.logger.warning(f"Channel {name} already exists, connecting to it.")
            return cls.connect(name, Worker.current_worker)
        channel._initialize(
            name,
            channel_worker_group,
            channel_worker_group.worker_info_list[0].worker,
            Worker.current_worker,
            maxsize=maxsize,
        )
        return channel

    @classmethod
    def connect(cls, channel_name: str, current_worker: Worker) -> "Channel":
        """Connect to an existing channel.

        Args:
            channel_name (str): The name of the channel to connect to.
            current_worker (Worker): The current worker that is connecting to the channel.

        Returns:
            Channel: An instance of the Channel class connected to the specified channel.

        """
        count = 0
        channel_worker_actor = None
        channel_worker_actor_name = WorkerAddress(
            root_group_name=channel_name, ranks=0
        ).get_name()
        while True:
            try:
                channel_worker_actor = ray.get_actor(
                    name=channel_worker_actor_name, namespace=Cluster.NAMESPACE
                )
                break
            except ValueError:
                time.sleep(0.001)
                count += 1
                if count % Cluster.TIMEOUT_WARN_TIME == 0:
                    Worker.logger.warning(
                        f"Waiting for channel {channel_name} to be up for {count // 1000} seconds..."
                    )

        channel = cls()
        maxsize = ray.get(channel_worker_actor.maxsize.remote())
        channel._initialize(
            channel_name,
            None,
            channel_worker_actor,
            current_worker,
            maxsize=maxsize,
        )
        return channel

    def _initialize(
        self,
        channel_name: str,
        channel_worker_group: WorkerGroup,
        channel_worker_actor: ray.actor.ActorHandle,
        current_worker: Worker,
        local_channel: Optional["LocalChannel"] = None,
        maxsize: int = 0,
    ):
        self._channel_name = channel_name
        self._channel_worker_group = channel_worker_group
        self._channel_worker_actor = channel_worker_actor
        self._current_worker = current_worker
        self._local_channel = local_channel
        self._maxsize = maxsize
        if self._local_channel is not None:
            self._local_channel_id = id(self._local_channel)
            Channel.local_channel_map[self._local_channel_id] = self._local_channel
        else:
            self._local_channel_id = None

    @property
    def is_local(self):
        """Check if the channel is a local channel."""
        return self._local_channel is not None

    def qsize(self, key: Any = DEFAULT_KEY) -> int:
        """Get the size of the channel queue.

        Args:
            key (Any): check the queue associated with the key.

        Returns:
            int: The number of items in the channel queue.

        """
        if self._local_channel is not None:
            return self._local_channel.qsize(key)
        return ray.get(self._channel_worker_actor.qsize.remote(key))

    def empty(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is empty.

        Args:
            key (Any): The key to check the queue emptiness for.

        Returns:
            bool: True if the channel queue is empty, False otherwise.

        """
        if self._local_channel is not None:
            return self._local_channel.empty(key)
        return ray.get(self._channel_worker_actor.empty.remote(key))

    def full(self, key: Any = DEFAULT_KEY) -> bool:
        """Check if the channel queue is full.

        Args:
            key (Any): The key to check the queue fullness for.

        Returns:
            bool: True if the channel queue is full, False otherwise.

        """
        if self._local_channel is not None:
            return self._local_channel.full(key)
        return ray.get(self._channel_worker_actor.full.remote(key))

    def put(
        self,
        item: Any,
        weight: int = 0,
        key: Any = DEFAULT_KEY,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Put an item into the channel queue.

        Args:
            item (Any): The item to put into the channel queue.
            weight (int): The priority weight of the item. Defaults to 0.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will put the item in the queue associated with that key.
            If the queue associated with the key does not exist, it will be created.
            async_op (bool): Whether to perform the operation asynchronously.

        """
        if self._local_channel is not None:
            assert async_op is False, "Local channel does not support async put."
            self._local_channel.put(item, weight, key)
            return

        # First run async put to avoid send blocking put
        if self._current_worker is not None:
            # Inside a worker, use send/recv
            put_kwargs = {
                "src_addr": self._current_worker.worker_address,
            }
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="put",
                **put_kwargs,
            )
            self._current_worker.send(
                (key, item, weight), self._channel_name, 0, async_op=True
            )

            if async_op:
                return async_channel_work
            else:
                async_channel_work.wait()
        else:
            # Outside a worker, use ray comm
            put_kwargs = {"item": item, "weight": weight, "key": key}
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="put_via_ray",
                **put_kwargs,
            )
            if async_op:
                return async_channel_work
            else:
                async_channel_work.wait()

    def put_nowait(self, item: Any, weight: int = 0, key: Any = DEFAULT_KEY):
        """Put an item into the channel queue without waiting. Raises asyncio.QueueFull if the queue is full.

        Args:
            item (Any): The item to put into the channel queue.
            weight (int): The priority weight of the item. Defaults to 0.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will put the item in the queue associated with that key.
            If the queue associated with the key does not exist, it will be created.

        Raises:
            asyncio.QueueFull: If the queue is full.

        """
        if self._local_channel is not None:
            self._local_channel.put(item, weight, key, nowait=True)
            return

        if self._current_worker is not None:
            put_kwargs = {
                "src_addr": self._current_worker.worker_address,
                "nowait": True,
            }
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="put",
                **put_kwargs,
            )
            self._current_worker.send(
                (key, item, weight), self._channel_name, 0, async_op=True
            )
            try:
                async_channel_work.wait()
            except asyncio.QueueFull:
                raise asyncio.QueueFull
        else:
            put_kwargs = {"item": item, "weight": weight, "key": key, "nowait": True}
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="put_via_ray",
                **put_kwargs,
            )
            try:
                async_channel_work.wait()
            except asyncio.QueueFull:
                raise asyncio.QueueFull

    def get(self, key: Any = DEFAULT_KEY, async_op: bool = False) -> AsyncWork | Any:
        """Get an item from the channel queue.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Any: The item retrieved from the channel queue.

        """
        if self._local_channel is not None:
            assert async_op is False, "Local channel does not support async get."
            return self._local_channel.get(key)

        if self._current_worker is not None:
            # Inside a worker, use send/recv
            query_id = uuid.uuid4().int
            get_kwargs = {
                "dst_addr": self._current_worker.worker_address,
                "query_id": query_id,
                "key": key,
            }
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get",
                **get_kwargs,
            )
            async_comm_work = self._current_worker.recv(
                self._channel_name, 0, async_op=True
            )
            if async_op:
                return AsyncChannelCommWork(async_comm_work, query_id)
            else:
                async_channel_work.wait()
                # query_id, data
                _, data = async_comm_work.wait()
                return data
        else:
            # Outside a worker, use ray comm
            get_kwargs = {"key": key}
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get_via_ray",
                **get_kwargs,
            )
            if async_op:
                return async_channel_work
            else:
                return async_channel_work.wait()

    def get_nowait(self, key: Any = DEFAULT_KEY) -> Any:
        """Get an item from the channel queue without waiting. Raises asyncio.QueueEmpty if the queue is empty.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.

        Returns:
            Any: The item retrieved from the channel queue.

        Raises:
            asyncio.QueueEmpty: If the queue is empty.

        """
        if self._local_channel is not None:
            return self._local_channel.get(key, nowait=True)

        if self._current_worker is not None:
            query_id = uuid.uuid4().int
            get_kwargs = {
                "dst_addr": self._current_worker.worker_address,
                "query_id": query_id,
                "key": key,
                "nowait": True,
            }
            AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get",
                **get_kwargs,
            )
            query_id, data = self._current_worker.recv(self._channel_name, 0)
            if query_id == asyncio.QueueEmpty:
                raise asyncio.QueueEmpty
            return data
        else:
            get_kwargs = {"key": key, "nowait": True}
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get_via_ray",
                **get_kwargs,
            )
            return async_channel_work.wait()

    def get_batch(
        self,
        target_weight: int = 0,
        key: Any = DEFAULT_KEY,
        async_op: bool = False,
    ) -> AsyncWork | list[Any]:
        """Get a batch of items from the channel queue based on the set batch weight.

        It will try to get items until the total weight of the items is about to (i.e., the next item will) exceed the set batch weight.

        Args:
            target_weight (int): The target weight for the batch.
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            List[Any]: A list of items retrieved from the channel queue.

        """
        if self._local_channel is not None:
            assert async_op is False, "Local channel does not support async get_batch."
            return self._local_channel.get_batch(target_weight, key)

        if self._current_worker is not None:
            query_id = uuid.uuid4().int
            get_kwargs = {
                "dst_addr": self._current_worker.worker_address,
                "query_id": query_id,
                "target_weight": target_weight,
                "key": key,
            }
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get_batch",
                **get_kwargs,
            )
            async_comm_work = self._current_worker.recv(
                self._channel_name, 0, async_op=True
            )
            if async_op:
                return AsyncChannelCommWork(async_comm_work, query_id)
            else:
                async_channel_work.wait()
                # query_id, data
                _, data = async_comm_work.wait()
                return data
        else:
            get_kwargs = {"target_weight": target_weight, "key": key}
            async_channel_work = AsyncChannelWork(
                channel_name=self._channel_name,
                channel_key=key,
                channel_actor=self._channel_worker_actor,
                method="get_batch_via_ray",
                **get_kwargs,
            )
            if async_op:
                return async_channel_work
            else:
                return async_channel_work.wait()

    def __str__(self, key: Any = DEFAULT_KEY) -> str:
        """Get a all the items in the channel queue as a string.

        Args:
            key (Any): The key to get the item from. A unique identifier for a specific set of items.
            When a key is given, the channel will look for the item in the queue associated with that key.
        """
        if self._local_channel is not None:
            return str(self._local_channel.get_all(key))
        async_work = AsyncChannelCommWork(
            self._channel_worker_actor.get_all.remote(key=key)
        )
        items = async_work.wait()
        return str(items)

    def __setstate__(self, state_dict: dict[str, Any]):
        """Set current worker when the channel is unpickled."""
        self.__dict__.update(state_dict)
        # Set local channel
        if self._local_channel_id is not None and self._local_channel is not None:
            local_channel = Channel.local_channel_map.get(
                self._local_channel_id, self._local_channel
            )
            self._local_channel = local_channel
            Channel.local_channel_map[self._local_channel_id] = self._local_channel
        if self._current_worker is None:
            self._current_worker = Worker.current_worker
