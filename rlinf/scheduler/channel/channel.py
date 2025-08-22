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

import time
from typing import Any, Dict, List, Optional

import ray
import ray.actor

from ..cluster import Cluster
from ..collective import AsyncChannelWork, AsyncWork
from ..manager import WorkerAddress, WorkerInfo, WorkerManager
from ..placement import PackedPlacementStrategy
from ..worker import Worker, WorkerGroup

DEFAULT_QUEUE_NAME = "default_queue"


class Channel:
    """A load balancing channel for inter-worker communication.

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
        >>> class TestWorker(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def hello(self):
        ...         # Synchronous put of common object
        ...         channel = self.create_channel(
        ...             channel_name="test_channel", maxsize=10
        ...         )
        ...         channel.put("Hello from TestWorker")
        ...
        ...         # Synchronous put of tensor
        ...         tensor = torch.ones(1, device=torch.cuda.current_device())
        ...         channel.put(tensor)
        ...
        ...         # Asynchronous put of common object
        ...         async_work = channel.put(
        ...             "Hello from TestWorker asynchronously", async_op=True
        ...         )
        ...         async_work.wait()
        ...
        ...         # Asynchronous put of tensor using asyncio
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
        >>> class TestWorker2(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def hello(self):
        ...         channel = self.connect_channel(
        ...             channel_name="test_channel",
        ...         )
        ...
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
        >>> cluster = Cluster(num_nodes=1, num_gpus_per_node=8)
        >>> placement1 = PackedPlacementStrategy(
        ...     start_gpu_id=0, end_gpu_id=0
        ... )
        >>> worker_group1 = TestWorker.create_group().launch(
        ...     cluster, name="test", placement_strategy=placement1
        ... )
        >>> placement2 = PackedPlacementStrategy(
        ...     start_gpu_id=1, end_gpu_id=1
        ... )
        >>> worker_group2 = TestWorker2.create_group().launch(
        ...     cluster, name="test2", placement_strategy=placement2
        ... )
        >>> r1 = worker_group1.hello()
        >>> r2 = worker_group2.hello()
        >>> res = r1.wait()
        >>> res = r2.wait()

    """

    @classmethod
    def create(cls, name: str, gpu_id: int = 0, maxsize: int = 0) -> "Channel":
        """Create a new channel with the specified name, node ID, and GPU ID.

        Args:
            name (str): The name of the channel.
            gpu_id (int): The global ID of the GPU in the cluster where the channel will be created.
            maxsize (int): The maximum size of the channel queue. Defaults to 0 (unbounded).

        Returns:
            Channel: A new instance of the Channel class.

        """
        cluster = Cluster()
        placement = PackedPlacementStrategy(
            start_gpu_id=gpu_id,
            end_gpu_id=gpu_id,
        )
        try:
            from .channel_worker import ChannelWorker

            channel_worker_group = ChannelWorker.create_group(maxsize=maxsize).launch(
                cluster=cluster, name=name, placement_strategy=placement
            )
        except ValueError:
            raise ValueError(f"Channel named {name} already exists!")
        channel = cls()
        channel._initialize(
            name,
            channel_worker_group,
            channel_worker_group.worker_info_list[0].worker,
            Worker.current_worker,
            maxsize=maxsize,
        )
        return channel

    @classmethod
    def _create_in_worker(
        cls,
        current_worker: Worker,
        channel_name: str,
        group_name_affinity: Optional[str] = None,
        group_rank_affinity: Optional[int | List[int]] = None,
        maxsize: int = 0,
    ) -> "Channel":
        """Create a new channel with the specified placement rank and maximum size.

        Args:
            channel_name (str): The name of the channel.
            current_worker (Worker): The current worker that is creating the channel.
            group_name_affinity (str | None): The name of the group you wish to place the channel. Defaults is the current group.
            group_name_affinity (str): The name of the group you wish to place the channel data. Defaults is the current group.
            group_rank_affinity (int | List[int]): The rank of the group you wish to place the channel data. Default is the current worker's rank.
            maxsize (int): The maximum size of the channel queue. Defaults to 0 (unbounded).

        Returns:
            Channel: A new instance of the Channel class.

        """
        if group_name_affinity is None:
            group_name_affinity = current_worker.worker_address.root_group_name
        if group_rank_affinity is None:
            group_rank_affinity = current_worker.worker_address.rank_path

        affine_worker_address = WorkerAddress(
            root_group_name=group_name_affinity, ranks=group_rank_affinity
        )
        worker_manager = WorkerManager.get_proxy()
        affine_worker_info: WorkerInfo = None
        count = 0
        while affine_worker_info is None:
            affine_worker_info = worker_manager.get_worker_info(affine_worker_address)
            time.sleep(0.001)
            count += 1
            if count % Cluster.TIMEOUT_WARN_TIME == 0:
                Worker.logger.warning(
                    f"Waiting for {affine_worker_address} to be up for {count // 1000} seconds..."
                )
        master_node = affine_worker_info.node_id
        master_gpu = affine_worker_info.gpu_id

        cluster = Cluster()
        global_gpu_id = master_node * cluster.num_gpus_per_node + master_gpu
        placement = PackedPlacementStrategy(
            start_gpu_id=global_gpu_id,
            end_gpu_id=global_gpu_id,
        )
        try:
            from .channel_worker import ChannelWorker

            channel_worker_group = ChannelWorker.create_group(maxsize=maxsize).launch(
                cluster=cluster, name=channel_name, placement_strategy=placement
            )
        except ValueError:
            current_worker._logger.warning("Channel already exists, connecting to it.")
            return cls.connect(channel_name, current_worker)
        channel = cls()
        channel._initialize(
            channel_name,
            channel_worker_group,
            channel_worker_group.worker_info_list[0].worker,
            current_worker,
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
        maxsize: int = 0,
    ):
        self._channel_name = channel_name
        self._channel_worker_group = channel_worker_group
        self._channel_worker_actor = channel_worker_actor
        self._current_worker = current_worker
        self._maxsize = maxsize

    def create_queue(self, queue_name: str, maxsize: int = 0):
        """Create a new queue in the channel. No effect if a queue with the same name already exists.

        Args:
            queue_name (str): The name of the queue to create.
            maxsize (int): The maximum size of the queue. Defaults to 0 (unbounded).

        """
        return ray.get(
            self._channel_worker_actor.create_queue.remote(queue_name, maxsize)
        )

    def qsize(self, queue_name: str = DEFAULT_QUEUE_NAME) -> int:
        """Get the size of the channel queue.

        Args:
            queue_name (str): The name of the queue to check.

        Returns:
            int: The number of items in the channel queue.

        """
        return ray.get(self._channel_worker_actor.qsize.remote(queue_name))

    def empty(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is empty.

        Args:
            queue_name (str): The name of the queue to check.

        Returns:
            bool: True if the channel queue is empty, False otherwise.

        """
        return ray.get(self._channel_worker_actor.empty.remote(queue_name))

    def full(self, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
        """Check if the channel queue is full.

        Args:
            queue_name (str): The name of the queue to check.

        Returns:
            bool: True if the channel queue is full, False otherwise.

        """
        return ray.get(self._channel_worker_actor.full.remote(queue_name))

    def put(
        self,
        item: Any,
        weight: int = 0,
        queue_name: str = DEFAULT_QUEUE_NAME,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Put an item into the channel queue.

        Args:
            item (Any): The item to put into the channel queue.
            weight (int): The priority weight of the item. Defaults to 0.
            queue_name (str): The name of the queue to put the item into.
            async_op (bool): Whether to perform the operation asynchronously.

        """
        # First run async put to avoid send blocking put
        if self._current_worker is not None:
            put_work = AsyncChannelWork(
                self._channel_worker_actor.put.remote(
                    src_addr=self._current_worker.worker_address,
                    weight=weight,
                    queue_name=queue_name,
                )
            )
            self._current_worker.send(item, self._channel_name, 0, async_op=True)

            if async_op:
                return put_work
            else:
                put_work.wait()
        else:
            put_work = AsyncChannelWork(
                self._channel_worker_actor.put_via_ray.remote(
                    item=item, weight=weight, queue_name=queue_name
                )
            )
            if async_op:
                return put_work
            else:
                put_work.wait()

    def put_nowait(
        self, item: Any, weight: int = 0, queue_name: str = DEFAULT_QUEUE_NAME
    ):
        """Put an item into the channel queue without waiting. Raises asyncio.QueueFull if the queue is full.

        Args:
            item (Any): The item to put into the channel queue.
            weight (int): The priority weight of the item. Defaults to 0.
            queue_name (str): The name of the queue to put the item into.

        Raises:
            asyncio.QueueFull: If the queue is full.

        """
        if self._current_worker is not None:
            put_work = AsyncChannelWork(
                self._channel_worker_actor.put.remote(
                    src_addr=self._current_worker.worker_address,
                    weight=weight,
                    queue_name=queue_name,
                    nowait=True,
                )
            )
            self._current_worker.send(item, self._channel_name, 0, async_op=True)
            put_work.wait()
        else:
            put_work = AsyncChannelWork(
                self._channel_worker_actor.put_via_ray.remote(
                    item=item, weight=weight, queue_name=queue_name, nowait=True
                )
            )
            put_work.wait()

    def get(
        self, queue_name: str = DEFAULT_QUEUE_NAME, async_op: bool = False
    ) -> AsyncWork | Any:
        """Get an item from the channel queue.

        Args:
            queue_name (str): The name of the queue to get the item from.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Any: The item retrieved from the channel queue.

        """
        if self._current_worker is not None:
            self._channel_worker_actor.get.remote(
                self._current_worker.worker_address, queue_name=queue_name
            )
            return self._current_worker.recv(self._channel_name, 0, async_op=async_op)
        else:
            async_work = AsyncChannelWork(
                self._channel_worker_actor.get_via_ray.remote(queue_name=queue_name)
            )
            if async_op:
                return async_work
            else:
                return async_work.wait()

    def get_nowait(self, queue_name: str = DEFAULT_QUEUE_NAME) -> Any:
        """Get an item from the channel queue without waiting. Raises asyncio.QueueEmpty if the queue is empty.

        Args:
            queue_name (str): The name of the queue to get the item from.

        Returns:
            Any: The item retrieved from the channel queue.

        Raises:
            asyncio.QueueEmpty: If the queue is empty.

        """
        if self._current_worker is not None:
            self._channel_worker_actor.get.remote(
                self._current_worker.worker_address, queue_name=queue_name, nowait=True
            )
            return self._current_worker.recv(self._channel_name)
        else:
            async_work = AsyncChannelWork(
                self._channel_worker_actor.get_via_ray.remote(
                    queue_name=queue_name, nowait=True
                )
            )
            return async_work.wait()

    def get_batch(
        self,
        target_weight: int = 0,
        queue_name: str = DEFAULT_QUEUE_NAME,
        async_op: bool = False,
    ) -> AsyncWork | List[Any]:
        """Get a batch of items from the channel queue based on the set batch weight.

        It will try to get items until the total weight of the items is about to (i.e., the next item will) exceed the set batch weight.

        Args:
            target_weight (int): The target weight for the batch.
            queue_name (str): The name of the queue to get the batch from.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            List[Any]: A list of items retrieved from the channel queue.

        """
        if self._current_worker is not None:
            self._channel_worker_actor.get_batch.remote(
                self._current_worker.worker_address, target_weight, queue_name
            )
            return self._current_worker.recv(self._channel_name, 0, async_op=async_op)
        else:
            async_work = AsyncChannelWork(
                self._channel_worker_actor.get_batch_via_ray.remote(
                    target_weight=target_weight, queue_name=queue_name
                )
            )
            if async_op:
                return async_work
            else:
                return async_work.wait()

    def __str__(self, queue_name: str = DEFAULT_QUEUE_NAME) -> str:
        """Get a all the items in the channel queue as a string."""
        async_work = AsyncChannelWork(
            self._channel_worker_actor.get_all.remote(queue_name=queue_name)
        )
        items = async_work.wait()
        return str(items)

    def __setstate__(self, state_dict: Dict[str, Any]):
        """Set current worker when the channel is unpickled."""
        self.__dict__.update(state_dict)
        if self._current_worker is None:
            self._current_worker = Worker.current_worker
