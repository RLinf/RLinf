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
import queue
import threading
import time
from typing import Any, Callable, Optional, overload

import ray.actor
import torch.distributed as dist
from torch.futures import Future

from ..worker import Worker


class AsyncWork:
    """Base class for asynchronous work."""

    @overload
    async def async_wait(self) -> Any:
        raise NotImplementedError("AsyncWork must implement async_wait method")

    @overload
    def wait(self) -> Any:
        raise NotImplementedError("AsyncWork must implement wait method")

    @overload
    def then(self, func: Callable, *args, **kwargs) -> "AsyncFuncWork":
        raise NotImplementedError("AsyncWork must implement wait method")

    @overload
    def done(self):
        raise NotImplementedError("AsyncWork must implement done method")

    @overload
    def get_next_work(self) -> "Optional[AsyncWork]":
        raise NotImplementedError("AsyncWork must implement get_next_work method")

    def get_last_work(self) -> "AsyncWork":
        """Get the last AsyncWork chained to this one."""
        cur_work = self
        while True:
            next_work = cur_work.get_next_work()
            if next_work is None:
                return cur_work
            cur_work = next_work


class AsyncFuncWork(AsyncWork):
    """Async work class for chaining callback function."""

    def __init__(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        """Initialize the AsyncFuncWork with a function and its arguments.

        Args:
            func (Callable): The function to call when the work is completed.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        """
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._done = Future()
        self._result = None
        self._next_work = None
        self._cuda_event = None

    def __call__(self, future: Future):
        """Execute the function and set the done flag."""
        self._result = self._func(*self._args, **self._kwargs)
        if (
            Worker.current_worker.has_accelerator
            and Worker.torch_platform.is_initialized()
        ):
            self._cuda_event = Worker.torch_platform.Event()
            self._cuda_event.record()
        if isinstance(self._result, AsyncWork):
            # If the result is another AsyncWork, find the last work in the chain
            # Set the flag only after all works are done
            last_work_in_chain = self._result.get_last_work()
            last_work_in_chain.then(self._done.set_result, True)
        else:
            self._done.set_result(True)

    def then(self, func: Callable, *args, **kwargs) -> "AsyncFuncWork":
        """Set a callback function to be called when the work is completed.

        Args:
            func (Callable): The function to call when the work is completed. Currently doesn't support return values.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        """
        # NOTE: If the _done flag is already set, the next work will be executed in the current thread
        # Do not make any assumptions about which thread the next work will be executed
        next_work = AsyncFuncWork(func, *args, **kwargs)
        self._next_work = next_work
        self._done.then(next_work)
        return next_work

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        while not self._done.done():
            await asyncio.sleep(0.001)  # Yield control to the event loop
        if self._cuda_event is not None:
            self._cuda_event.wait()
        result = self._result
        if isinstance(result, AsyncWork):
            # If the result is another AsyncWork, wait for it to complete
            return result.wait()
        else:
            return result

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        while not self._done.done():
            time.sleep(0.001)
        if self._cuda_event is not None:
            self._cuda_event.wait()
        result = self._result
        if isinstance(result, AsyncWork):
            # If the result is another AsyncWork, wait for it to complete
            return result.wait()
        else:
            return result

    def done(self):
        """Query the completion state of the work."""
        return self._done.done()

    def get_next_work(self):
        """Get the next AsyncWork chained to this one."""
        return self._next_work


class AsyncCollWork(AsyncWork):
    """Wrapper for dist.Work to allow asyncio-like awaitables."""

    def __init__(
        self,
        works: list[dist.Work],
    ):
        """Initialize the AsyncCollWork with a list of dist.Work objects.

        Args:
            works (List[dist.Work]): The list of dist.Work objects to wrap.

        """
        super().__init__()
        if not isinstance(works, list):
            works = [works]
        self._works = works
        self._next_work = None
        self._futures = [work.get_future() for work in works]

    async def async_wait(self):
        """Async wait for the work to complete."""
        for work in self._works:
            while not work.is_completed():
                await asyncio.sleep(0.001)  # Yield control to the event loop

    def wait(self):
        """Wait for the work to complete."""
        for work in self._works:
            work.wait()

    def then(self, func: Callable, *args, **kwargs) -> "AsyncFuncWork":
        """Set a callback function to be called when the work is completed.

        Args:
            func (Callable): The function to call when the work is completed. Currently doesn't support return values.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        """
        assert len(self._works) == 1, "then() does not support multiple works"
        next_work = AsyncFuncWork(func, *args, **kwargs)
        self._next_work = next_work
        self._futures[0].then(next_work)
        return next_work

    def get_next_work(self):
        """Get the next AsyncWork chained to this one."""
        return self._next_work

    def done(self):
        """Query the completion state of the work."""
        return all(future.done() for future in self._futures)

    def __add__(self, other: "AsyncCollWork") -> "AsyncCollWork":
        """Combine two AsyncCollWork objects."""
        if other is None:
            return self
        return AsyncCollWork(self._works + other._works)


class AsyncChannelWork(AsyncWork):
    """Asynchronous work for channel operations.

    This class handles the asynchronous execution of operations on a channel.
    It runs a dedicated thread to process channel operations asynchronously.
    All operations are enqueued and processed in the order they are created, so as to ensure the correct execution order.

    For each channel's each key, a dedicated asyncio coroutine is created to process all its operations.
    So, execution order is preserved within each channel-key combination.
    Different channels, or same channel under different keys, are thus processed concurrently without blocking each other, and no ordering is guaranteed between them.
    """

    # Operation queues used to communicate with the operation processing thread
    async_op_queue: queue.Queue["AsyncChannelWork"] = queue.Queue()
    # Operation queues for each channel-key combination
    channel_op_queue_map: dict[str, asyncio.Queue] = {}
    # Global thread lock
    lock: threading.Lock = None
    # Channel operation processing thread
    execution_thread: threading.Thread = None

    def __init__(
        self,
        channel_name: str,
        channel_key: str,
        channel_actor: ray.actor.ActorHandle,
        method: str,
        *args,
        **kwargs,
    ):
        """Initialize the AsyncChannelWork.

        Args:
            channel_name (str): The name of the channel.
            channel_key (str): The key for the channel.
            channel_actor (ray.actor.ActorHandle): The actor handle for the channel.
            method (str): The method to call on the channel actor.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.
        """
        self._channel_key = f"{channel_name}:{channel_key}"
        self._channel_actor = channel_actor
        self._method = method
        self._args = args
        self._kwargs = kwargs
        self._future = Future()

        # Create lock if not exist
        if AsyncChannelWork.lock is None:
            AsyncChannelWork.lock = threading.Lock()

        # Create thread if not exist
        with AsyncChannelWork.lock:
            if AsyncChannelWork.execution_thread is None:
                AsyncChannelWork.execution_thread = threading.Thread(
                    target=self._run, daemon=True
                )
                AsyncChannelWork.execution_thread.start()

        # Enqueue the operation
        AsyncChannelWork.async_op_queue.put(self)

    @staticmethod
    def _run():
        """Run the channel work."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Dedicated coroutine for processing a channel-key's operations
        async def process_work(channel_op_queue: asyncio.Queue[AsyncChannelWork]):
            while True:
                work = await channel_op_queue.get()
                try:
                    result = await work._execute()
                except Exception as e:
                    work._future.set_exception(e)
                else:
                    work._future.set_result(result)

        # Main loop of the processing thread
        async def run_loop():
            while True:
                try:
                    operation = AsyncChannelWork.async_op_queue.get(block=False)
                except queue.Empty:
                    await asyncio.sleep(0.001)  # Yield control to the event loop
                    continue

                op_queue_map = AsyncChannelWork.channel_op_queue_map
                channel_key = operation._channel_key

                # Create a new operation queue for the channel-key if it doesn't exist
                if channel_key not in op_queue_map:
                    channel_op_queue = asyncio.Queue()
                    op_queue_map[channel_key] = channel_op_queue
                    asyncio.create_task(process_work(channel_op_queue))

                channel_op_queue = op_queue_map[channel_key]
                await channel_op_queue.put(operation)

        loop.run_until_complete(run_loop())

    async def _execute(self):
        """Execute the operation."""
        method = getattr(self._channel_actor, self._method)
        return await method.remote(*self._args, **self._kwargs)

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        while not self._future.done():
            await asyncio.sleep(0.01)
        return self._future.value()

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        self._future.wait()
        return self._future.value()

    def done(self):
        """Query the completion state of the work."""
        return self._future.done()


class AsyncChannelCommWork(AsyncWork):
    """Asynchronous work for channel operations."""

    channel_data_store: dict[int, Future] = {}
    store_lock = threading.Lock()  # Protect store access

    def __init__(self, async_comm_work: AsyncWork, query_id: int):
        """Initialize the AsyncChannelWork with a async recv comm of the get operation.

        A query_id should be provided to identify the data get query.
        This is because the received data of this recv call may be from another get,
        as the async task execution order at the channel worker side is non-deterministic.
        And so we need to associate the data with an identifier and store it for later retrieval.

        Args:
            async_comm_work (AsyncWork): The async communication work to wrap.
            query_id (int): The query ID to associate with the work.

        """
        self._async_comm_work = async_comm_work
        # The async_comm_work's value is not necessarily the data of the get query associated with the query_id
        # Only when the query_id's Future is set is the data available
        self._query_id = query_id
        with AsyncChannelCommWork.store_lock:
            if query_id not in AsyncChannelCommWork.channel_data_store:
                AsyncChannelCommWork.channel_data_store[query_id] = Future()
            self._data_future = AsyncChannelCommWork.channel_data_store[query_id]
        self._async_comm_work.then(self._store_channel_data)

    def _store_channel_data(self):
        """Store channel data in the channel data store."""
        query_id, data = self._async_comm_work.wait()
        with AsyncChannelCommWork.store_lock:
            if query_id not in AsyncChannelCommWork.channel_data_store:
                AsyncChannelCommWork.channel_data_store[query_id] = Future()
            data_future = AsyncChannelCommWork.channel_data_store[query_id]
        data_future.set_result(data)

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        while not self._data_future.done():
            await asyncio.sleep(0.01)  # Yield control to the event loop
        with AsyncChannelCommWork.store_lock:
            AsyncChannelCommWork.channel_data_store.pop(self._query_id, None)
        return self._data_future.value()

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        self._data_future.wait()
        with AsyncChannelCommWork.store_lock:
            AsyncChannelCommWork.channel_data_store.pop(self._query_id, None)
        return self._data_future.value()

    def done(self):
        """Query the completion state of the work."""
        return self._data_future.done()
