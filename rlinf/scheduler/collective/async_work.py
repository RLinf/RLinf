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
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Callable, Optional, Sequence, overload

import ray.actor
import torch.distributed as dist
from torch.futures import Future

from ..worker import Worker


async def _await_torch_future(future: Future) -> None:
    """Wait for a torch Future without occupying an executor thread.

    A run_in_executor(None, future.wait) submission parks one default-executor
    thread per pending wait, so more concurrent waits than pool threads
    serialize even when their futures are already done.
    """
    if future.done():
        return
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    def _wake(_fut: Future) -> None:
        # Runs in the thread completing the future. The loop may already be
        # closed at shutdown; nobody is waiting then.
        try:
            loop.call_soon_threadsafe(lambda: waiter.done() or waiter.set_result(None))
        except RuntimeError:
            pass

    future.add_done_callback(_wake)
    await waiter


class PeerCommunicationError(RuntimeError):
    """A collective direction failed earlier, so its later operations do not run.

    One direction of a collective group (send or recv) has one failure state,
    shared by every pool queue of that direction: the exception of the first
    operation that failed there. Every later operation of that direction
    raises this error, chained to that first exception, instead of running.
    The data stream of a failed direction may have lost messages, so later
    receives on it cannot be trusted either.
    """


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
        pass_self: bool = False,
        **kwargs,
    ):
        """Initialize the AsyncFuncWork with a function and its arguments.

        Args:
            func (Callable): The function to call when the work is completed.
            *args: Positional arguments to pass to the function.
            pass_self (bool): If True, ``self`` (this AsyncFuncWork instance) is
                prepended to ``func``'s positional arguments at call time. Use
                this when the wrapped function needs to attribute timing or
                other side effects back to this work without going through a
                global / thread-local.
            **kwargs: Keyword arguments to pass to the function.

        """
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._pass_self = pass_self
        self._done = Future()
        self._result = None
        self._next_work = None
        self._cuda_event = None
        self._exec_time = None  # Time spent in the function execution
        self._perf_time = None  # Time set externally

    @property
    def time(self) -> float:
        """Get the performance time of the function."""
        if self._perf_time is not None:
            return self._perf_time
        else:
            return self._exec_time

    def add_perf_time(self, perf_time: float):
        """Accumulate ``perf_time`` into the work's payload-transfer time.

        Used by collective payload-tracking helpers that may fire multiple times
        per work (e.g. dataclass sends that ship a tensor list and a skeleton).
        """
        if self._perf_time is None:
            self._perf_time = perf_time
        else:
            self._perf_time += perf_time

    def __call__(self, future: Future):
        """Execute the function and set the done flag.

        No exception may escape this call: the work queue thread calls it,
        and an exception that leaves it ends the thread, so this work and
        every later work on the same queue would stay pending forever. A
        failure is set on the done future instead.
        """
        start = time.perf_counter()
        if future is not None:
            # A work chained with then() is called with the previous step's
            # done future; when that step failed, this one has no input to
            # work on.
            try:
                future.wait()
            except Exception as exc:
                self._exec_time = time.perf_counter() - start
                self._done.set_exception(exc)
                return
        try:
            if self._pass_self:
                self._result = self._func(self, *self._args, **self._kwargs)
            else:
                self._result = self._func(*self._args, **self._kwargs)
        except Exception as exc:
            self._exec_time = time.perf_counter() - start
            self._done.set_exception(exc)
            return
        self._exec_time = time.perf_counter() - start
        if (
            Worker.current_worker.has_accelerator
            and Worker.torch_platform.is_initialized()
        ):
            self._cuda_event = Worker.torch_platform.Event()
            self._cuda_event.record()
        if isinstance(self._result, AsyncWork):
            # If the result is another AsyncWork, find the last work in the
            # chain and complete only after all of them are done; an error
            # anywhere in the chain reaches this work's waiter.
            last_work_in_chain = self._result.get_last_work()
            last_work_in_chain.add_done_callback(
                lambda _: self._complete_after_chain(last_work_in_chain)
            )
        else:
            self._done.set_result(True)

    def _complete_after_chain(self, last_work: "AsyncWork") -> None:
        """Complete this work once the AsyncWork it returned has finished.

        Registered with add_done_callback, so it also runs when the chain
        failed and passes that failure on to this work's waiters.
        """
        try:
            last_work.wait()
        except Exception as exc:
            self._done.set_exception(exc)
        else:
            self._done.set_result(True)

    def then(self, func: Callable, *args, **kwargs) -> "AsyncFuncWork":
        """Set a callback function to be called when the work is completed.

        A work chained with then() does not run its function when the work
        before it failed; it passes the exception on.

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

    def add_done_callback(self, func: Callable[[Future], None]) -> None:
        """Call ``func`` when this work completes, whether it succeeded or failed.

        Unlike :meth:`then`, the callback also runs after a failure, so it can
        pass the error on. It is called with this work's done future; calling
        ``wait()`` on that future raises the work's exception. It runs in the
        thread that completes the work.
        """
        self._done.add_done_callback(func)

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        if not self._done.done():
            await _await_torch_future(self._done)
        # Waiting above does not read the future; this raises when the work
        # failed, as wait() does.
        self._done.wait()
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
        self._done.wait()
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
        # Keep the executor wait: NCCL work.wait() only synchronizes the
        # current stream with the work's completion, which is not equivalent
        # to the completion of work.get_future().
        await asyncio.get_event_loop().run_in_executor(None, self.wait)

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

    def add_done_callback(self, func: Callable[[Future], None]) -> None:
        """Call ``func`` when this work completes, whether it succeeded or failed.

        Unlike :meth:`then`, the callback also runs after a failure, so it can
        pass the error on. It is called with the future of its collective;
        calling ``wait()`` on that future raises the work's exception. It runs
        in the thread that completes the work.
        """
        assert len(self._works) == 1, (
            "add_done_callback() does not support multiple works"
        )
        self._futures[0].add_done_callback(func)

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
    """Asynchronous work for channel operations."""

    # Global thread lock, reentrant: a remote call that has already finished
    # runs _complete in the thread that registers it, which may hold this lock.
    lock: threading.RLock = threading.RLock()
    # Last future per key in execution loop
    last_future_per_key: dict[str, Future] = {}

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

        # Enqueue the operation
        with AsyncChannelWork.lock:
            last_fut = AsyncChannelWork.last_future_per_key.get(self._channel_key)
            # Recorded before the call is issued: a call that has already
            # finished runs _complete right here, and _complete removes the
            # entry only if it finds its own future.
            AsyncChannelWork.last_future_per_key[self._channel_key] = self._future
            if last_fut is None:
                self._execute()
            else:
                last_fut.then(lambda _: self._execute())

    def _execute(self):
        method = getattr(self._channel_actor, self._method)
        future: ConcurrentFuture = method.remote(*self._args, **self._kwargs).future()
        future.add_done_callback(self._complete)

    def _complete(self, future: ConcurrentFuture) -> None:
        """Complete the work's future with the remote call's outcome."""
        with AsyncChannelWork.lock:
            if not self._future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    # An exception raised in a concurrent.futures callback is
                    # only logged, leaving this future forever pending; every
                    # waiter on it and every later operation chained on the
                    # same channel key hangs.
                    self._future.set_exception(exc)
                else:
                    self._future.set_result(result)
            # Keys are per group on the payload channels: keeping the entry of
            # a completed key grows this table for the whole run.
            if (
                AsyncChannelWork.last_future_per_key.get(self._channel_key)
                is self._future
            ):
                del AsyncChannelWork.last_future_per_key[self._channel_key]

    def fail(self, exc: Exception) -> None:
        """Fail this work with ``exc`` while its future is still pending.

        The remote call this work issued may still be in flight, and may
        complete later; the first outcome to arrive wins, so a settled future
        is left alone. The key's table entry goes the way _complete removes
        it, so a later operation on the key does not queue behind a work that
        never returns.
        """
        with AsyncChannelWork.lock:
            if not self._future.done():
                self._future.set_exception(exc)
            if (
                AsyncChannelWork.last_future_per_key.get(self._channel_key)
                is self._future
            ):
                del AsyncChannelWork.last_future_per_key[self._channel_key]

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        if not self._future.done():
            await _await_torch_future(self._future)
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

    # Futures of pending gets, by (channel name, channel rank) and then by
    # query ID.
    channel_data_store: dict[tuple[str, int], dict[int, Future]] = {}
    # Channel ranks whose recv failed: later data from them is dropped, since
    # the message stream may have lost whatever the failed recv was reading.
    failed_peers: set[tuple[str, int]] = set()
    store_lock = threading.Lock()  # Protect store access

    def __init__(
        self,
        async_comm_work: AsyncWork,
        query_id: int,
        channel_actor: ray.actor.ActorHandle,
        peer: tuple[str, int],
    ):
        """Initialize the AsyncChannelWork with a async recv comm of the get operation.

        A query_id should be provided to identify the data get query.
        This is because the received data of this recv call may be from another get,
        as the async task execution order at the channel worker side is non-deterministic.
        And so we need to associate the data with an identifier and store it for later retrieval.

        Args:
            async_comm_work (AsyncWork): The async communication work to wrap.
            query_id (int): The query ID to associate with the work.
            channel_actor (ray.actor.ActorHandle): The actor handle for the channel.
            peer (tuple[str, int]): The ``(channel name, channel rank)`` this
                work's recv receives from.

        """
        self._async_comm_work = async_comm_work
        # The async_comm_work's value is not necessarily the data of the get query associated with the query_id
        # Only when the query_id's Future is set is the data available
        self._query_id = query_id
        self._channel_actor = channel_actor
        self._peer = peer
        with AsyncChannelCommWork.store_lock:
            self._data_future = AsyncChannelCommWork.channel_data_store.setdefault(
                peer, {}
            ).setdefault(query_id, Future())
        # Runs on success and on failure: a failed recv has to reach the
        # gets still waiting on this channel rank.
        self._async_comm_work.add_done_callback(lambda _: self._store_channel_data())

    def _store_channel_data(self):
        """Store channel data in the channel data store."""
        try:
            data, query_id = self._async_comm_work.wait()
        except Exception as exc:
            # Recvs from one channel rank are not matched to gets: each takes
            # the next message, and the query ID sent with the data names its
            # get. After a failed recv, the data of a get still waiting on
            # this channel rank may never arrive, so fail every such get; the
            # getter waits on its data future, not on this callback. Gets on
            # other channel ranks have their own recvs and are left alone.
            with AsyncChannelCommWork.store_lock:
                pending = AsyncChannelCommWork.channel_data_store.pop(self._peer, {})
                for future in pending.values():
                    if not future.done():
                        future.set_exception(exc)
                if not self._data_future.done():
                    self._data_future.set_exception(exc)
                # Recvs of this channel rank that were already running when
                # this one failed still deliver, and their data belongs to a
                # stream nobody can match to a get any more.
                AsyncChannelCommWork.failed_peers.add(self._peer)
            return
        with AsyncChannelCommWork.store_lock:
            if self._peer in AsyncChannelCommWork.failed_peers:
                # The rank failed earlier: this data cannot be trusted, and
                # storing it would leave an entry no get ever takes.
                return
            data_future = AsyncChannelCommWork.channel_data_store.setdefault(
                self._peer, {}
            ).setdefault(query_id, Future())
            if not data_future.done():
                data_future.set_result(data)

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        if not self._data_future.done():
            await _await_torch_future(self._data_future)
        with AsyncChannelCommWork.store_lock:
            AsyncChannelCommWork.channel_data_store.get(self._peer, {}).pop(
                self._query_id, None
            )
        return self._data_future.value()

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        self._data_future.wait()
        with AsyncChannelCommWork.store_lock:
            AsyncChannelCommWork.channel_data_store.get(self._peer, {}).pop(
                self._query_id, None
            )
        return self._data_future.value()

    def done(self):
        """Query the completion state of the work."""
        return self._data_future.done()


class AsyncRayWork(AsyncWork):
    """Asynchronous work for ray operations."""

    def __init__(self, ray_object: ray.ObjectRef):
        """Initialize the AsyncRayWork."""
        self._ray_object = ray_object

    async def async_wait(self):
        """Async wait for the work to complete."""
        return await self._ray_object

    def wait(self):
        """Wait for the work to complete."""
        return ray.get(self._ray_object)


class AsyncRouteWork(AsyncWork):
    """Aggregate multiple async channel operations into one routed result."""

    def __init__(
        self,
        works: Sequence[AsyncWork],
        finalize_fn: Callable[[list[Any]], Any],
    ):
        """Initialize an async routed work handle.

        Args:
            works: Async work handles for each routed shard.
            finalize_fn: Function that validates and merges completed shard results.
        """
        self._works = list(works)
        self._finalize_fn = finalize_fn
        self._result: Any = None
        self._has_result = False

    async def async_wait(self):
        """Wait asynchronously for all routed works and return the finalized result."""
        if self._has_result:
            return self._result
        results = [await work.async_wait() for work in self._works]
        self._result = self._finalize_fn(results)
        self._has_result = True
        return self._result

    def wait(self):
        """Wait synchronously for all routed works and return the finalized result."""
        if self._has_result:
            return self._result
        results = [work.wait() for work in self._works]
        self._result = self._finalize_fn(results)
        self._has_result = True
        return self._result

    def then(self, func: Callable, *args, **kwargs):
        """Register a chained callback.

        Raises:
            NotImplementedError: AsyncRouteWork does not support chained callbacks.
        """
        raise NotImplementedError("AsyncRouteWork does not support chained callbacks.")

    def done(self):
        """Return whether all routed works have completed."""
        if self._has_result:
            return True
        return all(work.done() for work in self._works)

    def get_next_work(self):
        """Return the next pending work if supported."""
        return None
