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
from typing import Any, Callable, List, Optional, overload

import ray
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
        works: List[dist.Work],
    ):
        """Initialize the AsyncCollWork with a list of dist.Work objects.

        Args:
            works (List[dist.Work]): The list of dist.Work objects to wrap.

        """
        super().__init__()
        if not isinstance(works, List):
            works = [works]
        self._works = works
        self._next_work = None
        self._futures = [work.get_future() for work in works]

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        for work in self._works:
            while not work.is_completed():
                await asyncio.sleep(0.001)  # Yield control to the event loop

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
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
    """Asynchronous work for channel operations."""

    def __init__(self, func_result: ray.ObjectRef):
        """Initialize the AsyncChannelWork with a Ray function result.

        Args:
            func_result (ray.ObjectRef): The Ray function result to wrap.

        """
        self._func_result = func_result

    async def async_wait(self):
        """Async wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        return await self._func_result

    def wait(self):
        """Wait for the work to complete.

        Returns:
            Any: The result of the work if applicable, otherwise None.

        """
        return ray.get(self._func_result)
