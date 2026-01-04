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


from contextlib import AbstractContextManager

from ..manager import DeviceLockManager, PortLockManager
from .worker import Worker


class DeviceLock(AbstractContextManager):
    """The lock (can be used as a context manager like conventional locks) to manage accelerator device resources.

    When multiple workers run on the same accelerators, this lock ensures that only one worker can access the accelerator resources at a time.
    This is useful for preventing contention on device memory and computation resources, especially when multiple workers colocate on the same device.

    This class is the worker-side handle for the device lock, which interacts with a global lock manager to acquire and release locks on behalf of the worker.
    """

    def __init__(self, worker: Worker):
        """Initialize the device lock."""
        self._worker = worker
        self._lock_manager = DeviceLockManager.get_proxy()

    def acquire(self):
        """Lock accelerator devices for the current worker.

        This is useful for resource isolation, e.g., accelerator memory and computation resources, when multiple workers run on the same accelerators.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is not None:
            self._lock_manager.acquire_devices(
                self._worker.worker_address, self._worker.global_accelerator_ids
            )
        else:
            raise RuntimeError("Cannot lock accelerators when not running in a worker.")

    def release(self):
        """Unlock accelerators for the current worker.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is not None:
            self._lock_manager.release_devices(
                self._worker.worker_address, self._worker.global_accelerator_ids
            )
        else:
            raise RuntimeError(
                "Cannot unlock accelerators when not running in a worker."
            )

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.release()


class PortLock:
    """A global lock to manage network port resources."""

    def __init__(self, worker: Worker):
        """Initialize the port lock."""
        self._worker = worker
        self._lock_manager = PortLockManager.get_proxy()

    def acquire(self, port: int) -> bool:
        """Lock a network port for the current worker.

        This is useful for preventing port conflicts when multiple workers run on the same node.

        Args:
            port (int): The network port to lock.

        Returns:
            bool: True if the port is successfully locked, False otherwise.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is not None:
            return self._lock_manager.acquire(
                self._worker._cluster_node_rank, self._worker._worker_name, port
            )
        else:
            raise RuntimeError("Cannot lock ports when not running in a worker.")
