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
import threading
import time

from ..cluster import Cluster
from .manager import Manager
from .worker_manager import WorkerAddress


class WorkerDeviceLock(asyncio.Lock):
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
            self._worker_address = worker_address
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


class DeviceLockManager(Manager):
    """Global manager for device locks.

    This manager holds the lock of every accelerator device in the cluster, and offers APIs for workers to acquire and release these locks.
    """

    MANAGER_NAME = "DeviceLockManager"

    def __init__(self):
        """Initialize the lock manager."""
        cluster = Cluster()
        self._device_locks = [
            WorkerDeviceLock() for _ in range(cluster.num_accelerators)
        ]

    async def acquire_devices(
        self, worker_address: WorkerAddress, accel_ids: list[int]
    ):
        """Lock the specified accelerator device IDs.

        Args:
            worker_address (WorkerAddress): The address of the worker requesting the lock.
            accel_ids (List[int]): The list of accelerator IDs to lock.
        """
        await asyncio.gather(
            *(
                self._device_locks[accel_id].acquire(worker_address)
                for accel_id in accel_ids
            )
        )

    async def release_devices(
        self, worker_address: WorkerAddress, accel_ids: list[int]
    ):
        """Release the specified accelerator device IDs.

        Args:
            worker_address (WorkerAddress): The address of the worker releasing the lock.
            accel_ids (List[int]): The list of accelerator IDs to release.
        """
        await asyncio.gather(
            *(
                self._device_locks[accel_id].release(worker_address)
                for accel_id in accel_ids
            )
        )


class PortLockManager(Manager):
    """Global manager for port locks.

    This manager holds the lock of every network port in the cluster, and offers APIs for workers to acquire and release these locks.
    """

    MANAGER_NAME = "PortLockManager"

    def __init__(self):
        """Initialize the port lock manager."""
        # Mapping from (node_rank, port) to lock status
        self._port_locks: dict[(int, int), str] = {}
        self._release_thread = threading.Thread(
            target=self._release_daemon, daemon=True
        )
        self._lock = threading.Lock()
        self._release_thread.start()

    def acquire(self, node_rank: int, worker_name: str, port: int) -> bool:
        """Lock the specified port on the given node rank.

        Args:
            node_rank (int): The rank of the node.
            worker_name (str): The name of the worker requesting the lock.
            port (int): The port number to lock.

        Returns:
            bool: True if the port was successfully locked, False if it was already locked.
        """
        with self._lock:
            key = (node_rank, port)
            if self._port_locks.get(key, None) is not None:
                return False
            self._port_locks[key] = worker_name
            return True

    def _release_daemon(self):
        """Daemon thread to periodically clean up released ports when it finds a worker is no longer alive."""
        from ..worker import Worker

        while True:
            with self._lock:
                locked_workers = set(self._port_locks.values())
            dead_workers = set()
            for worker_name in locked_workers:
                if not Worker.check_worker_alive(worker_name):
                    dead_workers.add(worker_name)
            with self._lock:
                for key, worker_name in list(self._port_locks.items()):
                    if worker_name in dead_workers:
                        self._port_locks.pop(key)
            time.sleep(60)  # Check every 60 seconds
