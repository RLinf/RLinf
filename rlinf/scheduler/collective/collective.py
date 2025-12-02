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
from typing import Optional

from ..cluster import Cluster
from ..manager import CollectiveGroupInfo, CollectiveManager, WorkerInfo
from ..worker import Worker, WorkerAddress
from .collective_group import CollectiveGroup


class Collective:
    """The singleton class for managing and handling calls to local collective groups."""

    def __init__(self, cur_worker: Worker):
        """Initialize the Collective.

        Args:
            cur_worker (Worker): The current worker instance that will be used to manage collective groups.

        """
        self._name_group_map: dict[str, CollectiveGroup] = {}
        self._coll_manager = CollectiveManager.get_proxy()
        self._worker_manager = cur_worker.manager_proxy
        self._cur_worker_address = cur_worker.worker_address
        self._logger = cur_worker._logger

    def create_collective_group(
        self, worker_addresses: list[WorkerAddress], group_name: Optional[str] = None
    ) -> CollectiveGroup:
        """Create a collective group with the given workers and name. If the group already exists, it will return the existing group.

        Args:
            worker_addresses (List[WorkerAddress]): The list of workers to include in the collective group.
            group_name (str, optional): The name of the collective group. If None, a name will be generated based on the worker names.

        Returns:
            CollectiveGroup: The created collective group.

        """
        if group_name is None:
            group_name = self._get_group_name(worker_addresses)

        # Already exists locally, return the existing group
        if group_name in self._name_group_map:
            return self._name_group_map[group_name]

        # Check if the group already exists in the global collective manager
        group_info = self._coll_manager.get_collective_group(group_name)
        self._name_group_map[group_name] = CollectiveGroup(
            group_info, self, group_name, worker_addresses, self._cur_worker_address
        )
        return self._name_group_map[group_name]

    def _get_group_name(self, workers: list[WorkerAddress]):
        return "cg-" + "-".join([worker.get_name() for worker in workers])

    def _get_worker_info_safe(self, worker_address: WorkerAddress) -> WorkerInfo:
        """Busy wait for the worker info to be available."""
        worker_info = self._worker_manager.get_worker_info(worker_address)
        count = 0
        while worker_info is None:
            time.sleep(0.001)
            worker_info = self._worker_manager.get_worker_info(worker_address)
            count += 1
            if count % Cluster.TIMEOUT_WARN_TIME == 0:
                self._logger.warning(
                    f"Waited {count / 1000} seconds for worker {worker_address.get_name()} to be ready..."
                )
        return worker_info

    def _get_group_info_safe(self, group_name: str) -> CollectiveGroupInfo:
        """Busy wait for the group info to be available."""
        group_info = self._coll_manager.get_collective_group(group_name)
        count = 0
        while group_info is None:
            time.sleep(0.001)
            group_info = self._coll_manager.get_collective_group(group_name)
            count += 1
            if count % Cluster.TIMEOUT_WARN_TIME == 0:
                self._logger.warning(
                    f"Waited {count // 1000} seconds for collective group {group_name} to be ready..."
                )
        return group_info
