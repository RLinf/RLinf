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

from typing import Optional

from .manager import Manager
from .worker_manager import WorkerInfo


class CollectiveGroupInfo:
    """Metadata that stores information about a collective group."""

    def __init__(self, group_name: str, workers: list[WorkerInfo], master_addr: str):
        """Initialize the CollectiveGroupInfo.

        Args:
            group_name (str): The name of the collective group.
            workers (list[WorkerInfo]): List of WorkerInfo objects representing the workers in the collective
            master_addr (str): The address of the master node for the collective group.

        """
        self.group_name = group_name
        self.workers = workers
        self.master_addr = master_addr
        self.master_port: Optional[int] = None

        assert len(workers) == len(set(workers)), (
            f"Workers in collective group {group_name} must be unique. Found duplicates."
        )

        self.world_size = len(workers)

    def __eq__(self, other):
        """Check if two CollectiveGroupInfo instances are equal."""
        return (
            isinstance(other, CollectiveGroupInfo)
            and self.group_name == other.group_name
            and self.workers == other.workers
            and self.master_addr == other.master_addr
            and self.world_size == other.world_size
        )

    def __ne__(self, other):
        """Check if two CollectiveGroupInfo instances are not equal."""
        return not self.__eq__(other)


class CollectiveManager(Manager):
    """Global manager of collective metadata information."""

    MANAGER_NAME = "CollectiveManager"

    def __init__(self):
        """Initialize the CollectiveManager."""
        self._name_info_map: dict[str, CollectiveGroupInfo] = {}

    def register_collective_group(self, group_info: CollectiveGroupInfo):
        """Create a collective group with the given name and workers.

        Args:
            group_info (CollectiveGroupInfo): The collective group information to register.

        Raises:
            ValueError: If the collective group already exists with a different configuration.

        """
        if (
            group_info.group_name in self._name_info_map
            and group_info != self._name_info_map[group_info.group_name]
        ):
            raise ValueError(
                f"Collective group {group_info.group_name} already exists but tried to register a different group, old one is {self._name_info_map[group_info.group_name]}, new one is {group_info}."
            )

        self._name_info_map[group_info.group_name] = group_info

    def get_collective_group(self, group_name: str) -> Optional[CollectiveGroupInfo]:
        """Get the collective group information by name.

        Args:
            group_name (str): The name of the collective group to retrieve.

        Returns:
            Optional[CollectiveGroupInfo]: The collective group information if found, otherwise None.

        """
        if group_name not in self._name_info_map:
            return None

        return self._name_info_map[group_name]

    def set_master_port_info(self, group_name: str, master_port: int):
        """Set the master port for a collective group.

        Args:
            group_name (str): The name of the collective group.
            master_port (int): The master port to set.

        """
        if group_name in self._name_info_map:
            self._name_info_map[group_name].master_port = master_port
        else:
            raise ValueError(f"Collective group {group_name} does not exist.")

    def get_master_port_info(self, group_name: str) -> Optional[int]:
        """Get the master port for a collective group.

        Args:
            group_name (str): The name of the collective group.

        Returns:
            Optional[int]: The master port if set, otherwise None.

        """
        if group_name in self._name_info_map:
            return self._name_info_map[group_name].master_port
        return None

    def reset_master_port_info(self, group_name: str):
        """Reset the master port for a collective group.

        Args:
            group_name (str): The name of the collective group.

        """
        if group_name in self._name_info_map:
            self._name_info_map[group_name].master_port = None
        else:
            raise ValueError(f"Collective group {group_name} does not exist.")
