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

import bisect
from dataclasses import dataclass

from ..hardware import AcceleratorType
from .manager import Manager


class WorkerAddress:
    """A class that describes the path to a worker in a WorkerGroup."""

    def __init__(self, root_group_name: str, ranks: int | list[int] = []):
        """Initialize the WorkerAddress.

        Args:
            root_group_name (str): The name of the root worker group.
            ranks (int | List[int]): The ranks of the worker in the group. If a single integer is provided, it will be treated as a list with one element. If a list is provided, it means the path to the worker in the group.

        """
        assert ":" not in root_group_name, (
            "Root group name cannot contain ':' character."
        )
        if isinstance(ranks, int):
            ranks = [ranks]
        self.root_group_name = root_group_name
        self.rank_path = ranks
        self.rank = ranks[-1] if ranks else 0

    def get_name(self) -> str:
        """Convert the WorkerName to a string representation.

        Returns:
            str: The string representation of the worker name.

        """
        return self.root_group_name + "".join([f":{rank}" for rank in self.rank_path])

    def get_parent_rank(self) -> int:
        """Get the rank of the parent worker in the WorkerAddress.

        Returns:
            int: The rank of the parent worker, or 0 if this is the root worker.

        """
        return self.rank_path[-2] if len(self.rank_path) > 1 else 0

    def get_parent_address(self) -> "WorkerAddress":
        """Get the parent WorkerAddress by removing the last rank from the rank path.

        Returns:
            WorkerAddress: The parent WorkerAddress.

        """
        if len(self.rank_path) == 0:
            # I am Root!
            return None
        return WorkerAddress(self.root_group_name, self.rank_path[:-1])

    def get_child_address(self, rank: int) -> "WorkerAddress":
        """Get the child WorkerAddress by adding a new rank to the rank path.

        Args:
            rank (int): The rank of the child worker.

        Returns:
            WorkerAddress: The child WorkerAddress.

        """
        return WorkerAddress(self.root_group_name, self.rank_path + [rank])

    def __eq__(self, other: "WorkerAddress"):
        """Check if two WorkerAddress instances are equal."""
        if other is None:
            return False
        return (
            self.root_group_name == other.root_group_name
            and self.rank_path == other.rank_path
        )

    def __ne__(self, value):
        """Check if two WorkerAddress instances are not equal."""
        return not self.__eq__(value)

    def __hash__(self):
        """Hash function for WorkerAddress."""
        return hash((self.root_group_name, tuple(self.rank_path)))

    @classmethod
    def from_name(cls, worker_name: str) -> "WorkerAddress":
        """Create a WorkerName instance from a string representation.

        Args:
            worker_name (str): The string representation of the worker name.

        Returns:
            WorkerAddress: An instance of WorkerAddress.

        """
        components = worker_name.split(":")
        root_group_name = components[0]
        ranks = [int(rank) for rank in components[1:]]
        return cls(root_group_name, ranks)

    @classmethod
    def from_parent_name_rank(cls, parent_name: str, rank: int) -> "WorkerAddress":
        """Create a WorkerName instance from a parent name and a rank.

        Args:
            parent_name (str): The name of the parent worker group.
            rank (int): The rank of the child worker.

        Returns:
            WorkerAddress: An instance of WorkerAddress.

        """
        parent_address = cls.from_name(parent_name)
        return parent_address.get_child_address(rank)


@dataclass
class WorkerInfo:
    """For local access to worker properties instead of calling remote functions of the Ray Actor."""

    address: "WorkerAddress"
    """WorkerAddress of the worker."""

    rank: int
    """Rank of the worker in the group."""

    cluster_node_rank: int
    """Node ID where the worker is placed."""

    accelerator_type: AcceleratorType
    """Type of accelerator where the worker is placed."""

    accelerator_rank: int
    """Accelerator ID where the worker is placed."""

    node_ip: str
    """IP address of the node where the worker is placed."""

    node_port: int
    """Port of the node where the worker is placed."""

    available_accelerators: list[int]
    """List of global accelerator IDs available to the worker."""

    def __hash__(self):
        """Hash function for WorkerInfo."""
        return self.address.__hash__()


class WorkerNode:
    """A tree structure to manage workers and their dependencies."""

    def __init__(self, worker_address: WorkerAddress, worker_info: WorkerInfo = None):
        """Initialize the WorkerNode.

        Args:
            worker_address (WorkerAddress): The address of the worker.
            worker_info (WorkerInfo): The information about the worker. Defaults to None.

        """
        self._worker_address = worker_address
        self._worker_info = worker_info
        self._nodes: list[WorkerNode] = []

    @classmethod
    def find_node(
        cls, root: "WorkerNode", worker_address: WorkerAddress
    ) -> "WorkerNode":
        """Find a worker node by its name in the tree structure.

        Args:
            root (WorkerNode): The root node of the worker tree.
            worker_address (WorkerAddress): The address of the worker to find.

        Returns:
            WorkerNode: The found worker node, or None if not found.

        """
        root_worker_name = root._worker_address.get_name()

        # Not in the this root
        if root_worker_name != worker_address.root_group_name:
            return None

        # Direct descendance of the root node
        if (
            len(worker_address.rank_path) == 0
            and root_worker_name == worker_address.root_group_name
        ):
            return root

        cur_node = root
        for rank in worker_address.rank_path:
            found = False
            for node in cur_node._nodes:
                if node._worker_address.rank == rank:
                    cur_node = node
                    found = True
                    break
            if not found:
                # If the node is not found, it means the worker is not registered
                return None

        return cur_node

    def add_child(self, rank: int, worker_info: WorkerInfo):
        """Add a child worker node to the current node.

        Args:
            rank (int): The rank of the child worker node.
            worker_info (WorkerInfo): The information about the child worker node.

        """
        child_address = self._worker_address.get_child_address(rank)
        child_node = WorkerNode(child_address, worker_info)

        # Maintain sorted order of child nodes based on their rank
        bisect.insort(self._nodes, child_node, key=lambda x: x._worker_address.rank)

    def __str__(self):
        """Produce the string representation of the worker node tree."""
        tree = ""

        def _str_helper(node: "WorkerNode", depth: int = 0) -> str:
            indent = "  " * depth
            tree_str = f"{indent}{node._worker_address.get_name()}\n"
            for child in node._nodes:
                tree_str += _str_helper(child, depth + 1)
            return tree_str

        tree += _str_helper(self)
        return tree


class WorkerManager(Manager):
    """Global manager of worker and communication information."""

    MANAGER_NAME = "WorkerManager"

    def __init__(self):
        """Initialize the WorkerManager."""
        self._root_workers: list[WorkerNode] = []

    def register_worker(self, worker_address: WorkerAddress, worker_info: WorkerInfo):
        """Register a new worker in the worker manager.

        Args:
            worker_address (WorkerAddress): The address of the worker to register.
            worker_info (WorkerInfo): The information about the worker to register.

        """
        # Find the parent node or create a new root node if no parent is specified
        parent_address = worker_address.get_parent_address()
        rank = worker_address.rank
        if parent_address is None:
            self._root_workers.append(WorkerNode(parent_address, worker_info))
        else:
            for root in self._root_workers:
                node = WorkerNode.find_node(root, parent_address)
                if node is not None:
                    node.add_child(rank, worker_info)
                    return

            # Create a new root node if the parent is not found
            root = WorkerNode(parent_address)
            assert len(root._nodes) == 0, (
                f"Root node {root._worker_address.get_name()} already has children."
            )
            root.add_child(rank, worker_info)
            self._root_workers.append(root)

    def get_worker_info(self, worker_address: WorkerAddress) -> WorkerInfo:
        """Get the worker information by its address.

        Args:
            worker_address (WorkerAddress): The address of the worker to retrieve.

        Returns:
            WorkerInfo: The information about the worker. Returns None if the worker is not found.

        """
        for root in self._root_workers:
            node = WorkerNode.find_node(root, worker_address)
            if node is not None:
                return node._worker_info
        return None
