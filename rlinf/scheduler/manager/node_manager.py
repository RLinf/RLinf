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

from ..cluster import NodeInfo
from .manager import Manager


class NodeManager(Manager):
    """Global manager of node metadata information."""

    MANAGER_NAME = "NodeManager"

    def __init__(
        self,
        nodes: list[NodeInfo],
    ):
        """Initialize the NodeManager.

        Args:
            nodes (list[NodeInfo]): List of NodeInfo objects representing the nodes in the cluster

        """
        self._nodes = nodes

    def get_nodes(self):
        """Get the list of nodes in the cluster."""
        return self._nodes
