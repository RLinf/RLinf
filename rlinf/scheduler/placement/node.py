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

from typing import Optional, Sequence

from ..cluster import Cluster, NodeGroupInfo
from .placement import Placement, PlacementStrategy


class NodePlacementStrategy(PlacementStrategy):
    """This placement strategy places processes on specific nodes (using *global* node rank) without limiting accelerators. This is useful for CPU-only workers who do not rely on accelerators.

    .. note::
            The global node rank means the node rank across the entire cluster. For example, if a cluster has 16 nodes, the node ranks are 0~15.

    Example::

        >>> from rlinf.scheduler import (
        ...     Cluster,
        ...     Worker,
        ...     NodePlacementStrategy,
        ... )
        >>>
        >>> class MyWorker(Worker):
        ...     def __init__(self, msg: str = "Hello, World!"):
        ...         super().__init__()
        ...         self._msg = msg
        ...
        ...     def hello(self):
        ...         return self._rank
        ...
        >>>
        >>> cluster = Cluster(num_nodes=1)
        >>>
        >>> # `NodePlacementStrategy` allows you to specify the *global* node ranks for each process.
        >>> placement = NodePlacementStrategy([0] * 4)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="node_placement", placement_strategy=placement
        ... )
        >>> my_worker.hello().wait() # This will run 4 processes on the first node
        [0, 1, 2, 3]

    """

    def __init__(
        self,
        node_ranks: list[int],
        node_group_label: Optional[str | Sequence[str]] = None,
    ):
        """Initialize the NodePlacementStrategy.

        .. note::
            The node ranks will be sorted.

        Args:
            node_ranks (List[int]): A list of node ranks to allocate for the processes.
            node_group_label (Optional[str | Sequence[str]]): The label or list of labels of the node groups to which the node ranks belong. If specified, the node_ranks are local ranks within the selected node groups. Otherwise, node_ranks are global ranks.

        """
        super().__init__()
        assert len(node_ranks) > 0, "The node_ranks list must not be empty."

        self._node_ranks = sorted(node_ranks)
        if node_group_label is None:
            self._node_group_labels = None
        elif isinstance(node_group_label, str):
            self._node_group_labels = [node_group_label]
        elif isinstance(node_group_label, Sequence):
            self._node_group_labels = [str(label) for label in node_group_label]
        else:
            self._node_group_labels = [str(node_group_label)]
        self._placement_strategy = "NODE"

        self._logger.info("")
        self._logger.info(
            f"Using node placement with node ranks: {self._node_ranks}, "
            f"node groups: {self._node_group_labels or [NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL]}."
        )

    def _resolve_node_groups(self, cluster: Cluster) -> list[NodeGroupInfo]:
        """Resolve user-specified node groups against the cluster."""
        if not self._node_group_labels:
            return [cluster.get_node_group()]

        node_groups: list[NodeGroupInfo] = []
        seen_node_ranks: set[int] = set()
        for label in self._node_group_labels:
            node_group = cluster.get_node_group(label)
            assert node_group is not None, (
                f"Node group with label {label} not found in the cluster."
            )
            # Ensure node groups do not overlap
            overlap = seen_node_ranks.intersection(set(node_group.node_ranks))
            assert not overlap, (
                f"Node groups {self._node_group_labels} have overlapping nodes {overlap}. "
                "Node placement across multiple groups requires disjoint node sets."
            )
            seen_node_ranks.update(node_group.node_ranks)
            node_groups.append(node_group)
        return node_groups

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        """Generate a list of placements based on the node placement strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and hardware.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes.

        """
        placements: list[Placement] = []
        node_groups = self._resolve_node_groups(cluster)

        # Flatten node ranks in the order of the provided node groups
        flat_nodes: list[tuple[int, str]] = []
        for ng in node_groups:
            flat_nodes.extend([(nr, ng.label) for nr in ng.node_ranks])

        total_nodes = len(flat_nodes)
        for rank in self._node_ranks:
            if rank < 0 or rank >= total_nodes:
                raise IndexError(
                    f"Node rank {rank} is out of range for selected node groups {self._node_group_labels} "
                    f"(total nodes {total_nodes})."
                )

        cluster_node_ranks = []
        resolved_group_labels = []
        for ng_rank in self._node_ranks:
            cluster_node_rank, label = flat_nodes[ng_rank]
            cluster_node_ranks.append(cluster_node_rank)
            resolved_group_labels.append(label)

        for rank, cluster_node_rank in enumerate(cluster_node_ranks):
            visible_devices = list(
                range(cluster.get_node_info(cluster_node_rank).num_accelerators)
            )
            visible_devices = [str(device) for device in visible_devices]
            placements.append(
                Placement(
                    rank=rank,
                    cluster_node_rank=cluster_node_rank,
                    placement_node_rank=-1,
                    accelerator_type=cluster.get_node_info(
                        cluster_node_rank
                    ).accelerator_type,
                    local_accelerator_rank=-1
                    if len(visible_devices) == 0
                    else visible_devices[0],
                    local_rank=-1,
                    local_world_size=0,
                    visible_accelerators=visible_devices,
                    isolate_accelerator=isolate_accelerator,
                    local_hardware_ranks=[],
                    node_group_label=resolved_group_labels[rank],
                )
            )

        node_rank = 0
        local_rank = 0
        local_world_size = 0
        current_node_id = placements[0].cluster_node_rank
        node_local_world_size: dict[int, int] = {}
        for placement in placements:
            if placement.cluster_node_rank != current_node_id:
                assert placement.cluster_node_rank > current_node_id, (
                    "Placements must be sorted by node_id."
                )
                node_local_world_size[current_node_id] = local_world_size
                current_node_id = placement.cluster_node_rank
                node_rank += 1
                local_rank = 0
                local_world_size = 0
            placement.placement_node_rank = node_rank
            placement.local_rank = local_rank
            local_rank += 1
            local_world_size += 1
        # For the last node
        node_local_world_size[current_node_id] = local_world_size

        for placement in placements:
            placement.local_world_size = node_local_world_size[
                placement.cluster_node_rank
            ]

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
