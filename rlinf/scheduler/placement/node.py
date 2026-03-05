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

from ..cluster import Cluster
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

    def __init__(self, node_ranks: list[int], node_group_label: Optional[str] = None):
        """Initialize the NodePlacementStrategy.

        .. note::
            The node ranks will be sorted.

        Args:
            node_ranks (List[int]): A list of node ranks to allocate for the processes.
            node_group_label (Optional[str]): The label of the node group to which the node ranks belong. If specified, the node_ranks means local ranks within the node group. Otherwise, node_ranks is the global ranks.

        """
        super().__init__()
        assert len(node_ranks) > 0, "The node_ranks list must not be empty."

        self._node_ranks = sorted(node_ranks)
        self._node_group_label = node_group_label
        self._placement_strategy = "NODE"

        self._logger.info("")
        self._logger.info(f"Using node placement with node ranks: {self._node_ranks}.")

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
        node_group = cluster.get_node_group(self._node_group_label)
        cluster_node_ranks = node_group.group_ranks_to_global_ranks(self._node_ranks)

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
                    node_group_label=node_group.label,
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
