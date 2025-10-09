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


from typing import Dict, List

from ..cluster import Cluster
from .placement import Placement, PlacementStrategy


class NodePlacementStrategy(PlacementStrategy):
    """This placement strategy places processes on specific nodes (using *global* node ID) without limiting accelerators. This is useful for CPU-only workers who do not rely on accelerators.

    .. note::
            The global node ID means the node ID across the entire cluster. For example, if a cluster has 16 nodes, the node IDs are 0~15.

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
        >>> # `NodePlacementStrategy` allows you to specify the *global* node IDs for each process.
        >>> placement = NodePlacementStrategy([0] * 4)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="node_placement", placement_strategy=placement
        ... )
        >>> my_worker.hello().wait() # This will run 4 processes on the first node
        [0, 1, 2, 3]

    """

    def __init__(self, node_ids: List[int]):
        """Initialize the NodePlacementStrategy.

        .. note::
            The node IDs will be sorted.

        Args:
            node_ids (List[int]): A list of node IDs to allocate for the processes.

        """
        super().__init__()
        assert len(node_ids) > 0, "The node_ids list must not be empty."

        self._node_ids = sorted(node_ids)
        self._placement_strategy = "NODE"

        self._logger.info("")
        self._logger.info(f"Using node placement with node IDs: {self._node_ids}.")

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the node placement strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and accelerators.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on accelerators.

        """
        placements: List[Placement] = []
        for rank, node_id in enumerate(self._node_ids):
            assert node_id < cluster.num_nodes, (
                f"Node ID {node_id} exceeds number of available nodes {cluster.num_nodes}"
            )
            visible_devices = list(range(cluster.get_node_num_accelerators(node_id)))
            visible_devices = [str(device) for device in visible_devices]
            placements.append(
                Placement(
                    rank=rank,
                    node_id=node_id,
                    node_rank=-1,
                    accelerator_type=cluster.get_node_info(node_id).accelerator_type,
                    local_accelerator_id=-1
                    if len(visible_devices) == 0
                    else visible_devices[0],
                    local_rank=-1,
                    local_world_size=0,
                    visible_accelerators=visible_devices,
                    isolate_accelerator=isolate_accelerator,
                )
            )

        node_rank = 0
        local_rank = 0
        local_world_size = 0
        current_node_id = placements[0].node_id
        node_local_world_size: Dict[int, int] = {}
        for placement in placements:
            if placement.node_id != current_node_id:
                assert placement.node_id > current_node_id, (
                    "Placements must be sorted by node_id."
                )
                node_local_world_size[current_node_id] = local_world_size
                current_node_id = placement.node_id
                node_rank += 1
                local_rank = 0
                local_world_size = 0
            placement.node_rank = node_rank
            placement.local_rank = local_rank
            local_rank += 1
            local_world_size += 1
        # For the last node
        node_local_world_size[current_node_id] = local_world_size

        for placement in placements:
            placement.local_world_size = node_local_world_size[placement.node_id]

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
