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
from ..hardware import Accelerator
from .placement import Placement, PlacementStrategy


class PackedPlacementStrategy(PlacementStrategy):
    """Placement strategy that allows processes to be placed on hardware (e.g., GPUs) in a close-packed manner. One process can have one or multiple hardware.

    The following example shows how to use the placement strategy.

    Example::

        >>> from rlinf.scheduler import (
        ...     Cluster,
        ...     Worker,
        ...     PackedPlacementStrategy,
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
        ...     def available_gpus(self):
        ...         import torch
        ...         available_gpus = torch.cuda.device_count()
        ...         gpu_ids = [
        ...             torch.cuda.get_device_properties(i) for i in range(available_gpus)
        ...         ]
        ...         return available_gpus
        >>>
        >>> cluster = Cluster(num_nodes=1)
        >>>
        >>> # `PackedPlacementStrategy` will fill up nodes with workers before moving to the next node.
        >>> placement = PackedPlacementStrategy(start_hardware_rank=0, end_hardware_rank=3)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="packed_placement", placement_strategy=placement
        ... )
        >>> my_worker.available_gpus().wait() # This will run 4 processes on the first node's GPU 0, 1, 2, 3, each using 1 GPU.
        [1, 1, 1, 1]
        >>>
        >>>
        >>> # `num_hardware_per_process` allows for one process to hold multiple accelerators/GPUs.
        >>> # For example, if you want a process to hold 4 GPUs, you can set the `num_hardware_per_process` to 4.
        >>> placement_chunked = PackedPlacementStrategy(
        ...     start_hardware_rank=0, end_hardware_rank=3, num_hardware_per_process=2
        ... )
        >>> my_worker_chunked = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="chunked_placement",
        ...     placement_strategy=placement_chunked,
        ... )
        >>> my_worker_chunked.available_gpus().wait()  # This will run 2 processes, each using 2 GPUs (0-1 and 2-3) of the first node.
        [2, 2]
        >>>
        >>>
        >>> # `stride` allows for strided placement of workers across GPUs.
        >>> # For example, if you want to place workers on every second GPU, you can set the stride to 2.
        >>> placement_strided = PackedPlacementStrategy(
        ...     start_hardware_rank=0, end_hardware_rank=3, stride=2, num_hardware_per_process=2
        ... )
        >>> my_worker_strided = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="strided_placement",
        ...     placement_strategy=placement_strided,
        ... )
        >>> # This will run 2 processes, each using 2 GPUs (0,2 1,3) of the first node.
        >>> my_worker_strided.available_gpus().wait()
        [2, 2]

    """

    def __init__(
        self,
        start_hardware_rank: int,
        end_hardware_rank: int,
        num_hardware_per_process: int = 1,
        stride: int = 1,
        node_group: Optional[str] = None,
    ):
        """Initialize the PackedPlacementStrategy.

        Args:
            start_hardware_rank (int): The global rank of the starting hardware in the cluster or node group for the placement.
            end_hardware_rank (int): The global rank of the end hardware in the cluster or node group for the placement.
            num_hardware_per_process (int): The number of hardware resources to allocate for each process.
            stride (int): The stride to use when allocating hardware. This allows one process to have multiple hardware in a strided manner, e.g., GPU 0, 2, 4 (stride 2) or GPU 0, 3, 6 (stride 3).
            node_group (Optional[str]): The label of the node group to use for placement. This allows you to assign the placement to nodes within a specific group, especially in a heterogeneous cluster. If None, the entire cluster is considered.

        """
        super().__init__()

        self._start_hw_rank = start_hardware_rank
        self._end_hw_rank = end_hardware_rank
        self._node_group = node_group
        assert self._start_hw_rank >= 0, (
            f"The start hardware rank {self._start_hw_rank} must be non-negative."
        )
        assert self._end_hw_rank >= 0, (
            f"The end hardware rank {self._end_hw_rank} must be non-negative."
        )
        assert self._end_hw_rank >= self._start_hw_rank, (
            f"The end hardware rank {self._end_hw_rank} must be greater than or equal to the start hardware rank {self._start_hw_rank}."
        )
        self._num_hardware = self._end_hw_rank - self._start_hw_rank + 1

        self._placement_strategy = "PACKED"
        self._num_hardware_per_process = num_hardware_per_process
        self._stride = stride

        assert (
            self._num_hardware % (self._num_hardware_per_process * self._stride) == 0
        ), (
            f"The number of hardware {self._num_hardware} must be divisible by num_hardware_per_process * stride ({self._num_hardware_per_process * self._stride})."
        )

        self._logger.info("")
        self._logger.info(
            f"Using packed placement starting from hardware {self._start_hw_rank}, ending at hardware {self._end_hw_rank}, with {self._num_hardware_per_process} hardware per process and stride {self._stride}."
        )

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        """Generate a list of placements based on the packed strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and accelerators.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            list[Placement]: A list of Placement objects representing the placements of processes on accelerators.

        """
        rank = 0
        placements: list[Placement] = []
        node_group = cluster.get_node_group(self._node_group)

        start_node = node_group.get_node_by_hardware_rank(self._start_hw_rank)
        hw_usage_map: dict[int, bool] = dict.fromkeys(
            range(self._start_hw_rank, self._end_hw_rank + 1), False
        )

        assert start_node is not None, (
            f"The start hardware rank {self._start_hw_rank} cannot be found in the node group with hardware type {node_group.hardware_type} and hardware ranks {node_group.local_hardware_ranks}."
        )

        start_hw_rank = self._start_hw_rank
        node_rank = 0
        cluster_node_rank = start_node.node_rank
        local_hw_rank = node_group.get_local_hardware_rank(self._start_hw_rank)
        local_rank = 0
        local_world_size = 1

        while True:
            # Generate the placement for one process
            assert local_hw_rank + (
                self._num_hardware_per_process - 1
            ) * self._stride <= cluster.get_node_info(
                cluster_node_rank
            ).get_hw_resource_count(node_group.hardware_type), (
                f"Trouble finding placement for Rank {rank} which starts at hardware {local_hw_rank} in node {cluster_node_rank} and node group {node_group.label}, with {self._num_hardware_per_process} hardware and stride {self._stride}. But only {cluster.get_node_info(cluster_node_rank).get_hw_resource_count(node_group.hardware_type)} hardware available in the node. As a result, this process will spread across multiple nodes, which is impossible."
            )

            local_hw_ranks = list(
                range(
                    local_hw_rank,
                    local_hw_rank + self._num_hardware_per_process * self._stride,
                    self._stride,
                )
            )
            global_hw_ranks = list(
                range(
                    start_hw_rank,
                    start_hw_rank + self._num_hardware_per_process * self._stride,
                    self._stride,
                )
            )
            for hw_rank in global_hw_ranks:
                hw_usage_map[hw_rank] = True

            if isolate_accelerator and node_group.hardware_type == Accelerator.HW_TYPE:
                local_accel_ranks = local_hw_ranks
                visible_accelerators = [
                    str(accel_rank) for accel_rank in local_hw_ranks
                ]
            else:
                local_accel_ranks = list(
                    range(cluster.get_node_info(cluster_node_rank).num_accelerators)
                )
                visible_accelerators = [
                    str(accel_rank) for accel_rank in local_accel_ranks
                ]

            placements.append(
                Placement(
                    rank=rank,
                    cluster_node_rank=cluster_node_rank,
                    placement_node_rank=node_rank,
                    accelerator_type=cluster.get_node_info(
                        cluster_node_rank
                    ).accelerator_type,
                    local_accelerator_rank=local_accel_ranks[0],
                    local_rank=local_rank,
                    local_world_size=0,
                    visible_accelerators=visible_accelerators,
                    isolate_accelerator=isolate_accelerator,
                    local_hardware_ranks=local_hw_ranks,
                    node_group_label=node_group.label,
                )
            )

            # The next placement
            rank += 1
            found_all = True
            for hw_rank in sorted(hw_usage_map.keys()):
                if not hw_usage_map[hw_rank]:
                    start_hw_rank = hw_rank
                    found_all = False
                    break

            next_cluster_node_rank = node_group.get_node_by_hardware_rank(
                start_hw_rank
            ).node_rank
            if next_cluster_node_rank != cluster_node_rank:
                # Place to the next node
                node_rank += 1
                cluster_node_rank = next_cluster_node_rank
                local_hw_rank = 0
                local_rank = 0
                next_node = True
            else:
                local_hw_rank = node_group.get_local_hardware_rank(start_hw_rank)
                local_rank += 1
                next_node = False

            if next_node or found_all:
                # If we are at the end of a node, set local_world_size for all previous placements whose local_world_size == 0
                # Reversal traverse the placements to set local_world_size
                for i in range(len(placements) - 1, -1, -1):
                    if placements[i].local_world_size == 0:
                        placements[i].local_world_size = local_world_size
                    else:
                        break
                local_world_size = 1  # Reset for the next node
            else:
                local_world_size += 1

            if found_all:
                break

            assert cluster_node_rank in node_group.node_ranks, (
                f"Not enough nodes {node_group.node_ranks} in the node group {node_group.label} to generate the placement."
            )

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
