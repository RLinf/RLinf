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


class PackedPlacementStrategy(PlacementStrategy):
    """Placement strategy that allows processes to be placed on accelerators/GPUs in a close-packed manner. One process can have one or multiple accelerators.

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
        >>> placement = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=3)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="packed_placement", placement_strategy=placement
        ... )
        >>> my_worker.available_gpus().wait() # This will run 4 processes on the first node's GPU 0, 1, 2, 3, each using 1 GPU.
        [1, 1, 1, 1]
        >>>
        >>>
        >>> # `num_accelerators_per_process` allows for one process to hold multiple accelerators/GPUs.
        >>> # For example, if you want a process to hold 4 GPUs, you can set the `num_accelerators_per_process` to 4.
        >>> placement_chunked = PackedPlacementStrategy(
        ...     start_accelerator_id=0, end_accelerator_id=3, num_accelerators_per_process=2
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
        ...     start_accelerator_id=0, end_accelerator_id=3, stride=2, num_accelerators_per_process=2
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
        start_accelerator_id: int,
        end_accelerator_id: int,
        num_accelerators_per_process: int = 1,
        stride: int = 1,
    ):
        """Initialize the PackedPlacementStrategy.

        Args:
            start_accelerator_id (int): The global ID of the starting accelerator in the cluster for the placement.
            end_accelerator_id (int): The global ID of the end accelerator in the cluster for the placement.
            num_accelerators_per_process (int): The number of accelerators to allocate for each process.
            stride (int): The stride to use when allocating accelerators. This allows one process to have multiple accelerators in a strided manner, e.g., Accelerator 0, 2, 4 (stride 2) or Accelerator 0, 3, 6 (stride 3).

        """
        super().__init__()

        self._start_accel_id = start_accelerator_id
        self._end_accel_id = end_accelerator_id
        assert self._start_accel_id >= 0, (
            f"The start accelerator ID {self._start_accel_id} must be non-negative."
        )
        assert self._end_accel_id >= 0, (
            f"The end accelerator ID {self._end_accel_id} must be non-negative."
        )
        assert self._end_accel_id >= self._start_accel_id, (
            f"The end accelerator ID {self._end_accel_id} must be greater than or equal to the start accelerator ID {self._start_accel_id}."
        )
        self._num_accelerators = self._end_accel_id - self._start_accel_id + 1

        self._placement_strategy = "PACKED"
        self._num_accelerators_per_process = num_accelerators_per_process
        self._stride = stride

        assert (
            self._num_accelerators % (self._num_accelerators_per_process * self._stride)
            == 0
        ), (
            f"The number of accelerators {self._num_accelerators} must be divisible by num_accelerators_per_process * stride ({self._num_accelerators_per_process * self._stride})."
        )

        self._logger.info("")
        self._logger.info(
            f"Using packed placement starting from accelerator {self._start_accel_id}, ending at accelerator {self._end_accel_id}, with {self._num_accelerators_per_process} accelerators per process and stride {self._stride}."
        )

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the packed strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and accelerators.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on accelerators.

        """
        rank = 0
        placements: List[Placement] = []
        start_node = cluster.get_node_id_from_accel_id(self._start_accel_id)
        accel_usage_map: Dict[int, bool] = dict.fromkeys(
            range(self._start_accel_id, self._end_accel_id + 1), False
        )

        assert start_node < cluster.num_nodes, (
            f"The start accelerator ID {self._start_accel_id} is in Node ID {start_node}, but the cluster only has {cluster.num_nodes} nodes."
        )

        start_accel_id = self._start_accel_id
        node_rank = 0
        node_id = start_node
        local_accel_id = cluster.global_accel_id_to_local_accel_id(self._start_accel_id)
        local_rank = 0
        local_world_size = 1

        while True:
            # Generate the placement for one process
            assert local_accel_id + (
                self._num_accelerators_per_process - 1
            ) * self._stride <= cluster.get_node_num_accelerators(node_id), (
                f"Trouble finding placement for Rank {rank} which starts at accelerator {local_accel_id} in node {node_id}, with {self._num_accelerators_per_process} accelerators and stride {self._stride}. But only {cluster.get_node_num_accelerators(node_id)} accelerators available in the node. As a result, this process will spread across multiple nodes, which is impossible."
            )

            local_accelerators = list(
                range(
                    local_accel_id,
                    local_accel_id + self._num_accelerators_per_process * self._stride,
                    self._stride,
                )
            )
            global_accelerators = list(
                range(
                    start_accel_id,
                    start_accel_id + self._num_accelerators_per_process * self._stride,
                    self._stride,
                )
            )
            for accel_id in global_accelerators:
                accel_usage_map[accel_id] = True

            if isolate_accelerator:
                visible_accelerators = [
                    str(accel_id) for accel_id in local_accelerators
                ]
            else:
                visible_accelerators = [
                    str(accel_id)
                    for accel_id in range(cluster.get_node_num_accelerators(node_id))
                ]

            placements.append(
                Placement(
                    rank=rank,
                    node_id=node_id,
                    node_rank=node_rank,
                    accelerator_type=cluster.get_node_info(node_id).accelerator_type,
                    local_accelerator_id=local_accel_id,
                    local_rank=local_rank,
                    local_world_size=0,
                    visible_accelerators=visible_accelerators,
                    isolate_accelerator=isolate_accelerator,
                )
            )

            # The next placement
            rank += 1
            found_all = True
            for accel_id in sorted(accel_usage_map.keys()):
                if not accel_usage_map[accel_id]:
                    start_accel_id = accel_id
                    found_all = False
                    break

            next_node_id = cluster.get_node_id_from_accel_id(start_accel_id)
            if next_node_id != node_id:
                # Place to the next node
                assert next_node_id == node_id + 1, (
                    f"Rank {rank} is trying to move from Node {node_id} to Node {next_node_id}, "
                    f"but this is not allowed."
                )
                node_rank += 1
                node_id = next_node_id
                local_accel_id = 0
                local_rank = 0
                next_node = True
            else:
                local_accel_id = cluster.global_accel_id_to_local_accel_id(
                    start_accel_id
                )
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

            assert node_id < cluster.num_nodes, (
                f"Not enough ({cluster.num_nodes}) nodes in the cluster to generate the placement."
            )

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
