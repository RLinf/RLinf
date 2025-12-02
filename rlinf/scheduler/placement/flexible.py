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

from ..cluster import Cluster, NodeGroupInfo
from ..hardware import Accelerator
from .placement import Placement, PlacementStrategy


class FlexiblePlacementStrategy(PlacementStrategy):
    """This placement strategy allows processes to be placed on any hardware (accelerators, robots, etc.) by specifying a list of *global* hardware ranks for each process.

    .. note::
            The global hardware rank means the hardware rank across the entire cluster or a node group if node_group_label is given. For example, if a cluster has 2 nodes, each with 8 GPUs, then the global GPU ranks are 0~7 for node 0 and 8~15 for node 1.

    The following example shows how to use the placement strategy.

    Example::

        >>> from rlinf.scheduler import (
        ...     Cluster,
        ...     Worker,
        ...     FlexiblePlacementStrategy,
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
        >>> # `FlexiblePlacementStrategy` allows you to specify the *global* accelerator/GPU ranks for each process.
        >>> placement = FlexiblePlacementStrategy([[0, 1], [2], [3]])
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="flexible_placement", placement_strategy=placement
        ... )
        >>> # This will run 3 processes on the first node's GPU 0, 1, 2, 3, where the first process uses GPUs 0 and 1, the second process uses GPU 2, and the third process uses GPU 3.
        >>> my_worker.available_gpus().wait()
        [2, 1, 1]

    """

    def __init__(
        self,
        hardware_ranks_list: list[list[int]],
        node_group_label: Optional[str] = None,
    ):
        """Initialize the FlexiblePlacementStrategy.

        .. note::
            The hardware ranks in each inner list must be on the same node and must be unique.

        .. note::
            The hardware ranks will be sorted in ascending order both within each process and across processes (based on the first rank).

        Args:
            hardware_ranks_list (List[List[int]]): A list of lists, where each inner list contains the hardware (e.g., GPU) ranks to allocate for a specific process.
            node_group_label (Optional[str]): The label of the node group to which the accelerator ranks belong. If specified, the accelerator ranks mean local ranks within the node group. Otherwise, accelerator ranks are global ranks.

        """
        super().__init__()
        assert len(hardware_ranks_list) > 0, (
            "The hardware_ranks_list must not be empty."
        )

        self._node_group_label = node_group_label
        self._hardware_ranks_list = hardware_ranks_list
        all_hardware_ranks = sorted(
            [hw_rank for hw_ranks in hardware_ranks_list for hw_rank in hw_ranks]
        )
        self._start_rank = all_hardware_ranks[0]
        self._end_rank = all_hardware_ranks[-1]
        assert self._start_rank >= 0, (
            f"The start hardware rank {self._start_rank} must be non-negative."
        )
        assert self._end_rank >= 0, (
            f"The end hardware rank {self._end_rank} must be non-negative."
        )
        assert self._end_rank >= self._start_rank, (
            f"The end hardware rank {self._end_rank} must be greater than or equal to the start hardware rank {self._start_rank}."
        )

        self._placement_strategy = "FLEXIBLE"

        self._logger.info("")
        self._logger.info(
            f"Using flexible placement with hardware ranks: {self._hardware_ranks_list}."
        )

    def _verify_hw_ranks_for_process(
        self,
        hw_ranks: list[int],
        node_group: NodeGroupInfo,
    ):
        """Verify that the accelerator ranks for a process are valid."""
        for hw_rank in hw_ranks:
            # Check that all hardware ranks are within the node range
            assert 0 <= hw_rank < node_group.hardware_resource_count, (
                f"{node_group.hardware_type} hardware rank {hw_rank} is out of range in {node_group.label}. Must be between 0 and {node_group.hardware_resource_count - 1}."
            )

        # Check that all hardware ranks of a process are on the same node
        node_ranks = {
            node_group.get_node_by_hardware_rank(hw_rank).node_rank
            for hw_rank in hw_ranks
        }
        assert len(node_ranks) == 1, (
            f"All hardware ranks {hw_ranks} for a process must be on the same node. Instead, the hardware exist across node ranks {node_ranks}"
        )

        # Check that all hardware ranks of a process are unique
        assert len(hw_ranks) == len(set(hw_ranks)), (
            f"All hardware ranks {hw_ranks} for a process must be unique."
        )

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        """Generate a list of placements based on the flexible strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and accelerators.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on accelerators.

        """
        node_group = cluster.get_node_group(self._node_group_label)
        assert node_group is not None, (
            f"Node group with label {self._node_group_label} not found in the cluster."
        )
        # Verify and sort the hardware ranks for each process
        for i, hw_ranks in enumerate(self._hardware_ranks_list):
            self._verify_hw_ranks_for_process(hw_ranks, node_group)
            self._hardware_ranks_list[i] = sorted(hw_ranks)
        # Sort the list of hardware ranks for processes based on the first hardware rank in each list
        self._hardware_ranks_list.sort(key=lambda x: x[0])

        cluster_node_ranks = [
            node_group.get_node_by_hardware_rank(hw_ranks[0]).node_rank
            for hw_ranks in self._hardware_ranks_list
        ]
        cluster_node_rank_hw_ranks: list[tuple[int, list[int]]] = list(
            zip(cluster_node_ranks, self._hardware_ranks_list)
        )

        placements: list[Placement] = []
        for rank, (cluster_node_rank, hw_ranks) in enumerate(
            cluster_node_rank_hw_ranks
        ):
            local_hw_ranks = [
                node_group.get_local_hardware_rank(hw_rank) for hw_rank in hw_ranks
            ]
            if isolate_accelerator and node_group.hardware_type == Accelerator.HW_TYPE:
                local_accel_ranks = local_hw_ranks
                visible_accelerators = [
                    str(accel_rank) for accel_rank in local_accel_ranks
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
                    placement_node_rank=-1,
                    accelerator_type=cluster.get_node_info(
                        cluster_node_rank
                    ).accelerator_type,
                    local_accelerator_rank=local_accel_ranks[0],
                    local_rank=-1,
                    local_world_size=0,
                    visible_accelerators=visible_accelerators,
                    isolate_accelerator=isolate_accelerator,
                    local_hardware_ranks=local_hw_ranks,
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
