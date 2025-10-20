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


from typing import Dict, List, Tuple

from ..cluster import Cluster
from .placement import Placement, PlacementStrategy


class FlexiblePlacementStrategy(PlacementStrategy):
    """This placement strategy allows processes to be placed on any accelerators (GPUs) by specifying a list of *global* accelerator IDs for each process.

    .. note::
            The global accelerator ID means the accelerator ID across the entire cluster. For example, if a cluster has 2 nodes, each with 8 GPUs, then the global GPU IDs are 0~7 for node 0 and 8~15 for node 1.

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
        >>> # `FlexiblePlacementStrategy` allows you to specify the *global* accelerator/GPU IDs for each process.
        >>> placement = FlexiblePlacementStrategy([[0, 1], [2], [3]])
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="flexible_placement", placement_strategy=placement
        ... )
        >>> # This will run 3 processes on the first node's GPU 0, 1, 2, 3, where the first process uses GPUs 0 and 1, the second process uses GPU 2, and the third process uses GPU 3.
        >>> my_worker.available_gpus().wait()
        [2, 1, 1]

    """

    def __init__(self, accelerator_ids_list: List[List[int]]):
        """Initialize the FlexiblePlacementStrategy.

        .. note::
            The accelerator IDs in each inner list must be on the same node and must be unique.

        .. note::
            The accelerator IDs of different processes should not overlap.

        .. note::
            The accelerator IDs will be sorted in ascending order both within each process and across processes (based on the first ID).

        Args:
            accelerator_ids_list (List[List[int]]): A list of lists, where each inner list contains the accelerator (e.g., GPU) IDs to allocate for a specific process.

        """
        super().__init__()
        assert len(accelerator_ids_list) > 0, (
            "The accelerator_id_list must not be empty."
        )

        self._accel_ids_list = accelerator_ids_list
        all_accelerator_ids = sorted(
            [accel_id for accel_ids in accelerator_ids_list for accel_id in accel_ids]
        )
        assert len(all_accelerator_ids) == len(set(all_accelerator_ids)), (
            f"The accelerator IDs of different processes {accelerator_ids_list} should not overlap."
        )
        self._start_accel_id = all_accelerator_ids[0]
        self._end_accel_id = all_accelerator_ids[-1]
        assert self._start_accel_id >= 0, (
            f"The start accelerator ID {self._start_accel_id} must be non-negative."
        )
        assert self._end_accel_id >= 0, (
            f"The end accelerator ID {self._end_accel_id} must be non-negative."
        )
        assert self._end_accel_id >= self._start_accel_id, (
            f"The end accelerator ID {self._end_accel_id} must be greater than or equal to the start accelerator ID {self._start_accel_id}."
        )

        self._placement_strategy = "FLEXIBLE"

        self._logger.info("")
        self._logger.info(
            f"Using flexible placement with accelerator IDs: {self._accel_ids_list}."
        )

    def _verify_accelerator_ids_for_process(
        self,
        accel_ids: List[int],
        cluster: Cluster,
    ):
        """Verify that the accelerator IDs for a process are valid."""
        for accel_id in accel_ids:
            # Check that all accelerator IDs are within the node range
            assert 0 <= accel_id < cluster.num_accelerators_in_cluster, (
                f"Accelerator ID {accel_id} is out of range. Must be between 0 and {cluster.num_accelerators_in_cluster - 1}."
            )

        # Check that all accelerator IDs of a process are on the same node
        node_ids = {
            cluster.get_node_id_from_accel_id(accel_id) for accel_id in accel_ids
        }
        assert len(node_ids) == 1, (
            f"All accelerator IDs {accel_ids} for a process must be on the same node."
        )

        # Check that all accelerator IDs of a process are unique
        assert len(accel_ids) == len(set(accel_ids)), (
            f"All accelerator IDs {accel_ids} for a process must be unique."
        )

    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the flexible strategy.

        Args:
            cluster (Cluster): The cluster object containing information about the nodes and accelerators.
            isolate_accelerator (bool): Whether accelerators not allocated to a worker will *not* be visible to the worker (by settings envs like CUDA_VISIBLE_DEVICES). Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on accelerators.

        """
        # Verify and sort the accelerator IDs for each process
        for i, accel_ids in enumerate(self._accel_ids_list):
            self._verify_accelerator_ids_for_process(accel_ids, cluster)
            self._accel_ids_list[i] = sorted(accel_ids)
        # Sort the list of accelerator IDs for processes based on the first accelerator ID in each list
        self._accel_ids_list.sort(key=lambda x: x[0])

        node_ids = [
            cluster.get_node_id_from_accel_id(accel_ids[0])
            for accel_ids in self._accel_ids_list
        ]
        node_id_accel_ids: List[Tuple[int, List[int]]] = list(
            zip(node_ids, self._accel_ids_list)
        )

        placements: List[Placement] = []
        for rank, (node_id, accel_ids) in enumerate(node_id_accel_ids):
            local_accelerator_ids = [
                cluster.global_accel_id_to_local_accel_id(accel_id)
                for accel_id in accel_ids
            ]
            if isolate_accelerator:
                visible_accelerators = [
                    str(accelerator_id) for accelerator_id in local_accelerator_ids
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
                    node_rank=-1,
                    accelerator_type=cluster.get_node_info(node_id).accelerator_type,
                    local_accelerator_id=local_accelerator_ids[0],
                    local_rank=-1,
                    local_world_size=0,
                    visible_accelerators=visible_accelerators,
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
