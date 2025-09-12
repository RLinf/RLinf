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

from .placement import Placement, PlacementStrategy


class FlexiblePlacementStrategy(PlacementStrategy):
    """This placement strategy allows processes to be placed on any GPUs by specifying a list of GPU IDs for each process."""

    def __init__(self, gpu_ids_list: List[List[int]]):
        """Initialize the FlexiblePlacementStrategy.

        .. note::
            The GPU IDs in each inner list must be on the same node and must be unique.

        .. note::
            The GPU IDs of different processes should not overlap.

        .. note::
            The GPU IDs will be sorted in ascending order both within each process and across processes (based on the first GPU ID).

        Args:
            gpu_ids_list (List[List[int]]): A list of lists, where each inner list contains the GPU IDs to allocate for a specific process.

        """
        assert len(gpu_ids_list) > 0, "The gpu_id_list_per_process must not be empty."

        self._gpu_ids_list = gpu_ids_list
        all_gpu_ids = sorted([gpu_id for gpu_ids in gpu_ids_list for gpu_id in gpu_ids])
        assert len(all_gpu_ids) == len(set(all_gpu_ids)), (
            f"The GPU IDs of different processes {gpu_ids_list} should not overlap."
        )
        super().__init__(all_gpu_ids[0], all_gpu_ids[-1])
        self._placement_strategy = "FLEXIBLE"

        self._logger.info("")
        self._logger.info(
            f"Using flexible placement with GPU IDs: {self._gpu_ids_list}."
        )

    def _verify_gpu_ids_for_process(
        self, gpu_ids: List[int], num_nodes_in_cluster: int, num_gpus_per_node: int
    ):
        """Verify that the GPU IDs for a process are valid."""
        for gpu_id in gpu_ids:
            # Check that all GPU IDs are within the node range
            assert 0 <= gpu_id < num_nodes_in_cluster * num_gpus_per_node, (
                f"GPU ID {gpu_id} is out of range. Must be between 0 and {num_nodes_in_cluster * num_gpus_per_node - 1}."
            )

        # Check that all GPU IDs of a process are on the same node
        node_ids = {gpu_id // num_gpus_per_node for gpu_id in gpu_ids}
        assert len(node_ids) == 1, (
            f"All GPU IDs {gpu_ids} for a process must be on the same node."
        )

        # Check that all GPU IDs of a process are unique
        assert len(gpu_ids) == len(set(gpu_ids)), (
            f"All GPU IDs {gpu_ids} for a process must be unique."
        )

    def get_placement(
        self,
        num_nodes_in_cluster: int,
        num_gpus_per_node: int,
        isolate_gpu: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the flexible strategy.

        Args:
            num_nodes_in_cluster (int): Total number of nodes in the cluster.
            num_gpus_per_node (int): Number of GPUs per node.
            isolate_gpu (bool): Whether to isolate the GPUs for each process by setting `CUDA_VISIBLE_DEVICES`. Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on GPUs.

        """
        # Verify and sort the GPU IDs for each process
        for i, gpu_ids in enumerate(self._gpu_ids_list):
            self._verify_gpu_ids_for_process(
                gpu_ids, num_nodes_in_cluster, num_gpus_per_node
            )
            self._gpu_ids_list[i] = sorted(gpu_ids)
        # Sort the list of GPU IDs for processes based on the first GPU ID in each list
        self._gpu_ids_list.sort(key=lambda x: x[0])

        node_ids = [gpu_ids[0] // num_gpus_per_node for gpu_ids in self._gpu_ids_list]
        node_id_gpu_ids: List[Tuple[int, List[int]]] = list(
            zip(node_ids, self._gpu_ids_list)
        )

        placements: List[Placement] = []
        for rank, (node_id, gpu_ids) in enumerate(node_id_gpu_ids):
            local_gpu_ids = [gpu_id % num_gpus_per_node for gpu_id in gpu_ids]
            if isolate_gpu:
                cuda_visible_devices = [str(gpu_id) for gpu_id in local_gpu_ids]
            else:
                cuda_visible_devices = [
                    str(gpu_id) for gpu_id in range(num_gpus_per_node)
                ]
            placements.append(
                Placement(
                    rank=rank,
                    node_id=node_id,
                    node_rank=-1,
                    local_gpu_id=local_gpu_ids[0],
                    local_rank=-1,
                    local_world_size=0,
                    cuda_visible_devices=cuda_visible_devices,
                    isolate_gpu=isolate_gpu,
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
