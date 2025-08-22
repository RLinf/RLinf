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

from .placement import Placement, PlacementStrategy


class PackedPlacementStrategy(PlacementStrategy):
    """Placement strategy that allows processes to be placed on GPUs in a close-packed manner. One process can have one or multiple GPUs."""

    def __init__(
        self,
        start_gpu_id: int,
        end_gpu_id: int,
        num_gpus_per_process: int = 1,
        stride: int = 1,
    ):
        """Initialize the PackedPlacementStrategy.

        Args:
            start_gpu_id (int): The global starting GPU ID in the cluster for the placement.
            end_gpu_id (int): The global ending GPU ID in the cluster for the placement.
            num_gpus_per_process (int): The number of GPUs to allocate for each process.
            stride (int): The stride to use when allocating GPUs. This allows one process to have multiple GPUs in a strided manner, e.g., GPU 0, 2, 4 (stride 2) or GPU 0, 3, 6 (stride 3).

        """
        super().__init__(start_gpu_id, end_gpu_id)
        self._placement_strategy = "PACKED"
        self._num_gpus_per_process = num_gpus_per_process
        self._stride = stride

        assert self._num_gpus % (self._num_gpus_per_process * self._stride) == 0, (
            f"The number of GPUs {self._num_gpus} must be divisible by num_gpus_per_process * stride ({self._num_gpus_per_process * self._stride})."
        )

        self._logger.info("")
        self._logger.info(
            f"Using packed placement starting from GPU {self._start_gpu_id}, ending at GPU {self._end_gpu_id}, with {self._num_gpus_per_process} GPUs per process and stride {self._stride}."
        )

    def get_placement(
        self,
        num_nodes_in_cluster: int,
        num_gpus_per_node: int,
        isolate_gpu: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the packed strategy.

        Args:
            num_nodes_in_cluster (int): Total number of nodes in the cluster.
            num_gpus_per_node (int): Number of GPUs per node.
            isolate_gpu (bool): Whether to isolate the GPUs for each process by setting `CUDA_VISIBLE_DEVICES`. Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on GPUs.

        """
        rank = 0
        placements: List[Placement] = []
        start_node = self._start_gpu_id // num_gpus_per_node
        gpu_usage_map: Dict[int, bool] = dict.fromkeys(
            range(self._start_gpu_id, self._end_gpu_id + 1), False
        )

        assert start_node < num_nodes_in_cluster, (
            f"The start GPU ID {self._start_gpu_id} is in Node ID {start_node}, but the cluster only has {num_nodes_in_cluster} nodes."
        )

        start_gpu_id = self._start_gpu_id
        node_rank = 0
        node_id = start_node
        local_gpu_id = self._start_gpu_id % num_gpus_per_node
        local_rank = 0
        local_world_size = 1

        while True:
            # Generate the placement for one process
            assert (
                local_gpu_id + (self._num_gpus_per_process - 1) * self._stride
                <= num_gpus_per_node
            ), (
                f"Trouble finding placement for Rank {rank} which starts at GPU {local_gpu_id} in node {node_id}, with {self._num_gpus_per_process} GPUs and stride {self._stride}. But only {num_gpus_per_node} GPUs available in the node. As a result, this process will spread across multiple nodes, which is impossible."
            )

            local_gpus = list(
                range(
                    local_gpu_id,
                    local_gpu_id + self._num_gpus_per_process * self._stride,
                    self._stride,
                )
            )
            global_gpus = list(
                range(
                    start_gpu_id,
                    start_gpu_id + self._num_gpus_per_process * self._stride,
                    self._stride,
                )
            )
            for gpu_id in global_gpus:
                gpu_usage_map[gpu_id] = True

            if isolate_gpu:
                cuda_visible_devices = [str(gpu_id) for gpu_id in local_gpus]
            else:
                cuda_visible_devices = [
                    str(gpu_id) for gpu_id in range(num_gpus_per_node)
                ]

            placements.append(
                Placement(
                    rank=rank,
                    node_id=node_id,
                    node_rank=node_rank,
                    local_gpu_id=local_gpu_id,
                    local_rank=local_rank,
                    local_world_size=0,
                    cuda_visible_devices=cuda_visible_devices,
                    isolate_gpu=isolate_gpu,
                )
            )

            # The next placement
            rank += 1
            found_all = True
            for gpu_id in sorted(gpu_usage_map.keys()):
                if not gpu_usage_map[gpu_id]:
                    start_gpu_id = gpu_id
                    found_all = False
                    break

            next_node_id = start_gpu_id // num_gpus_per_node
            if next_node_id != node_id:
                # Place to the next node
                assert next_node_id == node_id + 1, (
                    f"Rank {rank} is trying to move from Node {node_id} to Node {next_node_id}, "
                    f"but this is not allowed."
                )
                node_rank += 1
                node_id = next_node_id
                local_gpu_id = 0
                local_rank = 0
                next_node = True
            else:
                local_gpu_id = start_gpu_id % num_gpus_per_node
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

            assert node_id < num_nodes_in_cluster, (
                f"Not enough ({num_nodes_in_cluster}) nodes in the cluster to generate the placement."
            )

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
