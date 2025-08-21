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

import logging
from dataclasses import dataclass
from typing import List, overload


@dataclass
class Placement:
    """Class representing the placement of a worker on a specific GPU."""

    rank: int
    """Global rank of the worker in the cluster."""

    node_id: int
    """Node ID where the worker is placed."""

    node_rank: int
    """Rank of the node in the cluster."""

    local_gpu_id: int
    """Local GPU ID on the node."""

    local_rank: int
    """Local rank of the worker on the node."""

    local_world_size: int
    """Local world size (number of workers) on the node."""

    cuda_visible_devices: List[str]
    """List of CUDA visible devices for the worker."""

    isolate_gpu: bool
    """Flag to indicate if the local rank should be set to zero. This is useful for workers that require multiple GPUs."""


class PlacementStrategy:
    """Base class for placement strategies.

    The following example shows how to place the worker on specified nodes and GPUs with the placement strategy.

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
        >>> cluster = Cluster(num_nodes=1, num_gpus_per_node=8)
        >>>
        >>> # Launch 8 processes
        >>> my_worker_group = MyWorker.create_group(msg="Hello").launch(cluster=cluster)
        >>> # This will execute the hello method on all the 8 processes in the group.
        >>> futures = my_worker_group.hello()
        >>> # Wait for all workers to complete and get the results.
        >>> futures.wait()
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>>
        >>>
        >>> # Create a placement strategy. This controls how workers are placed on the cluster.
        >>> # `PackedPlacementStrategy` will fill up nodes with workers before moving to the next node.
        >>> placement = PackedPlacementStrategy(start_gpu_id=4, end_gpu_id=7)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="packed_group", placement_strategy=placement
        ... )
        >>> my_worker.available_gpus().wait() # This will run 4 processes on the first node's GPU 4, 5, 6, 7, each using 1 GPU.
        [1, 1, 1, 1]
        >>>
        >>>
        >>> # `num_gpus_per_process` allows for one process to hold multiple GPUs.
        >>> # For example, if you want a process to hold 4 GPUs, you can set the `num_gpus_per_process` to 4.
        >>> placement_chunked = PackedPlacementStrategy(
        ...     start_gpu_id=0, end_gpu_id=7, num_gpus_per_process=4
        ... )
        >>> my_worker_chunked = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="my_worker_chunked",
        ...     placement_strategy=placement_chunked,
        ... )
        >>> my_worker_chunked.available_gpus().wait()  # This will run 2 processes, each using 4 GPUs (0-3 and 4-7) of the first node.
        [4, 4]
        >>>
        >>>
        >>> # `stride` allows for strided placement of workers across GPUs.
        >>> # For example, if you want to place workers on every second GPU, you can set the stride to 2.
        >>> placement_strided = PackedPlacementStrategy(
        ...     start_gpu_id=0, end_gpu_id=7, stride=2, num_gpus_per_process=2
        ... )
        >>> my_worker_strided = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="my_worker_strided",
        ...     placement_strategy=placement_strided,
        ... )
        >>> my_worker_strided.available_gpus().wait()  # This will run 4 processes, each using 2 GPUs (0,2 1,3 4,6 5,7) of the first node.
        [2, 2, 2, 2]

    """

    def __init__(self, start_gpu_id: int, end_gpu_id: int):
        """Initialize the PlacementStrategy.

        Args:
            start_gpu_id (int): The starting GPU ID for the placement.
            end_gpu_id (int): The ending GPU ID for the placement.

        """
        self._placement_strategy = None
        self._start_gpu_id = start_gpu_id
        self._end_gpu_id = end_gpu_id
        assert start_gpu_id >= 0, (
            f"The start GPU ID {start_gpu_id} must be non-negative."
        )
        assert end_gpu_id >= 0, f"The end GPU ID {end_gpu_id} must be non-negative."
        assert end_gpu_id >= start_gpu_id, (
            f"The end GPU ID {end_gpu_id} must be greater than or equal to the start GPU ID {start_gpu_id}."
        )
        self._num_gpus = end_gpu_id - start_gpu_id + 1
        self._logger = logging.getLogger(name=self.__class__.__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @overload
    def get_placement(
        self, num_gpus_per_node: int, isolate_gpu: bool = True
    ) -> List[Placement]:
        return None
