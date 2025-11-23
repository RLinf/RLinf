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
from typing import overload

from ..accelerator import AcceleratorType
from ..cluster import Cluster


@dataclass
class Placement:
    """Class representing the placement of a worker on a specific GPU."""

    rank: int
    """Global rank of the worker in the cluster."""

    node_id: int
    """Node ID where the worker is placed."""

    node_rank: int
    """Rank of the node in the cluster."""

    local_accelerator_id: int
    """Local GPU ID on the node."""

    accelerator_type: AcceleratorType
    """Type of accelerators on the node."""

    local_rank: int
    """Local rank of the worker on the node."""

    local_world_size: int
    """Local world size (number of workers) on the node."""

    visible_accelerators: list[str]
    """List of CUDA visible devices for the worker."""

    isolate_accelerator: bool
    """Flag to indicate if the local rank should be set to zero. This is useful for workers that require multiple GPUs."""


class PlacementStrategy:
    """Base class for placement strategies."""

    def __init__(self):
        """Initialize the PlacementStrategy."""
        self._placement_strategy = None
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
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        return None
