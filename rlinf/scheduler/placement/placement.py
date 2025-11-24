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
from typing import TYPE_CHECKING, overload

from omegaconf import DictConfig

from ..cluster import Cluster, NodeGroupInfo, parse_rank_config
from ..hardware import AcceleratorType

if TYPE_CHECKING:
    from .flexible import FlexiblePlacementStrategy
    from .node import NodePlacementStrategy


@dataclass
class Placement:
    """Class representing the placement of a worker on a specific GPU."""

    rank: int
    """Global rank of the worker in the cluster."""

    cluster_node_rank: int
    """Global node rank in the cluster where the worker is placed."""

    placement_node_rank: int
    """Local rank of the node in the placement."""

    local_accelerator_rank: int
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

    local_hardware_ranks: list[int]
    """The assigned local hardware ranks of the worker"""

    node_group_label: str
    """The label of the node group where the worker is placed."""


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


class ComponentPlacement:
    """Base component placement for parsing cluster.component_placement config.

    The component placement config is defined as either:

    .. code-block:: yaml

        group_name1,group_name2,...: resource_ranks1:process_ranks1, resource_ranks2:process_ranks2,...

    or:

    .. code-block:: yaml

        group_name1,group_name2,...:
            node_group: <node_group_label>
            placement: "resource_ranks1":"process_ranks1", "resource_ranks2":"process_ranks2",...

    A simple example is:

    .. code-block:: yaml

        cluster:
            num_nodes: 1
            actor,inference: 0-7

    which means both the actor and inference groups' process 0-7 evenly occupy accelerator 0 to 7.

    A more complex example is:

    .. code-block:: yaml

        cluster:
        num_nodes: 2
        component_placement:
            actor:
                node_group: a800
                placement: 0-8
            rollout:
                node_group: 4090
                placement: 0-8
            env:
                node_group: robot # Assuming robot hardware type is defined in the node group config
                placement: 0-3:0-7
            agent:
                node_group: node
                placement: 0-1:0-200,2-3:201-511

    which means:

    - The actor group occupies accelerators 0-8 on node group 'a800'.

    - The rollout group occupies accelerators 0-8 on node group '4090'.

    - The env group occupies robot hardware 0-3 on node group 'robot', with each robot hardware shared by 2 processes.

    - The agent group occupies nodes 0-1 for process 0-200, and nodes 2-3 for process 201-511.

    The concrete specifications of the config format are as follows:

    - `resource_ranks` is the ranks of the resources (e.g., GPUs, robots, or nodes) to use for the component(s). resource ranks are by default the accelerator ranks (within the node group if `node_group` is given, counted from 0) if no hardware is specified in the config. If the nodes do not have accelerators, resource ranks are the node ranks. If a hardware is specified in the node group config, the resource ranks are the hardware ranks within the label node group, e.g., for nodes with robotic systems.

      The format of `resource_ranks` is an integer range a-b, which means all ranks from a to b including a and b. For example, 0-3 means rank 0, 1, 2, 3. Alternatively, "all" can be used to specify all resources.

    - `process_ranks` is the ranks of the processes of the component(s), following the same format of `resource_ranks`. The processes will be evenly assigned to the specified resource ranks. For example, 0-3:0-7 means process 0-7 will be evenly assigned to resource ranks 0-3, with 2 processes sharing 1 resource. If the number of processes is smaller than the number of resources, it means one process occupy multiple resources. If `process_ranks` is not specified, each process will be assigned to one resource rank in order. For example, 0-4 means process 0-4 will be assigned to resource ranks 0-4 respectively.

      Fancier syntax mixing the two formats is also supported, e.g., 0-1:0-3,3-5,7-10:7-14, which means process 0-3 will be evenly assigned to resource ranks 0-1, process 4-6 will be assigned to resource ranks 3-5 (implicitly inferred by the scheduler) respectively, and process 7-14 will be evenly assigned to resource ranks 7-10. Note that even if the process ranks are not specified, they are assumed to be continuous from 0 to N-1, where N is the total number of processes. Failure to follow this rule will raise an assertion error.

    - For the second format, the `node_group` label is the label defined in cluster.node_groups.label, which is optional. If not specified, all nodes in the cluster are used. A `node` label is reserved by the scheduler for allocating on node ranks only (no accelerators or other hardware).
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Parsing component placement configuration.

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
            cluster (Cluster): The cluster to use for placement.
        """
        self._config = config
        assert hasattr(config, "cluster"), (
            f"Cluster config must be provided for component placement. But got: {config}"
        )
        assert hasattr(config.cluster, "component_placement"), (
            f"component_placement must be provided in cluster config for component placement. But got: {config.cluster}"
        )
        self._placement_config: DictConfig = config.cluster.component_placement
        self._placement_mode = None

        self._placements: dict[str, PlacementStrategy] = {}
        self._components: list[str] = []
        self._component_world_size: dict[str, int] = {}
        self._component_rank_map: dict[str, dict[tuple[int], list[int]]] = {}

        for component_names in self._placement_config.keys():
            component_placement = self._placement_config[component_names]
            if not isinstance(component_placement, str) and not isinstance(
                component_placement, DictConfig
            ):
                component_placement = str(component_placement)
            component_names = str(component_names)
            component_names = component_names.split(",")
            component_names = [c.strip() for c in component_names]
            self._parse_component_placement(
                cluster, component_placement, component_names
            )
            self._components.extend(component_names)

        assert len(self._components) == len(set(self._components)), (
            f"Duplicate component names found in component placement config: {self._placement_config}. Component names must be unique."
        )

    def _parse_component_placement(
        self,
        cluster: Cluster,
        component_placement: str | DictConfig,
        component_names: list[str],
    ) -> PlacementStrategy:
        """Parse the component placement configuration into a PlacementStrategy.

        Args:
            cluster (Cluster): The cluster to use for placement.
            component_placement (str | DictConfig): The component placement configuration.
            component_names (list[str]): The names of the components to place.
        """
        assert isinstance(component_placement, (str, DictConfig)), (
            f"component_placement must be either a string or a DictConfig. But got: {type(component_placement)}: {component_placement}"
        )
        # Format (1) group_name1,group_name2,...: resource_ranks:process_ranks
        if isinstance(component_placement, str):
            node_group = cluster.get_node_group()
            rank_map_str = component_placement
        # Format (2) group_name1,group_name2,...:
        #         node_group: <node_group_label>
        #         placement: resource_ranks:process_ranks
        elif isinstance(component_placement, DictConfig):
            if hasattr(component_placement, "node_group"):
                node_group_label = component_placement.node_group
                node_group = cluster.get_node_group(node_group_label)
            else:
                node_group = cluster.get_node_group()
            assert hasattr(component_placement, "placement"), (
                f"placement must be specified in component_placement config: {component_placement}"
            )
            rank_map_str = component_placement.placement

        assert node_group is not None, (
            f'Node group not found for components {component_names} with label "{node_group_label}".'
        )
        rank_map = self._parse_rank_map(rank_map_str, node_group)
        if node_group.hardware_type is None:
            placement_strategy = self._gen_node_placement(
                rank_map, node_group, component_names
            )
        else:
            placement_strategy = self._gen_resource_placement(
                rank_map, node_group, component_names
            )

        num_processes = sum(len(process_ranks) for process_ranks in rank_map.values())
        for component_name in component_names:
            assert component_name not in self._placements, (
                f"Component {component_name} has multiple placements defined."
            )
            self._placements[component_name] = placement_strategy
            self._component_world_size[component_name] = num_processes
            self._component_rank_map[component_name] = rank_map

    def _parse_rank_map(
        self, rank_map_str: str, node_group: NodeGroupInfo
    ) -> dict[list[int], list[int]]:
        """Parse the rank map string into a dictionary mapping resource ranks to process ranks.

        The string is a comma separated string, where each part is the format: resource_ranks:process_ranks. process_ranks is optional. If not specified, process_ranks will be counted from 0 to the number of resource_ranks - 1.
        """
        rank_map_str = str(rank_map_str)
        rank_map: dict[tuple[int], list[int]] = {}
        parsed_resource_ranks: list[int] = []
        parsed_process_ranks: list[int] = []

        rank_map_parts = rank_map_str.strip().split(",")
        for rank_map_part in rank_map_parts:
            rank_map_part = rank_map_part.strip()
            if rank_map_part == "":
                continue
            rank_part = rank_map_part.split(":")
            assert 1 <= len(rank_part) <= 2, (
                f"Invalid rank map string: {rank_map_part} in placement config: {rank_map_str}. Expected format: resource_ranks:process_ranks"
            )

            # Resource ranks parsing
            resource_ranks_str = rank_part[0].strip()
            try:
                resource_ranks = parse_rank_config(
                    resource_ranks_str,
                    list(range(node_group.hardware_resource_count)),
                    node_group.hardware_type,
                )
            except AssertionError as e:
                raise AssertionError(
                    f"Error parsing resource ranks in placement string: {rank_map_part} of placement config: {rank_map_str}. {str(e)}"
                )

            # Resource ranks validation
            assert resource_ranks, (
                f"No valid resource ranks found in placement string: {rank_map_part}."
            )
            assert set(resource_ranks).isdisjoint(set(parsed_resource_ranks)), (
                f"Duplicate resource ranks found in placement string: {rank_map_str}. Resource ranks must be unique."
            )
            assert (
                resource_ranks[0] > parsed_resource_ranks[-1]
                if parsed_resource_ranks
                else True
            ), (
                f"Resource ranks must be in ascending order in placement string: {rank_map_str}."
            )
            parsed_resource_ranks.extend(resource_ranks)

            # Process ranks parsing
            if len(rank_part) == 2:
                process_ranks_str = rank_part[1].strip()
                assert process_ranks_str != "all", (
                    f"The latter part of a ':' separated placement string (i.e., the process ranks) cannot be 'all': {rank_map_str}."
                )
                process_ranks = parse_rank_config(process_ranks_str)
            else:
                process_ranks = list(range(len(resource_ranks)))
                process_ranks = (
                    [r + max(parsed_process_ranks) + 1 for r in process_ranks]
                    if parsed_process_ranks
                    else process_ranks
                )

            # Process ranks validation
            assert process_ranks, (
                f"No valid process ranks found in placement string: {rank_map_part}."
            )
            assert set(process_ranks).isdisjoint(set(parsed_process_ranks)), (
                f"Duplicate process ranks found in placement string: {rank_map_str}. Process ranks must be unique."
            )
            assert (
                process_ranks[0] == parsed_process_ranks[-1] + 1
                if parsed_process_ranks
                else True
            ), (
                f"Process ranks must be in ascending order and continuous in placement string: {rank_map_str}."
            )
            assert process_ranks == list(
                range(process_ranks[0], process_ranks[0] + len(process_ranks))
            ), f"Process ranks must be continuous in placement string: {rank_map_str}."
            parsed_process_ranks.extend(process_ranks)

            # Resource ranks and process ranks validation
            assert (
                len(process_ranks) % len(resource_ranks) == 0
                or len(resource_ranks) % len(process_ranks) == 0
            ), (
                f"The number of process ranks {len(process_ranks)} must be divisible by the number of resource ranks {len(resource_ranks)} in placement string: {rank_map_part} or the number of resource ranks must be divisible by the number of process ranks."
            )

            rank_map[tuple(resource_ranks)] = process_ranks

        return rank_map

    def _rank_map_to_process_resources_map(
        self, rank_map: dict[list[int], list[int]]
    ) -> dict[int, list[int]]:
        """Convert the rank map to a process to resource mapping.

        Args:
            rank_map (dict[list[int], list[int]]): The rank map.

        Returns:
            dict[int, list[int]]: The process to resource mapping.
        """
        process_resources_map: dict[int, list[int]] = {}
        for resource_ranks, process_ranks in rank_map.items():
            if len(resource_ranks) >= len(process_ranks):
                # More resources than processes, one process occupy multiple resources
                resources_per_process = len(resource_ranks) // len(process_ranks)
                for i, process_rank in enumerate(process_ranks):
                    process_resources_map[process_rank] = list(
                        resource_ranks[
                            i * resources_per_process : (i + 1) * resources_per_process
                        ]
                    )
            else:
                # More processes than resources, multiple processes share one resource
                processes_per_resource = len(process_ranks) // len(resource_ranks)
                for i, resource_rank in enumerate(resource_ranks):
                    for j in range(processes_per_resource):
                        process_rank = process_ranks[i * processes_per_resource + j]
                        if process_rank not in process_resources_map:
                            process_resources_map[process_rank] = []
                        process_resources_map[process_rank].append(resource_rank)
        return process_resources_map

    def _gen_node_placement(
        self,
        rank_map: dict[tuple[int], list[int]],
        node_group: NodeGroupInfo,
        component_names: list[str],
    ) -> "NodePlacementStrategy":
        from .node import NodePlacementStrategy

        process_resources_map = self._rank_map_to_process_resources_map(rank_map)
        node_rank_list = []
        for process_rank in sorted(process_resources_map.keys()):
            node_ranks = process_resources_map[process_rank]
            assert len(node_ranks) == 1, (
                f"Node placement is used for components {component_names} because there is no hardware in the selected nodes {node_group.nodes}. However, the number of processes {len(process_resources_map.keys())} of the components is smaller than the number of available nodes, which is impossible as one process cannot run across multiple nodes."
            )
            node_rank_list.append(node_ranks[0])

        try:
            return NodePlacementStrategy(node_rank_list, node_group.label)
        except AssertionError as e:
            raise AssertionError(
                f"Error in component placement for components {component_names}. Allocated node ranks for each process: {process_resources_map}. {str(e)}"
            )

    def _gen_resource_placement(
        self,
        rank_map: dict[tuple[int], list[int]],
        node_group: NodeGroupInfo,
        component_names: list[str],
    ) -> "FlexiblePlacementStrategy":
        from .flexible import FlexiblePlacementStrategy

        process_resources_map = self._rank_map_to_process_resources_map(rank_map)
        resource_ranks_list = []
        for process_rank in sorted(process_resources_map.keys()):
            resource_ranks = process_resources_map[process_rank]
            resource_ranks_list.append(resource_ranks)
        try:
            return FlexiblePlacementStrategy(resource_ranks_list, node_group.label)
        except AssertionError as e:
            raise AssertionError(
                f"Error in component placement for components {component_names}. Allocated hardware ranks for each process: {process_resources_map} for hardware type {node_group.hardware_type} and node group {node_group.label}. {str(e)}"
            )

    @property
    def placement_mode(self):
        """Get the placement mode for the component.

        Returns:
            PlacementMode: The placement mode for the component.
        """
        return self._placement_mode

    @property
    def components(self) -> list[str]:
        """Get the list of components defined in the placement.

        Returns:
            list[str]: The list of component names.
        """
        return self._components

    def get_hardware_ranks(self, component_name: str):
        """Get the hardware count for a specific component.

        Args:
            component_name (str): The name of the component.

        Returns:
            list[int]: The hardware ranks for the specified component.
        """
        assert component_name in self._component_rank_map, (
            f"Unknown component name: {component_name}"
        )
        rank_map = self._component_rank_map[component_name]
        hardware_ranks = []
        for resource_ranks in rank_map.keys():
            hardware_ranks.extend(resource_ranks)
        return hardware_ranks

    def get_world_size(self, component_name: str):
        """Get the world size for a specific component.

        Args:
            component_name (str): The name of the component.

        Returns:
            int: The world size for the specified component.
        """
        assert component_name in self._component_world_size, (
            f"Unknown component name: {component_name}"
        )
        return self._component_world_size[component_name]

    def get_strategy(self, component_name: str):
        """Get the placement strategy for a component based on the configuration.

        Args:
            component_name (str): The name of the component to retrieve the placement strategy for.

        Returns:
            PackedPlacementStrategy: The placement strategy for the specified component.
        """
        assert component_name in self._placements, (
            f"Component {component_name} does not exist in {type(self)} with placement mode {self._placement_mode}"
        )
        return self._placements[component_name]
