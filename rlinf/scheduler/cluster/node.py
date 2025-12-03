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

import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Optional

import ray
import ray.actor
import ray.util.scheduling_strategies
import yaml

from ..hardware import (
    Accelerator,
    AcceleratorType,
    Hardware,
    HardwareInfo,
    HardwareResource,
)
from .config import ClusterConfig, NodeGroupEnvConfig


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_labels: list[str]
    """Labels of the node, corresponding to the node group label in the cluster configuration."""

    node_rank: int
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    num_cpus: int
    """Number of CPUs available on the node."""

    python_interpreter_path: str
    """Path to the Python interpreter to be used on the node."""

    default_env_vars: dict[str, str]
    """Default environment variables on the node, which are the env vars set before ray start."""

    env_vars: dict[str, str]
    """Environment variables set on the node by the scheduler."""

    hardware_resources: list[HardwareResource] = field(default_factory=list)
    """List of hardware resources available on the node."""

    @property
    def num_accelerators(self) -> int:
        """Get the number of accelerators on the node."""
        return self.get_hw_resource_count(Accelerator.HW_TYPE)

    @property
    def accelerator_type(self) -> str:
        """Get the type of accelerators on the node."""
        for resource in self.hardware_resources:
            if resource.type == Accelerator.HW_TYPE and resource.count > 0:
                return Accelerator.get_accelerator_type_from_model(
                    resource.infos[0].model
                )
        return AcceleratorType.NO_ACCEL.value

    def get_hw_resource_count(self, hw_type: Optional[str]) -> int:
        """Get the count of a specific hardware resource type."""
        if hw_type is None:
            return 0
        for resource in self.hardware_resources:
            if resource.type == hw_type:
                return resource.count
        return 0

    def __str__(self) -> str:
        """String representation of the NodeInfo."""
        node_dict = asdict(self)
        node_dict.pop("default_env_vars", None)
        node_dict.pop("env_vars", None)
        return yaml.dump(node_dict, sort_keys=False)


@dataclass
class NodeGroupInfo:
    """Information about a group of nodes in the cluster."""

    label: str
    """Label of the node group."""

    nodes: list[NodeInfo]
    """List of nodes in the node group."""

    hardware_type: Optional[str] = None
    """Type of hardware of the node group. Can only contain one type of hardware."""

    ignore_hardware: bool = False
    """Whether to ignore hardware detection on the nodes in this group. If set to True, the nodes will be treated as CPU-only nodes."""

    env_configs: Optional[list[NodeGroupEnvConfig]] = None
    """Environment configurations for the node group."""

    DEFAULT_GROUP_LABEL: ClassVar[str] = "cluster"

    NODE_PLACEMENT_GROUP_LABEL: ClassVar[str] = "node"

    RESERVED_LABELS: ClassVar[list[str]] = ["cluster", "node"]

    @property
    def node_ranks(self) -> list[int]:
        """Get the list of node ranks in the node group."""
        return [node.node_rank for node in self.nodes]

    @property
    def hardware_resource_count(self) -> int:
        """Get the total count of the hardware resource type on the node group."""
        if self.hardware_type is None:
            # If hardware_type is not specified, the node itself is the hardware resource
            return len(self.nodes)
        total_count = 0
        for node in self.nodes:
            total_count += node.get_hw_resource_count(self.hardware_type)
        return total_count

    @property
    def local_hardware_infos(self) -> list[HardwareInfo]:
        """Get the hardware infos for each node in the group."""
        infos: list[HardwareInfo] = []
        for node in self.nodes:
            for resource in node.hardware_resources:
                if resource.type == self.hardware_type:
                    infos.extend(resource.infos)
        return infos

    @property
    def local_hardware_ranks(self) -> list[list[int]]:
        """Get the hardware ranks for each node in the group."""
        start_rank = 0
        hardware_ranks = []
        for node in self.nodes:
            hardware_count = node.get_hw_resource_count(self.hardware_type)
            hardware_ranks.append(list(range(start_rank, start_rank + hardware_count)))
            start_rank += hardware_count
        return hardware_ranks

    def group_ranks_to_global_ranks(self, group_ranks: list[int]):
        """Group-local ranks to global node ranks."""
        try:
            return [self.nodes[rank].node_rank for rank in group_ranks]
        except IndexError:
            raise IndexError(
                f"Group rank out of range. Node group '{self.label}' has {len(self.nodes)} nodes, but got group ranks: {group_ranks}"
            )

    def get_node_by_hardware_rank(self, hardware_rank: int):
        """Acquire node with the hardware rank in the node group.

        Args:
            hardware_rank (int): The hardware rank in the node group.

        Returns:
            NodeInfo: The node information that contains the hardware with the specified rank.
        """
        for i, node_hardware_ranks in enumerate(self.local_hardware_ranks):
            if hardware_rank in node_hardware_ranks:
                return self.nodes[i]

    def get_local_hardware_rank(self, hardware_rank: int):
        """Convert a global hardware rank in the node group to the local rank in its node.

        Args:
            hardware_rank (int): The global hardware rank in the node group.

        Returns:
            int: The local hardware rank in the node.
        """
        for node_hardware_ranks in self.local_hardware_ranks:
            if hardware_rank in node_hardware_ranks:
                return node_hardware_ranks.index(hardware_rank)

    def get_node_env_vars(self, node_rank: int) -> dict[str, str]:
        """Get the environment variables for a specific node in the group.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            dict[str, str]: The environment variables for the node.
        """
        env_vars = {}
        node_env_keys = []
        for env_config in self.env_configs or []:
            if node_rank in env_config.node_ranks and env_config.env_vars is not None:
                for env_var_dict in env_config.env_vars:
                    env_vars.update(env_var_dict)
                    assert set(env_var_dict.keys()).isdisjoint(node_env_keys), (
                        f"Environment variables {set(env_var_dict.keys()).intersection(set(node_env_keys))} in cluster configuration for node group '{self.label}' have been set in other env_configs of the same node group. Please ensure that environment variables are not duplicated across env_configs."
                    )
                    node_env_keys.extend(env_var_dict.keys())
                return env_vars
        return {}

    def get_node_python_interpreter_path(self, node_rank: int) -> Optional[str]:
        """Get the Python interpreter path for a specific node in the group.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            Optional[str]: The Python interpreter path for the node. None if not specified.
        """
        paths = []
        for env_config in self.env_configs or []:
            if (
                node_rank in env_config.node_ranks
                and env_config.python_interpreter_path is not None
            ):
                paths.append(env_config.python_interpreter_path)
        if len(paths) == 0:
            return None
        if len(set(paths)) > 1:
            raise ValueError(
                f"Conflicting Python interpreter paths {paths} found for node rank {node_rank} in node group '{self.label}'. Please ensure that only one Python interpreter path is specified for each node."
            )
        return paths[0]

    def __post_init__(self):
        """Post-initialization to validate the node group information."""
        # If hardware_type is not specified, set it to the default hardware type if available
        # Otherwise, leave it as None
        if self.hardware_type is None and not self.ignore_hardware:
            hw_resources: list[HardwareResource] = []
            for node in self.nodes:
                hw_resources.extend(node.hardware_resources)
            hw_types = {resource.type for resource in hw_resources}

            assert Hardware.DEFAULT_HW_TYPE is not None, (
                "Default hardware type is not set in HardwareEnumerationPolicy."
            )
            if Hardware.DEFAULT_HW_TYPE in hw_types:
                self.hardware_type = Hardware.DEFAULT_HW_TYPE

    def __str__(self) -> str:
        """String representation of the NodeGroupInfo."""
        group_dict = asdict(self)
        group_dict["nodes"] = ",".join([str(node.node_rank) for node in self.nodes])
        return yaml.dump(group_dict, sort_keys=False)


class NodeProbe:
    """Remote probe to get node hardware and environment information.

    This class launches one _RemoteNodeProbe actor on each node in the Ray cluster to collect hardware and environment information.
    """

    def __init__(self, cluster_num_nodes: int, cluster_cfg: Optional[ClusterConfig]):
        """Launch the HardwareEnumerator on the specified nodes."""
        from .cluster import Cluster

        assert ray.is_initialized(), (
            "Ray must be initialized before creating HardwareEnumerator."
        )

        self._probes: list[ray.actor.ActorHandle] = []
        self._nodes: list[NodeInfo] = []
        self._cluster_cfg = cluster_cfg

        node_infos = Cluster.get_alive_nodes()
        num_nodes = len(node_infos)
        for node_info in node_infos:
            node_ray_id = node_info["NodeID"]
            probe = _RemoteNodeProbe.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ray_id, soft=False
                ),
                name=f"NodeProbe_{node_ray_id}",
            ).remote(node_info, num_nodes, cluster_cfg, sys.executable)
            self._probes.append(probe)

        handles = []
        for probe in self._probes:
            handles.append(probe.get_node_info.remote())
        self._nodes = ray.get(handles)

        self._sort_nodes(cluster_num_nodes)
        self._configure_node_envs()

        # Node groups
        self._node_groups: list[NodeGroupInfo] = []
        if self._cluster_cfg is not None and self._cluster_cfg.node_groups is not None:
            for node_group_cfg in self._cluster_cfg.node_groups:
                group_nodes: list[NodeInfo] = []
                for node in self._nodes:
                    if node.node_rank in node_group_cfg.node_ranks:
                        group_nodes.append(node)
                if len(group_nodes) > 0:
                    node_group_info = NodeGroupInfo(
                        label=node_group_cfg.label,
                        nodes=group_nodes,
                        hardware_type=node_group_cfg.hardware_type,
                        env_configs=node_group_cfg.env_configs,
                        ignore_hardware=node_group_cfg.ignore_hardware,
                    )
                    self._node_groups.append(node_group_info)

        # Default node group including all nodes
        self._node_groups.append(
            NodeGroupInfo(label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=self._nodes)
        )
        # Reserved "node" label for node-level placement
        node_group = NodeGroupInfo(
            label=NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL,
            nodes=self._nodes,
            ignore_hardware=True,
        )
        self._node_groups.append(node_group)

        assert len({group.label for group in self._node_groups}) == len(
            self._node_groups
        ), (
            f"Node group labels must be unique, but got: {[group.label for group in self._node_groups]}"
        )

    @property
    def nodes(self):
        """Get the list of node information.

        Returns:
            list[NodeInfo]: List of node information.
        """
        return self._nodes

    @property
    def node_groups(self):
        """Get the list of node groups.

        Returns:
            list[NodeGroupInfo]: List of node groups.
        """
        return self._node_groups

    @property
    def head_node(self):
        """Get the head node information, which is the node that initializes the cluster.

        Returns:
            NodeInfo: Head node information.
        """
        current_id = ray.get_runtime_context().get_node_id()
        head_node = next(
            (node for node in self._nodes if node.ray_id == current_id), None
        )
        assert head_node is not None, (
            f"Head node with Ray ID {current_id} not found in the cluster nodes: {[node.ray_id for node in self._nodes]}"
        )
        return head_node

    def _configure_node_envs(self):
        """Configure each node's environments based on the cluster configuration.

        The environment variables follow the following precedence, with the later ones overriding the previous ones if set:
        1. Default environment variables on the node (set before ray start).
        2. Environment variables set between ray start and RLinf initialization on the head node (usually via bash scripts). These env vars are likely set by users intended to configure all nodes in the cluster.
        3. The env_vars field in the ClusterConfig, which are set in yaml config files to configure each node in the cluster. This is set in Cluster.allocate.
        """
        # Overwrite the the head node's python interpreter path as the current interpreter unless specified in the cluster config
        self.head_node.python_interpreter_path = sys.executable

        # First find env vars set between ray start and RLinf initialization on the head node
        head_node_default_env_vars = self.head_node.default_env_vars
        current_env_vars = os.environ
        modified_env_vars = {}
        for key, value in current_env_vars.items():
            if (
                key not in head_node_default_env_vars
                or head_node_default_env_vars[key] != value
            ):
                modified_env_vars[key] = value

        for node in self._nodes:
            # Start with default env vars on the node
            node.env_vars = node.default_env_vars.copy()

            # Update with modified env vars on the head node
            node.env_vars.update(modified_env_vars)

    def _sort_nodes(self, cluster_num_nodes: int):
        """Sort the node info list by node rank if available, otherwise by accelerator type and IP."""
        from .cluster import Cluster, ClusterEnvVar

        # Sort the node info list by node rank if available
        if all(node_info.node_rank != -1 for node_info in self._nodes):
            # NODE_RANK should be larger than 0
            assert all(node_info.node_rank >= 0 for node_info in self._nodes), (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should not be smaller than 0, but got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank < 0]}"
            )

            # NODE_RANK should be smaller than the number of nodes
            assert all(
                node_info.node_rank < len(self._nodes) for node_info in self._nodes
            ), (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should be smaller than the number of nodes {len(self._nodes)}, but got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank >= len(self._nodes)]}"
            )

            self._nodes.sort(key=lambda x: x.node_rank)

        else:
            # Either all nodes set NODE_RANK, or none of them should have.
            assert all(node_info.node_rank == -1 for node_info in self._nodes), (
                f"Either all nodes set {Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)}, or none of them should have. But got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank != -1]}"
            )

            # NODE_RANK not set, sort first by accelerator type, then by IP
            nodes_group_by_accel: dict[str, list[NodeInfo]] = {}
            for node in self._nodes:
                accel_name = node.accelerator_type
                nodes_group_by_accel.setdefault(accel_name, [])
                nodes_group_by_accel[accel_name].append(node)
            for accel_name in nodes_group_by_accel.keys():
                nodes_group_by_accel[accel_name].sort(key=lambda x: x.node_ip)
            self._nodes = [
                node for nodes in nodes_group_by_accel.values() for node in nodes
            ]

            node_rank = 0
            for node in self._nodes:
                node.node_rank = node_rank
                node_rank += 1

        # Handle num_nodes configuration mismatch with actual node number
        if len(self._nodes) > cluster_num_nodes:
            warnings.warn(
                f"The cluster is initialized with {cluster_num_nodes} nodes, but detected {len(self._nodes)} nodes have joined the ray cluster. So only the first {cluster_num_nodes} nodes are used."
            )
            self._nodes = self._nodes[:cluster_num_nodes]

        # Node ranks should be unique, continuous from 0 to num_nodes - 1
        node_ranks = [node_info.node_rank for node_info in self._nodes]
        assert sorted(node_ranks) == list(range(len(self._nodes))), (
            f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should be unique and continuous from 0 to {len(self._nodes) - 1}, but got: {node_ranks}"
        )


@ray.remote
class _RemoteNodeProbe:
    """Remote Ray actor that collect information on a node."""

    def __init__(
        self,
        node_info: dict[str, str],
        num_nodes: int,
        cluster_cfg: Optional[ClusterConfig],
        head_python_interpreter: str,
    ):
        from .cluster import Cluster, ClusterEnvVar

        # Node rank
        try:
            node_rank = int(Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK, -1))
        except ValueError:
            raise ValueError(
                f"Invalid NODE_RANK value: {Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK)}. Must be an integer."
            )
        if num_nodes == 1:
            node_rank = 0

        # Node label
        node_labels = []
        if cluster_cfg is not None and cluster_cfg.node_groups is not None:
            assert node_rank != -1, (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} must be set when there are more than one nodes are connected in Ray and cluster's nodes configuration is provided."
            )
            node_labels = cluster_cfg.get_node_labels_by_rank(node_rank)

        # Node hardware resources
        node_hw_configs = []
        if cluster_cfg is not None:
            node_hw_configs = cluster_cfg.get_node_hw_configs_by_rank(node_rank)
        hardware_resources: list[HardwareResource] = []
        for policy in Hardware.policy_registry:
            hw_resource = policy.enumerate(node_rank, node_hw_configs)
            if hw_resource is not None and hw_resource.count > 0:
                hardware_resources.append(hw_resource)

        # Python interpreter path
        if sys.executable != head_python_interpreter:
            warnings.warn(
                f"Python interpreter used to launch Ray on node with IP {node_info['NodeManagerAddress']} is different from that on the head node {head_python_interpreter}. Keep using the current interpreter {sys.executable} on this node."
            )
        python_interpreter_path = sys.executable
        if cluster_cfg is not None:
            cfg_python_interpreter_paths = (
                cluster_cfg.get_node_python_interpreter_path_by_rank(node_rank)
            )
            for path in cfg_python_interpreter_paths:
                assert os.path.exists(path), (
                    f"Python interpreter path {path} does not exist on node with node rank {node_rank}. Please check your cluster configuration."
                )

        self._node_info = NodeInfo(
            node_labels=node_labels,
            node_rank=node_rank,
            ray_id=node_info["NodeID"],
            node_ip=node_info["NodeManagerAddress"],
            num_cpus=int(node_info["Resources"].get("CPU", 0)),
            python_interpreter_path=python_interpreter_path,
            default_env_vars=os.environ.copy(),
            env_vars=os.environ.copy(),
            hardware_resources=hardware_resources,
        )

    def get_node_info(self):
        """Get the node information.

        Returns:
            NodeInfo: The node information.
        """
        return self._node_info
