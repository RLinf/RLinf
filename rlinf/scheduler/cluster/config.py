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

from dataclasses import asdict, dataclass
from typing import Optional

import yaml
from omegaconf import DictConfig, ListConfig

from ..hardware import HardwareConfig, NodeHardwareConfig
from .utils import dataclass_arg_check, parse_rank_config


@dataclass
class NodeGroupEnvConfig:
    """Configuration for software environments for a set of nodes in a node group."""

    node_ranks: list[int]
    """List of node ranks that belong to this node group."""

    env_vars: Optional[list[dict[str, str]]] = None
    """List of environment variables to be set on the nodes."""

    python_interpreter_path: Optional[str] = None
    """Path to the Python interpreter to be used on the nodes."""

    def __post_init__(self):
        """Post-initialization to convert node_ranks from str to list[int] if needed."""
        # Convert env_vars list of dicts to ensure each dict has only one key-value pair
        if self.env_vars is not None:
            env_vars = self.env_vars
            self.env_vars = []
            assert isinstance(env_vars, ListConfig), (
                f"env_vars must be a list of dicts. But got {type(env_vars)}: {env_vars}. Perhaps missing '-' in yaml before each environment variable?"
            )
            for env_var in env_vars:
                assert hasattr(env_var, "keys"), (
                    f"Each node env_var must be a dict in config. But got {type(env_var)}: {env_var}"
                )
                assert len(env_var) == 1, (
                    f"Each node env_var dict must contain exactly one key-value pair. But got: {env_var}"
                )
                env_var_key = str(list(env_var.keys())[0])
                env_var_value = str(list(env_var.values())[0])
                self.env_vars.append({env_var_key: env_var_value})

        if self.python_interpreter_path is not None:
            self.python_interpreter_path = str(self.python_interpreter_path)


@dataclass
class NodeGroupConfig:
    """Configuration for a group of nodes in the cluster with the same label.

    A node group is used to represent multiple nodes with identical hardware configurations.
    """

    label: str
    """Label of the node group. This is not case sensitive."""

    node_ranks: list[int]
    """List of node ranks that belong to this node group."""

    env_configs: Optional[list[NodeGroupEnvConfig]] = None
    """List of environment configurations for the nodes in this group."""

    hardware: Optional[NodeHardwareConfig] = None
    """List of hardware configurations for the nodes."""

    hardware_type: Optional[str] = None
    """Type of hardware for the nodes."""

    ignore_hardware: bool = False
    """Whether to ignore hardware detection on the nodes in this group. If set to True, the nodes will be treated as CPU-only nodes."""

    def __post_init__(self):
        """Post-initialization to convert hardware dicts to their respective dataclass instances."""
        if self.hardware is not None:
            # Arg check
            assert hasattr(self.hardware, "keys"), (
                f"Each hardware yaml config must be a dictionary. But got {type(self.hardware)}: {self.hardware}"
            )
            dataclass_arg_check(
                NodeHardwareConfig,
                self.hardware,
                error_suffix="in cluster node_group hardware yaml config",
            )
            self.hardware = NodeHardwareConfig(**self.hardware)
            assert self.hardware_type is None, "hardware_type should not be specified."
            self.hardware_type = self.hardware.type

        if self.ignore_hardware:
            assert self.hardware is None, (
                "Cannot specify hardware when ignore_hardware is set to True."
            )

        if self.env_configs is not None:
            # Arg check
            env_configs = self.env_configs
            self.env_configs = []
            assert isinstance(env_configs, ListConfig), (
                f"env_configs must be a list of dicts. But got {type(env_configs)}: {env_configs}. Perhaps missing '-' in yaml before node_ranks?"
            )
            for env_config in env_configs:
                assert hasattr(env_config, "keys"), (
                    f"Each node env_configs yaml config must be a dictionary. But got {type(env_config)}: {env_config}"
                )
                dataclass_arg_check(
                    NodeGroupEnvConfig,
                    env_config,
                    error_suffix="in cluster node_group env_configs yaml config",
                )
                env_config = NodeGroupEnvConfig(**env_config)
                self.env_configs.append(env_config)

        self.label = str(self.label)
        from .node import NodeGroupInfo

        assert self.label not in NodeGroupInfo.RESERVED_LABELS, (
            f"Node group label '{self.label}' is reserved by the scheduler. Please choose another label."
        )

    def _validate_env_configs(self):
        """Validate the env_configs to ensure no overlapping node ranks and no duplicate env vars."""
        if self.env_configs is None:
            return
        all_node_ranks = set()
        all_env_var_keys = set()
        for env_config in self.env_configs:
            # Check for overlapping node ranks
            for node_rank in env_config.node_ranks:
                assert node_rank in self.node_ranks, (
                    f"Node rank {node_rank} in env_config must be within node_ranks {self.node_ranks} in node group '{self.label}'."
                )
                assert node_rank not in all_node_ranks, (
                    f"Node rank {node_rank} in env_config is duplicated in node group '{self.label}'."
                )
                all_node_ranks.add(node_rank)

            # Check for duplicate env var keys
            if env_config.env_vars is not None:
                for env_var_dict in env_config.env_vars:
                    for env_var_key in env_var_dict.keys():
                        assert env_var_key not in all_env_var_keys, (
                            f"Environment variable '{env_var_key}' in env_config is duplicated in node group '{self.label}'."
                        )
                        all_env_var_keys.add(env_var_key)


@dataclass
class ClusterConfig:
    """Configuration for the entire cluster.

    The cluster configuration includes the number of nodes, component placements, and node group configurations.

    For component placement format, refer to `rlinf.scheduler.placement.component_placement`. Here is the detailed specification of the node group configuration.

    An example cluster node group configuration in YAML format, which describes a heterogeneous RL training setup with 2 types of accelerators (A800 for training and 4090 for rollout), Franka robot arm for real-world interaction, and node-level placement for agent processes:

    cluster:
      num_nodes: 18
      component_placement:
        actor:
          node_group: a800
          placement: 0-63  # Hardware ranks
        rollout:
          node_group: 4090
          placement: 0-63  # Hardware ranks
        env:
          node_group: franka
          placement: 0-1   # Hardware ranks
        agent:
          node_group: node
          placement: 0-1:0-199,2-3:200-399  # Hardware ranks:Process ranks

      node_groups:
        - label: a800
          node_ranks: 0-7
          env_configs:
            - node_ranks: 0-7
              python_interpreter_path: /opt/venv/openpi/bin/python3
              env_vars:
                - GLOO_SOCKET_IFNAME: "eth0"

        - label: 4090
          node_ranks: 8-15
          env_configs:
            - node_ranks: 8-15
              env_vars:
                - GLOO_SOCKET_IFNAME: "eth1"

        - label: franka
          node_ranks: 16-17
          hardware:
            type: Franka
            configs:
              - robot_ip: "10.10.10.1"
                node_rank: 16
                camera_serials:
                  - "322142001230"
                  - "322142001231"
              - robot_ip: "10.10.10.2"
                node_rank: 17
                camera_serials:
                  - "322142001232"
                  - "322142001233"

        The above configuration specifies:
        - num_nodes: Total of 18 nodes in the cluster.
        - component_placement: Placement of different components (actor, rollout, env, agent) across node groups.
        - node_groups: Three node groups defined:
            - a800: Node ranks 0-7 with A800 GPUs for training (labeled with "a800"). Per-node software environments (python interpreter and env vars) are configured via `env_configs`.

            - 4090: Node ranks 8-15 with 4090 GPUs for rollout (labeled with "4090"). Environment variables are configured via `env_configs`.

            - franka: Node ranks 16-17 with Franka robot arms (labeled with "franka"). Each node has specific hardware configurations including robot IPs and camera serials. Different types of hardware have different configurations; refer to the specific hardware type docs for more information.

        The concrete specification is as follows.

        1. label: A string label for the node group. The label is case sensitive. It is used to reference the node group in component placement configurations. Each label must be unique.

        Labels "cluster" and "node" are reserved by the scheduler and cannot be used. The "node" label can be used to place hardware-agnostic workers like agent workers on specific nodes even without GPUs or any other hardware.

        2. node_ranks: A list or range expression of node ranks (integers) that belong to this node group. Node ranks are zero-indexed ranks specified via RLINF_NODE_RANK environment variable before `ray start` on each node.

        3. env_configs (optional): A list of `NodeGroupEnvConfig` entries that describe software environments for subsets of nodes in the group.

            - Each `env_configs` item has its own `node_ranks`, `env_vars`, and `python_interpreter_path`.
            - `node_ranks` must be a subset of the parent node group's `node_ranks`, and different `env_configs` entries in the same group must not overlap in `node_ranks`.
            - `env_vars` is a list of one-key dicts; each environment variable key must be unique within a node group for a node.
            - `python_interpreter_path` is the interpreter to use on the specified nodes.

        4. hardware (optional): A `NodeHardwareConfig` describing hardware configurations for the nodes in this group.

            - Each hardware configuration's content is determined by the hardware's `type` field. When hardware is specified, one node group can only contain one type of hardware. Different types of hardware need to be placed in different node groups.
            - For a given `(node_rank, hardware_type)` pair, at most one node group may define hardware; otherwise initialization fails.

        5. ignore_hardware (optional): If set to True, hardware detection is disabled for this node group and the nodes are treated as CPU-only nodes, even if accelerators or other hardware are present.

        When hardware is not specified and `ignore_hardware` is False, accelerator hardware is automatically detected and used if it exists on any of the nodes. If no accelerator hardware is found, the node group is treated as CPU-only nodes and the node itself is the hardware resource, where each node has one such resource.

        When using `node_group` in the component_placement, the specified hardware ranks are thus (1) `hardware.type` hardware if it is specified in the node group; (2) automatically detected accelerator hardware if it exists; (3) node itself as hardware resource if no accelerator hardware is found.
        When using the reserved `node` node_group in the component_placement, it is a group with all nodes but no hardware. And so it can be used to perform hardware-agnostic placement of processes on specific nodes.
    """

    num_nodes: int
    """Total number of nodes in the cluster."""

    component_placement: list[dict[str, str]]
    """Placement of each component."""

    node_groups: Optional[list[NodeGroupConfig]] = None
    """List of node group configurations in the cluster."""

    @staticmethod
    def from_dict_cfg(cfg_dict: DictConfig) -> "ClusterConfig":
        """Create a ClusterConfig instance from a dictionary configuration.

        Args:
            cfg_dict (DictConfig): The dictionary configuration.

        Returns:
            ClusterConfig: The created ClusterConfig instance.
        """
        dataclass_arg_check(
            ClusterConfig, cfg_dict, error_suffix="in cluster yaml config"
        )
        return ClusterConfig(**cfg_dict)

    def get_node_labels_by_rank(self, node_rank: int) -> list[str]:
        """Get the node group labels for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            list[str]: The labels of the node group. Empty list if no matching node group is found.
        """
        if self.node_groups is None:
            return []
        labels = []
        for node_group in self.node_groups:
            if node_rank in node_group.node_ranks:
                labels.append(node_group.label)
        return labels

    def get_node_python_interpreter_path_by_rank(self, node_rank: int) -> list[str]:
        """Get all the python interpreter paths for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            list[str]: The python interpreter paths of the node. Empty list if no matching node group is found.
        """
        paths = []
        for node_group in self.node_groups or []:
            if (
                node_group.env_configs is not None
                and node_rank in node_group.node_ranks
            ):
                for env_config in node_group.env_configs:
                    if (
                        node_rank in env_config.node_ranks
                        and env_config.python_interpreter_path is not None
                    ):
                        paths.append(env_config.python_interpreter_path)
        if len(paths) == 0:
            return []
        return paths

    def get_node_hw_configs_by_rank(self, node_rank: int) -> list[HardwareConfig]:
        """Get the hardware configurations for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            list[Any]: The hardware configurations of the node. Empty list if no matching node group is found.
        """
        node_hw_configs: list[HardwareConfig] = []
        if self.node_groups is not None:
            for node_group in self.node_groups:
                if node_rank in node_group.node_ranks:
                    if node_group.hardware is not None:
                        for config in node_group.hardware.configs:
                            if config.node_rank == node_rank:
                                node_hw_configs.append(config)
        return node_hw_configs

    def __post_init__(self):
        """Post-initialization to convert nodes dicts to their respective dataclass instances."""
        if self.node_groups is not None:
            # Arg check
            for node_group in self.node_groups:
                assert hasattr(node_group, "keys"), (
                    f"Each node yaml config must be a dictionary. But got {type(node_group)}: {node_group}"
                )
                dataclass_arg_check(
                    NodeGroupConfig,
                    node_group,
                    error_suffix="in cluster node_groups yaml config",
                )
            self.node_groups = [
                NodeGroupConfig(**node_group) for node_group in self.node_groups
            ]

            # Convert node_ranks from str to list[int] if needed
            for node_group in self.node_groups:
                try:
                    node_group.node_ranks = parse_rank_config(
                        node_group.node_ranks,
                        list(range(self.num_nodes)),
                        "node",
                    )
                except AssertionError as e:
                    raise AssertionError(
                        f"Error parsing node_ranks {node_group.node_ranks} in node group '{node_group.label}'. {str(e)}"
                    )

                if node_group.env_configs is not None:
                    for env_config in node_group.env_configs:
                        try:
                            env_config.node_ranks = parse_rank_config(
                                env_config.node_ranks,
                                node_group.node_ranks,
                                "node",
                            )
                        except AssertionError as e:
                            raise AssertionError(
                                f"Error parsing node_ranks {env_config.node_ranks} in env_config of node group '{node_group.label}'. {str(e)}"
                            )
                        assert set(env_config.node_ranks).issubset(
                            set(node_group.node_ranks)
                        ), (
                            f"node_ranks {env_config.node_ranks} in env_config must be a subset of node_ranks {node_group.node_ranks} in node group '{node_group.label}'."
                        )

                # Can only be validated after the node_ranks are parsed
                node_group._validate_env_configs()

            # Validate hardware node_ranks
            node_hardware_type_map = {}
            for node_group in self.node_groups:
                if node_group.hardware is not None:
                    for cfg in node_group.hardware.configs:
                        assert cfg.node_rank in node_group.node_ranks, (
                            f"node_rank {cfg.node_rank} in hardware config must be within node_ranks {node_group.node_ranks} in node group '{node_group.label}'."
                        )

                        # Ensure that the same hardware type of the same node is not defined in multiple node groups
                        node_hardware_type_key = (
                            cfg.node_rank,
                            node_group.hardware.type,
                        )
                        if node_hardware_type_key in node_hardware_type_map:
                            assert (
                                node_hardware_type_map[node_hardware_type_key]
                                == node_group.label
                            ), (
                                f"Cannot have multiple hardware configs of the same type '{node_group.hardware.type}' for the same node_rank {cfg.node_rank} in different node groups '{node_hardware_type_map[node_hardware_type_key]}' and '{node_group.label}'."
                            )
                        else:
                            node_hardware_type_map[
                                (cfg.node_rank, node_group.hardware.type)
                            ] = node_group.label

        assert type(self.num_nodes) is int and self.num_nodes > 0, (
            f"'num_nodes' must be a positive integer. But got {self.num_nodes} of type {type(self.num_nodes)}."
        )

    def __str__(self) -> str:
        """String representation of the NodeInfo."""
        node_dict = asdict(self)
        node_dict.pop("component_placement", None)
        return yaml.dump(node_dict, sort_keys=False)
