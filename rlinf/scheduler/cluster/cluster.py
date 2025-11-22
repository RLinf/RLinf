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
import os
import signal
import sys
import time
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Optional

import ray
import ray.util.scheduling_strategies
from omegaconf import DictConfig
from packaging import version as vs
from ray._private import ray_logging
from ray.actor import ActorHandle
from ray.util.state import list_actors

from .config import ClusterConfig
from .node import NodeGroupInfo, NodeProbe

ray_version = version("ray")
assert vs.parse(ray_version) >= vs.parse("2.47.0"), (
    "Ray version 2.47.0 or higher is required. Run pip install ray[default]==2.47.0"
)

if TYPE_CHECKING:
    from ..worker import Worker


class ClusterEnvVar(str, Enum):
    """Scheduler environment variables. All env vars are prefixed with {Cluster.SYS_NAME}_ in usage."""

    CATCH_FAILURE = "CATCH_FAILURE"
    """Whether to catch failures in workers to avoid exiting the main process."""

    LOG_LEVEL = "LOG_LEVEL"
    """Logging level for the cluster and workers."""

    TIMEOUT = "TIMEOUT"
    """Timeout for the all inter-worker communications."""

    NODE_RANK = "NODE_RANK"
    """Rank of each node in the cluster."""

    COMM_NET_DEVICES = "COMM_NET_DEVICES"
    """Network devices to use for inter-node communication."""


class Cluster:
    """A singleton class that manages the cluster resources for Ray workers."""

    SYS_NAME = "RLinf"
    NAMESPACE = SYS_NAME
    LOGGING_LEVEL = os.getenv(
        f"{SYS_NAME.upper()}_{ClusterEnvVar.LOG_LEVEL.value}", "INFO"
    ).upper()
    TIMEOUT_WARN_TIME = 60000
    DEFAULT_SYS_ENV_VAR = {
        ClusterEnvVar.CATCH_FAILURE: "0",
        ClusterEnvVar.LOG_LEVEL: "INFO",
        ClusterEnvVar.TIMEOUT: "180",
        ClusterEnvVar.NODE_RANK: None,
        ClusterEnvVar.COMM_NET_DEVICES: None,
    }

    @classmethod
    def find_free_port(cls):
        """Find a free port on the node."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @classmethod
    def has_initialized(cls):
        """Check if the cluster has been initialized."""
        return hasattr(cls, "_instance") and cls._instance is not None

    def __new__(cls, *args, **kwargs):  # noqa D417
        """Create a singleton class that manages the cluster resources for Ray workers."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._has_initialized = False
        return cls._instance

    def __init__(
        self, num_nodes: Optional[int] = None, cluster_cfg: Optional[DictConfig] = None
    ):
        """Initialize the cluster.

        Args:
            num_nodes (int): The number of nodes in the cluster. When you wish to acquire the cluster instance in a processes other than the main driver process, do not pass this argument. Instead, use the `Cluster()` constructor without arguments.
            cluster_cfg (Optional[DictConfig]): The cluster's configuration dictionary. If set, num_nodes will be ignored and inferred from the config.
        """
        if self._has_initialized:
            return
        if num_nodes is not None or cluster_cfg is not None:
            self._ray_instance_count = 0
            self._init_and_launch_managers(num_nodes, cluster_cfg)
        else:
            self._init_from_existing_managers()
        self._has_initialized = True

    def _init_and_launch_managers(
        self, num_nodes: int, cluster_cfg: Optional[DictConfig]
    ):
        # Add logger
        self._logger = logging.getLogger(Cluster.SYS_NAME)
        self._logger.setLevel(Cluster.LOGGING_LEVEL)
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

        if ray.is_initialized():
            if self._ray_instance_count > 0:
                # For reinit Ray to switch namespace
                ray.shutdown()
            else:
                # Initializing Ray before us interferes with the namespace and logging level settings.
                raise RuntimeError(
                    "You have initialized Ray before creating the Cluster instance. This may be due to calling ray.init or creating certain Ray objects like Ray Queue before instantiating the Cluster class. Please ensure that the Cluster class is instantiated before Ray is initialized because it will interfere with our Ray namespace and logging settings."
                )

        # NOTE: Add os.environ variables to the worker environment.
        # When ray cluster has been started via `ray start` before running the Python script, ray will only capture the environment variables exported before `ray start` and ignore all subsequently exported environment variables.
        # To handle this, we need to manually pass the environment variables to Ray when initializing the cluster.
        # Any env vars conflicting with Worker env vars will be overwritten by Worker.
        if "RAY_DEDUP_LOGS" not in os.environ:
            # Default disabling deduplication of logs to ensure all logs are printed.
            ray_logging.RAY_DEDUP_LOGS = 0

        # Cluster configurations
        self._cluster_cfg = (
            ClusterConfig.from_dict_cfg(cluster_cfg) if cluster_cfg else None
        )
        if (
            self._cluster_cfg is not None
            and num_nodes is not None
            and self._cluster_cfg.num_nodes != num_nodes
        ):
            raise ValueError(
                f"num_nodes ({num_nodes}) passed in Cluster init does not match the number of nodes in configuration ({self._cluster_cfg.num_nodes}). Please ensure they are consistent."
            )
        self._num_nodes = (
            self._cluster_cfg.num_nodes if self._cluster_cfg is not None else num_nodes
        )
        assert self._num_nodes > 0, "num_nodes must be greater than 0."

        try:
            # First try to connect to an existing Ray cluster
            ray.init(
                address="auto",
                logging_level=Cluster.LOGGING_LEVEL,
                namespace=Cluster.NAMESPACE,
            )
        except ConnectionError:
            ray.init(
                logging_level=Cluster.LOGGING_LEVEL,
                namespace=Cluster.NAMESPACE,
            )

        # Wait for the cluster to be ready
        while len(Cluster.get_alive_nodes()) < self._num_nodes:
            self._logger.warning(
                f"Waiting for {self._num_nodes} nodes to be ready, currently {len(Cluster.get_alive_nodes())} nodes available."
            )
            time.sleep(1)

        # Get node info
        self._node_probe = NodeProbe(self._num_nodes, self._cluster_cfg)
        self._nodes = self._node_probe.nodes
        self._node_groups = self._node_probe.node_groups

        self._logger.info(
            f"{Cluster.SYS_NAME} is running on a cluster with {len(self._nodes)} node{'s' if len(self._nodes) > 1 else ''} and {self.num_accelerators} accelerator{'s' if self.num_accelerators > 1 else ''}. The nodes' details are: "
            + "\n"
            + "\n".join(str(node) for node in self._nodes)
            + "\n"
            + "Node groups' details are: \n"
            + "\n".join(str(group) for group in self._node_groups)
        )

        # Set environment variables
        self._set_scheduler_env_vars()

        # Launch managers
        from ..manager import (
            CollectiveManager,
            DeviceLockManager,
            NodeManager,
            WorkerManager,
        )

        try:
            self._worker_manager = (
                ray.remote(WorkerManager)
                .options(name=WorkerManager.MANAGER_NAME)
                .remote()
            )
            self._coll_manager = (
                ray.remote(CollectiveManager)
                .options(name=CollectiveManager.MANAGER_NAME)
                .remote()
            )
            self._node_manager = (
                ray.remote(NodeManager)
                .options(name=NodeManager.MANAGER_NAME)
                .remote(self._nodes, self._node_groups, self._cluster_cfg)
            )
            self._lock_manager = (
                ray.remote(DeviceLockManager)
                .options(name=DeviceLockManager.MANAGER_NAME)
                .remote()
            )
        except ValueError:
            # If the WorkerManager is already running, we need to switch the namespace
            self._ray_instance_count += 1
            Cluster.NAMESPACE = f"RLinf_{self._ray_instance_count}"
            return self._init_and_launch_managers(num_nodes)

        def signal_handler(sig, frame):
            # Exit the main process if SIGUSR1 is received, which is sent by the worker group when an exception occurs.
            sys.stdout.flush()
            sys.stderr.flush()

            alive_actors = list_actors(
                filters=[
                    ("STATE", "=", "ALIVE"),
                    ("RAY_NAMESPACE", "=", Cluster.NAMESPACE),
                ]
            )
            for actor_state in alive_actors:
                actor = ray.get_actor(actor_state.name)
                ray.kill(actor, no_restart=True)

            if ray.is_initialized():
                # Mimic ray's sleep before shutdown to ensure log messages are flushed
                time.sleep(0.5)
                ray.shutdown(_exiting_interpreter=True)
            print("Exiting main process due to a failure upon worker execution.")
            exit(-1)

        signal.signal(signal.SIGUSR1, signal_handler)

    def _init_from_existing_managers(self):
        if not ray.is_initialized():
            ray.init(
                address="auto",
                namespace=Cluster.NAMESPACE,
                logging_level=Cluster.LOGGING_LEVEL,
            )

        from ..manager.node_manager import NodeManager

        self._node_manager = NodeManager.get_proxy()
        self._nodes, self._node_groups, self._cluster_cfg = (
            self._node_manager.get_nodes()
        )
        self._num_nodes = len(self._nodes)

    @staticmethod
    def get_full_env_var_name(var: ClusterEnvVar) -> str:
        """Get the full environment variable name with system prefix."""
        return f"{Cluster.SYS_NAME.upper()}_{var.value}"

    def _set_scheduler_env_vars(self):
        """Set default environment variables for the system."""
        env_var_list = list(ClusterEnvVar._value2member_map_.values())
        for node in self._nodes:
            for env_var in env_var_list:
                env_var_name = Cluster.get_full_env_var_name(env_var)
                if env_var_name in os.environ:
                    node.env_vars[env_var_name] = os.environ[env_var_name]
                elif (
                    default_value := Cluster.DEFAULT_SYS_ENV_VAR[env_var]
                ) is not None and env_var_name not in node.env_vars:
                    node.env_vars[env_var_name] = default_value

    @staticmethod
    def get_sys_env_var(
        env_var: ClusterEnvVar, default: Optional[str] = None
    ) -> Optional[str]:
        """Get the system environment variable for the cluster."""
        return os.environ.get(Cluster.get_full_env_var_name(env_var), default)

    @property
    def num_nodes(self):
        """Get the number of nodes in the cluster."""
        return self._num_nodes

    @property
    def num_accelerators(self):
        """Get the number of accelerators in the cluster."""
        return sum(node.num_accelerators for node in self._nodes)

    @property
    def accelerator_ranks(self) -> list[list[int]]:
        """Get the global accelerator ranks for each node in the cluster."""
        node_start_accel_rank = 0
        node_accel_ranks = []
        for node in self._nodes:
            node_accel_ranks.append(
                list(
                    range(
                        node_start_accel_rank,
                        node_start_accel_rank + node.num_accelerators,
                    )
                )
            )
            node_start_accel_rank += node.num_accelerators
        return node_accel_ranks

    @staticmethod
    def get_alive_nodes():
        """Get the list of alive nodes in the Ray cluster."""
        return [node for node in ray.nodes() if node["Alive"]]

    def get_node_group(
        self, label: Optional[str] = NodeGroupInfo.DEFAULT_GROUP_LABEL
    ) -> Optional[NodeGroupInfo]:
        """Get the node group information by label.

        Args:
            label (Optional[str]): The label of the node group.

        Returns:
            Optional[NodeGroupInfo]: The node group information.
        """
        if label is None:
            label = NodeGroupInfo.DEFAULT_GROUP_LABEL
        label = str(label)
        return next((ng for ng in self._node_groups if ng.label == label), None)

    def get_node_info(self, node_rank: int):
        """Get the NodeInfo of a specific node rank."""
        if node_rank < 0 or node_rank >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_rank}. Must be between 0 and {self._num_nodes - 1}."
            )
        assert self._nodes[node_rank].node_rank == node_rank, (
            f"Nodes are not correctly sorted in the cluster. The {node_rank}-th node's node_rank is {self._nodes[node_rank].node_rank}."
        )
        return self._nodes[node_rank]

    def get_node_ip(self, node_rank: int) -> str:
        """Get the IP address of a specific node by its rank."""
        return self._nodes[node_rank].node_ip

    def allocate(
        self,
        cls: type["Worker"],
        worker_name: str,
        node_rank: int,
        max_concurrency: int,
        env_vars: dict,
        node_group_label: str,
        cls_args: tuple,
        cls_kwargs: dict,
    ) -> ActorHandle:
        """Allocate a ray remote class instance on a specific node and local rank.

        Args:
            cls (Type[Worker]): The class to allocate.
            worker_name (str): The name of the worker.
            node_rank (int): The rank of the node to allocate on.
            max_concurrency (Optional[int]): The maximum concurrency for the worker's underlying ray actor.
            env_vars (dict): Environment variables to set for the worker.
            node_group_label (str): The label of the node group to allocate on.
            cls_args (tuple): Positional arguments to pass to the class constructor.
            cls_kwargs (dict): Keyword arguments to pass to the class constructor.

        Returns:
            ray.ObjectRef: A reference to the allocated remote class instance.

        """
        if node_rank < 0 or node_rank >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_rank}. Must be between 0 and {self._num_nodes - 1}."
            )

        node = self._nodes[node_rank]
        node_group = self.get_node_group(node_group_label)
        remote_cls = ray.remote(cls)

        merged_env_vars = node.env_vars.copy()
        # Update with user-specified env vars in node group configs
        cfg_node_env_vars = node_group.get_node_env_vars(node_rank)
        merged_env_vars.update(cfg_node_env_vars)
        # Finally, update with worker-specified env vars
        merged_env_vars.update(env_vars)

        # Update Python interpreter path
        python_interpreter_path = node.python_interpreter_path
        cfg_python_path = node_group.get_node_python_interpreter_path(node_rank)
        if cfg_python_path is not None:
            python_interpreter_path = cfg_python_path

        options = {
            "runtime_env": {
                "py_executable": python_interpreter_path,
                "env_vars": merged_env_vars,
            },
            "name": worker_name,
            "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node.ray_id,
                soft=False,
            ),
        }
        if max_concurrency is not None:
            assert 1 <= max_concurrency <= 2**31 - 1, (
                f"Invalid max_concurrency: {max_concurrency}. Must be between 1 and {2**31 - 1} (max int32) due to Ray's native layer limitation."
            )
            options["max_concurrency"] = max_concurrency

        return remote_cls.options(**options).remote(*cls_args, **cls_kwargs)
