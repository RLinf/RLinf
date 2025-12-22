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
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Optional

import ray
import ray.util.scheduling_strategies
from packaging import version as vs
from ray._private import ray_logging
from ray.actor import ActorHandle
from ray.util.state import list_actors

from .accelerator import Accelerator, AcceleratorType

ray_version = version("ray")
assert vs.parse(ray_version) >= vs.parse("2.47.0"), (
    "Ray version 2.47.0 or higher is required. Run pip install ray[default]==2.47.0"
)

if TYPE_CHECKING:
    from .worker import Worker


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_rank: str
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    accelerator_type: AcceleratorType
    """Type of accelerator available on the node."""

    num_accelerators: int
    """Number of accelerators available on the node."""

    num_cpus: int
    """Number of CPUs available on the node."""


class Cluster:
    """A singleton class that manages the cluster resources for Ray workers."""

    SYS_NAME = "RLinf"
    NAMESPACE = SYS_NAME
    LOGGING_LEVEL = "INFO"
    TIMEOUT_WARN_TIME = 60000

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

    def __init__(self, num_nodes: Optional[int] = None):
        """Initialize the cluster.

        Args:
            num_nodes (int): The number of nodes in the cluster. When you wish to acquire the cluster instance in a processes other than the main driver process, do not pass this argument. Instead, use the `Cluster()` constructor without arguments.
        """
        if self._has_initialized:
            return
        if num_nodes is not None:
            self._ray_instance_count = 0
            self._init_and_launch_managers(num_nodes)
        else:
            self._init_from_existing_managers()
        self._has_initialized = True

    def _init_and_launch_managers(self, num_nodes: int):
        assert num_nodes > 0, "num_nodes must be greater than 0."

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

        self._num_nodes = num_nodes
        self._set_default_env_vars()

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
        try:
            # First try to connect to an existing Ray cluster
            ray.init(
                address="auto",
                logging_level=Cluster.LOGGING_LEVEL,
                namespace=Cluster.NAMESPACE,
                runtime_env={"env_vars": dict(os.environ)},
            )
        except ConnectionError:
            ray.init(
                logging_level=Cluster.LOGGING_LEVEL,
                namespace=Cluster.NAMESPACE,
                runtime_env={"env_vars": dict(os.environ)},
            )

        # Wait for the cluster to be ready
        while len(ray.nodes()) < self._num_nodes:
            self._logger.warning(
                f"Waiting for {self._num_nodes} nodes to be ready, currently {len(ray.nodes())} nodes available."
            )
            time.sleep(1)

        self._nodes: list[NodeInfo] = []
        for node in ray.nodes():
            accelerator_type, num_accelerators = (
                Accelerator.get_node_accelerator_type_and_num(node)
            )
            self._nodes.append(
                NodeInfo(
                    node_rank=0,
                    ray_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    accelerator_type=accelerator_type,
                    num_accelerators=num_accelerators,
                    num_cpus=int(node["Resources"].get("CPU", 0)),
                )
            )

        # Sort nodes first by accelerator type, then by IP
        nodes_group_by_accel_type: dict[AcceleratorType, list[NodeInfo]] = {
            accel_type: [] for accel_type in AcceleratorType
        }
        for node in self._nodes:
            nodes_group_by_accel_type[node.accelerator_type].append(node)
        for accel_type in nodes_group_by_accel_type.keys():
            nodes_group_by_accel_type[accel_type].sort(key=lambda x: x.node_ip)
        self._nodes = [
            node for nodes in nodes_group_by_accel_type.values() for node in nodes
        ]

        # Handle num_nodes configuration mismatch with actual node number
        if len(self._nodes) > self._num_nodes:
            warnings.warn(
                f"The cluster is initialized with {self._num_nodes} nodes, but detected {len(self._nodes)} nodes have joined the ray cluster. So only the first {self._num_nodes} nodes are used."
            )
            self._nodes = self._nodes[: self._num_nodes]

        self._logger.info(
            f"{Cluster.SYS_NAME} is running on a cluster with {len(self._nodes)} node{'s' if len(self._nodes) > 1 else ''} and {self.num_accelerators_in_cluster} accelerator{'s' if self.num_accelerators_in_cluster > 1 else ''}. The nodes' details are: {self._nodes}"
        )

        # Launch managers
        from .manager import (
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
                .remote(self._nodes)
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

        from .manager.node_manager import NodeManager

        self._node_manager = NodeManager.get_proxy()
        self._nodes = self._node_manager.get_nodes()
        self._num_nodes = len(self._nodes)

    def _set_default_env_vars(self):
        """Set default environment variables for the system."""
        env_var_list = ["CATCH_FAILURE", "LOG_LEVEL", "TIMEOUT"]
        system_name = Cluster.SYS_NAME.upper()
        for env_var in env_var_list:
            env_var = f"{system_name}_{env_var}"
            if env_var not in os.environ:
                if env_var == f"{system_name}_CATCH_FAILURE":
                    os.environ[env_var] = "0"
                elif env_var == f"{system_name}_LOG_LEVEL":
                    os.environ[env_var] = "INFO"
                elif env_var == f"{system_name}_TIMEOUT":
                    os.environ[env_var] = "180"

    @staticmethod
    def get_sys_env_var(env_var: str, default: Optional[str] = None) -> Optional[str]:
        """Get the system environment variable for the cluster."""
        system_name = Cluster.SYS_NAME.upper()
        env_var = f"{system_name}_{env_var}"
        return os.environ.get(env_var, default)

    @property
    def num_nodes(self):
        """Get the number of nodes in the cluster."""
        return self._num_nodes

    @property
    def num_accelerators_in_cluster(self):
        """Get the number of accelerators in the cluster."""
        return sum(node.num_accelerators for node in self._nodes)

    @property
    def node_accelerator_ids(self) -> list[list[int]]:
        """Get the global accelerator IDs for each node in the cluster."""
        node_start_accel_id = 0
        node_accel_ids = []
        for node in self._nodes:
            node_accel_ids.append(
                list(
                    range(
                        node_start_accel_id, node_start_accel_id + node.num_accelerators
                    )
                )
            )
            node_start_accel_id += node.num_accelerators
        return node_accel_ids

    def get_node_id_from_accel_id(self, accel_id: int) -> int:
        """Get the node ID from the global accelerator ID.

        Args:
            accel_id (int): The global accelerator ID.

        Returns:
            int: The node ID.
        """
        for i, ids in enumerate(self.node_accelerator_ids):
            if accel_id in ids:
                return i
        raise ValueError(f"Accelerator ID {accel_id} not found in any node.")

    def get_node_num_accelerators(self, node_id: int) -> int:
        """Get the number of accelerators in a specific node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            int: The number of accelerators in the node.
        """
        if node_id < 0 or node_id >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_id}. Must be between 0 and {self._num_nodes - 1}."
            )
        return self._nodes[node_id].num_accelerators

    def global_accel_id_to_local_accel_id(self, accel_id: int):
        """Get the local accelerator ID from the global accelerator ID.

        Args:
            accel_id (int): The global accelerator ID.

        Returns:
            int: The local accelerator ID.
        """
        node_id = self.get_node_id_from_accel_id(accel_id)
        node_accel_ids = self.node_accelerator_ids[node_id]
        assert accel_id in node_accel_ids, (
            f"Accelerator ID {accel_id} not found in node {node_id}."
        )
        return node_accel_ids.index(accel_id)

    def get_node_info(self, node_id: int):
        """Get the NodeInfo of a specific node rank."""
        return self._nodes[node_id]

    def get_node_ip(self, node_id: int) -> str:
        """Get the IP address of a specific node by its ID. Note that this is not the ray NodeID but the index of node in the cluster."""
        return self._nodes[node_id].node_ip

    def allocate(
        self,
        cls: type["Worker"],
        worker_name: str,
        node_id: int,
        max_concurrency: int,
        env_vars: dict,
        cls_args: list = [],
        cls_kwargs: dict = {},
    ) -> ActorHandle:
        """Allocate a ray remote class instance on a specific node and local rank.

        Args:
            cls (Type[Worker]): The class to allocate.
            worker_name (str): The name of the worker.
            node_id (int): The ID of the node to allocate on.
            max_concurrency (Optional[int]): The maximum concurrency for the worker's underlying ray actor.
            env_vars (dict): Environment variables to set for the worker.
            cls_args (List): Positional arguments to pass to the class constructor.
            cls_kwargs (dict): Keyword arguments to pass to the class constructor.

        Returns:
            ray.ObjectRef: A reference to the allocated remote class instance.

        """
        if node_id < 0 or node_id >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_id}. Must be between 0 and {self._num_nodes - 1}."
            )

        node = self._nodes[node_id]
        remote_cls = ray.remote(cls)

        options = {
            "runtime_env": {"env_vars": env_vars},
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
