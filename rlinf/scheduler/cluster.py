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
import signal
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, List, Optional, Type

import ray
import ray.util.scheduling_strategies
from packaging import version as vs
from ray._private import ray_logging
from ray.actor import ActorHandle
from ray.util.state import list_actors

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

    num_gpus: int
    """Number of GPUs available on the node."""

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

    def __init__(self, num_nodes: Optional[int] = None, num_gpus_per_node: int = 8):
        """Initialize the cluster.

        Args:
            num_nodes (int): The number of nodes in the cluster. When you wish to acquire the cluster instance in a processes other than the main driver process, do not pass this argument. Instead, use the `Cluster()` constructor without arguments.
            num_gpus_per_node (int): The number of GPUs available per node. Default is 8.
        """
        if self._has_initialized:
            return
        if num_nodes is not None:
            self._ray_instance_count = 0
            self._init_and_launch_managers(num_nodes, num_gpus_per_node)
        else:
            self._init_from_existing_managers()
        self._has_initialized = True

    def _init_and_launch_managers(self, num_nodes: int, num_gpus_per_node: int):
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
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
                # TODO : zhihao : local for debug
                # address="auto",
                address="local",
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
            print(
                f"Waiting for {self._num_nodes} nodes to be ready, currently {len(ray.nodes())} nodes available.",
                flush=True,
            )
            time.sleep(1)

        self._nodes: List[NodeInfo] = []
        for node in ray.nodes():
            self._nodes.append(
                NodeInfo(
                    node_rank=0,
                    ray_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    num_gpus=int(node["Resources"].get("GPU", 0)),
                    num_cpus=int(node["Resources"].get("CPU", 0)),
                )
            )

        self._master_ip = ray.util.get_node_ip_address()
        self._master_port = self.find_free_port()

        # Ensure master node is the node that launches the cluster
        self._master_node: NodeInfo = None
        other_nodes: List[NodeInfo] = []
        for node in self._nodes:
            if node.node_ip == self._master_ip:
                self._master_node = node
            else:
                other_nodes.append(node)
        assert self._master_node is not None, (
            f"Master node with IP {self._master_ip} not found in the cluster."
        )
        other_nodes = sorted(other_nodes, key=lambda x: x.node_ip)
        self._nodes = [self._master_node] + other_nodes

        # Launch managers
        from .manager import (
            CollectiveManager,
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
                .remote(
                    self._nodes,
                    self._num_gpus_per_node,
                    self._master_ip,
                    self._master_port,
                )
            )
        except ValueError:
            # If the WorkerManager is already running, we need to switch the namespace
            self._ray_instance_count += 1
            Cluster.NAMESPACE = f"RLinf_{self._ray_instance_count}"
            return self._init_and_launch_managers(num_nodes, num_gpus_per_node)

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
        self._master_ip = self._node_manager.get_master_ip()
        self._master_port = self._node_manager.get_master_port()
        self._num_gpus_per_node = self._node_manager.get_num_gpus_per_node()

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
    def master_addr(self):
        """Get the master address of the cluster."""
        return self._master_ip

    @property
    def master_port(self):
        """Get the master port of the cluster."""
        return self._master_port

    @property
    def num_gpus_per_node(self):
        """Get the number of GPUs per node."""
        return self._num_gpus_per_node

    @property
    def num_nodes(self):
        """Get the number of nodes in the cluster."""
        return self._num_nodes

    def get_node_ip(self, node_id: int) -> str:
        """Get the IP address of a specific node by its ID. Note that this is not the ray NodeID but the index of node in the cluster."""
        return self._nodes[node_id].node_ip

    def allocate(
        self,
        cls: Type["Worker"],
        worker_name: str,
        node_id: int,
        gpu_id: int,
        env_vars: dict,
        cls_args: List = [],
        cls_kwargs: dict = {},
    ) -> ActorHandle:
        """Allocate a ray remote class instance on a specific node and local rank.

        Args:
            cls (Type[Worker]): The class to allocate.
            worker_name (str): The name of the worker.
            node_id (int): The ID of the node to allocate on.
            gpu_id (int): The ID of the GPU to allocate.
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
        if gpu_id < 0 or gpu_id >= self._num_gpus_per_node:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. Must be between 0 and {self._num_gpus_per_node - 1}."
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

        return remote_cls.options(**options).remote(*cls_args, **cls_kwargs)
