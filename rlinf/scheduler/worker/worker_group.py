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

import asyncio
import os
import signal
import sys
import threading
import warnings
from dataclasses import dataclass
from typing import Generic, List, Optional, Type

import numpy as np
import ray
import ray.remote_function

from ..cluster import Cluster
from ..manager import WorkerInfo
from ..placement import PackedPlacementStrategy, PlacementStrategy
from .worker import Worker, WorkerAddress, WorkerClsType


class WorkerGroup(Generic[WorkerClsType]):
    """The class that enables a worker to become a group of workers that can be executed collectively."""

    def __init__(self, worker_cls: Type[Worker], args, kwargs):
        """Initialize the WorkerGroup with a worker class. Used as a decorator to create a worker group.

        Args:
            worker_cls (Type[Worker]): The worker class to be used in the group.
            args: The positional arguments of the class.
            kwargs: The keyword arguments of the class.

        """
        self._worker_cls = worker_cls
        self._worker_cls_args = args
        self._worker_cls_kwargs = kwargs
        self._worker_group_name = f"worker_group_{worker_cls.__name__}"

        self._workers = []
        self._cluster = None

        self._group_size = None
        self._master_addr = None
        self._master_port = None
        self._placement_strategy = (
            None  # The strategy to place workers on different GPUs
        )

        # Ranks to execute functions on in the worker group. If None, all workers will be executed.
        self._execution_ranks = None

        self._data_io_ranks = None

    @property
    def worker_cls_name(self) -> str:
        """Get the name of the worker class."""
        return self._worker_cls.__name__

    @property
    def worker_group_name(self) -> str:
        """Get the name of the worker group."""
        return self._worker_group_name

    @property
    def worker_info_list(self) -> List[WorkerInfo]:
        """Get the list of workers in the group."""
        return self._workers

    def launch(
        self: "WorkerGroup[WorkerClsType]",
        cluster: Cluster,
        placement_strategy: Optional[PlacementStrategy] = None,
        name: Optional[str] = None,
        isolate_gpu: bool = True,
        catch_system_failure: Optional[bool] = None,
    ) -> "Type[WorkerGroup | WorkerClsType]":
        """Create a worker group with the specified cluster and options.

        Args:
            cluster (ClusterResource): The cluster resource to use for worker placement.
            num_nodes (int): The number of nodes to create workers on.
            placement_strategy (Optional[PlacementStrategy]): The strategy to use for placing workers on nodes.
            name (str, optional): The name of the worker group.
            isolate_gpu (bool): Whether a worker should only see the GPUs that it's assigned via controlling CUDA_VISIBLE_DEVICES. Defaults to True.
            catch_system_failure (Optional[bool]): Whether to catch system exit and signals in the worker process. If None, the environment variable RLINF_CATCH_FAILURE will take effect, whose default value is True. If set, then it will override the environment variable.

        Returns:
            WorkerGroup: An instance of WorkerGroup with the specified configuration.

        """
        self._cluster = cluster

        self._master_addr = self._cluster.master_addr
        self._master_port = self._cluster.master_port
        self._placement_strategy = placement_strategy
        self._isolate_gpu = isolate_gpu
        self._catch_system_failure = catch_system_failure
        if self._catch_system_failure is None:
            self._catch_system_failure = Cluster.get_sys_env_var("CATCH_FAILURE") == "1"

        if self._placement_strategy is None:
            # Use all resources by default
            self._placement_strategy = PackedPlacementStrategy(
                0, cluster.num_nodes * cluster.num_gpus_per_node - 1
            )
        if name is not None:
            self._worker_group_name = name

        self._create_workers()
        self._attach_cls_func()
        self._is_ready()

        return self

    def execute_on(self, *ranks: int):
        """Set the ranks to execute functions on in the worker group. This function only affects the immediately subsequent call of any remote function of the WorkerGroup. After one call, the execute_on state is reset to execute on all ranks.

        Args:
            ranks (int): ranks to execute functions on. If None, all workers will be executed.

        """
        self._execution_ranks = list(ranks)
        return self

    def _nccl_env_vars(self):
        """Set the environment variables for NCCL. This is intended to align NCCL configurations of various workers."""
        env_vars = {
            "NCCL_CUMEM_ENABLE": "0",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        }
        if os.environ.get("NCCL_CUMEM_ENABLE", "0") != "0":
            warnings.warn(
                f"NCCL_CUMEM_ENABLE is set to {os.environ['NCCL_CUMEM_ENABLE']}. However, "
                "This may increase memory overhead with cudagraph+allreduce: "
                "https://github.com/NVIDIA/nccl/issues/1234, and thus set to 0 by both vLLM and SGLang, see https://github.com/vllm-project/vllm/pull/24141.",
            )
            env_vars["NCCL_CUMEM_ENABLE"] = os.environ["NCCL_CUMEM_ENABLE"]
        return env_vars

    def _close(self):
        """Close the worker group and release resources. This method is called when the worker group is no longer needed."""
        for worker_info in self._workers:
            # Call cleanup methods if they exist
            if hasattr(worker_info.worker, "_close"):
                ray.get(worker_info.worker._close.remote())
            ray.kill(worker_info.worker)
        self._workers.clear()
        self._cluster = None
        self._placement_strategy = None
        self._execution_ranks = None

    def _create_workers(self):
        """Create workers in the group, each worker is placed on a different GPU."""
        placements = self._placement_strategy.get_placement(
            self._cluster.num_nodes, self._cluster._num_gpus_per_node, self._isolate_gpu
        )
        self._world_size = len(placements)
        for placement in placements:
            worker_name = WorkerAddress.from_parent_name_rank(
                self._worker_group_name, placement.rank
            ).get_name()
            env_vars = {
                "GROUP_NAME": self._worker_group_name,
                "WORKER_NAME": worker_name,
                "WORLD_SIZE": str(self._world_size),
                "RANK": str(placement.rank),
                "NODE_RANK": str(placement.node_rank),
                "NODE_ID": str(placement.node_id),
                "GPU_ID": str(placement.local_gpu_id),
                "NODE_LOCAL_RANK": str(placement.local_rank),
                "NODE_LOCAL_WORLD_SIZE": str(placement.local_world_size),
                "RAY_ACTOR": str(1),
                "CUDA_VISIBLE_DEVICES": ",".join(placement.cuda_visible_devices),
                "CLUSTER_NAMESPACE": Cluster.NAMESPACE,
                "CATCH_SYSTEM_FAILURE": "1"
                if self._catch_system_failure
                else "0",  # Inform the Worker process to catch signals
                "ISOLATE_GPU": "1" if placement.isolate_gpu else "0",
                # Override Ray's control over GPU assignment
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
            }
            env_vars.update(self._nccl_env_vars())

            worker = self._cluster.allocate(
                cls=self._worker_cls,
                worker_name=worker_name,
                node_id=placement.node_id,
                gpu_id=placement.local_gpu_id,
                env_vars=env_vars,
                cls_args=self._worker_cls_args,
                cls_kwargs=self._worker_cls_kwargs,
            )

            @dataclass
            class WorkerRank:
                worker: ray.ObjectRef
                rank: int

            self._workers.append(WorkerRank(rank=placement.rank, worker=worker))

    def _attach_cls_func(self):
        """Attach the class function to the worker group so they can be called directly via the worker group instance.

        This allows execution of the original class methods collectively across multiple workers in the group.
        """
        # Get all callable methods of the WorkerGroup class and the Worker class
        worker_group_cls_func_list = [
            func
            for func in dir(WorkerGroup)
            if callable(getattr(WorkerGroup, func)) and not func.startswith("_")
        ]
        hidden_worker_group_cls_func_list = [
            func
            for func in dir(WorkerGroup)
            if callable(getattr(WorkerGroup, func))
            and func.startswith("_")
            and not func.startswith("__")
        ]
        func_list = [
            func
            for func in dir(self._worker_cls)
            if callable(getattr(self._worker_cls, func)) and not func.startswith("_")
        ]
        hidden_func_list = [
            func
            for func in dir(self._worker_cls)
            if callable(getattr(self._worker_cls, func))
            and func.startswith("_")
            and not func.startswith("__")
        ]
        for func_name in func_list:
            if func_name in worker_group_cls_func_list:
                raise ValueError(
                    f"Function {func_name} already exists in the {WorkerGroup.__name__} class, please rename it in the {self._worker_cls} class."
                )
            else:
                setattr(self, func_name, WorkerGroupFunc(self, func_name))

        for func_name in hidden_func_list:
            if func_name in hidden_worker_group_cls_func_list:
                raise ValueError(
                    f"Function {func_name} already exists in the {WorkerGroup.__name__} class, please rename it in the {self._worker_cls} class."
                )
            else:
                # If the function starts with an underscore, create a HiddenWorkerGroupFunc
                setattr(
                    self,
                    func_name,
                    HiddenWorkerGroupFunc(func_name, self._worker_cls.__name__),
                )

        # Attach ready checking func
        setattr(self, "_is_ready", WorkerGroupFunc(self, "__ray_ready__"))


class HiddenWorkerGroupFunc:
    """Hidden functions that start with an underscore are not exposed to the user."""

    def __init__(self, func_name: str, cls_name):
        """Initialize the hidden function (function that starts with an underscore) to attach to WorkerGroup.

        Args:
            func_name (str): The name of the function.
            cls_name (str): The name of the class that the function belongs to.

        """
        self._func_name = func_name
        self._cls_name = cls_name

    def __call__(self, *args, **kwargs):
        """Raise an error if the user tries to call a hidden function directly.

        This is to prevent users from calling functions that are not meant to be called publicly.

        Raises:
            ValueError: If the user tries to call a hidden function directly.

        """
        raise ValueError(
            f"Function {self._func_name} of class {self._cls_name} is hidden (starts with an '_') and cannot be called directly via {WorkerGroup.__name__}. You can either remove the '_' or implement its logic somewhere else."
        )


class WorkerGroupFunc:
    """Public functions of the WorkerGroup that can be called directly."""

    def __init__(self, worker_group: WorkerGroup, func_name: str):
        """Initialize the WorkerGroupFunc to attach to WorkerGroup.

        Args:
            worker_group (WorkerGroup): The worker group to attach the function to.
            func_name (str): The name of the function.
            func (Callable): The function to attach to the worker group.

        """
        self._worker_group = worker_group
        self._func_name = func_name

    @property
    def worker_group(self) -> WorkerGroup:
        """Get the worker group this function belongs to."""
        return self._worker_group

    @property
    def func_name(self) -> str:
        """Get the name of the function."""
        return self._func_name

    def __call__(self, *args, **kwargs):
        """Execute the function on the specified ranks in the worker group.

        This method collects the results from all workers in the group and returns a WorkerGroupFuncResult.
        """
        results = []

        if self._worker_group._execution_ranks is None:
            # If no specific ranks are set, execute on all workers in the group
            self._worker_group._execution_ranks = list(
                range(len(self._worker_group._workers))
            )
            assert (
                len(self._worker_group._execution_ranks)
                == self._worker_group._world_size
            ), (
                f"Execution ranks {self._worker_group._execution_ranks} do not match the world size {self._worker_group._world_size}."
            )
        assert not any(
            rank < 0 or rank >= len(self._worker_group._workers)
            for rank in self._worker_group._execution_ranks
        ), "Invalid rank(s) specified."

        # Execute the function on the specified ranks
        for rank in self._worker_group._execution_ranks:
            worker_p = self._worker_group._workers[rank]
            assert worker_p.rank == rank, (
                f"Worker rank mismatch: expected {rank}, got {worker_p.rank}."
            )
            results.append(
                getattr(worker_p.worker, self._func_name).remote(*args, **kwargs)
            )

        result = WorkerGroupFuncResult(
            self._worker_group,
            results,
            self._func_name,
            self._worker_group._worker_cls.__name__,
        )

        # Reset execution ranks after execution
        self._worker_group._execution_ranks = None

        return result


class WorkerGroupFuncResult:
    """A result object that contains the results of a function executed on a WorkerGroup."""

    def __init__(
        self,
        worker_group: WorkerGroup,
        results: List[ray.remote_function.RemoteFunction],
        func_name: str,
        cls_name: str,
    ):
        """Initialize the WorkerGroupFuncResult with the results of the Ray function execution.

        Upon creation, it starts a thread to wait for the results to complete.

        Args:
            worker_group (WorkerGroup): The worker group that the function was executed on.
            results (List[ray.remote_function.RemoteFunction]): The results of the Ray function execution.
            func_name (str): The name of the function that was executed.
            cls_name (str): The name of the class that the function belongs to.

        """
        self._worker_group: Worker = worker_group
        self._remote_results = results
        self._local_results = None
        self._func_name = func_name
        self._pid = os.getpid()
        self._cls_name = cls_name
        self._wait_done = False

        # Every definition should be put before the thread starts
        self._wait_thread = threading.Thread(target=self._wait_for_results, daemon=True)
        self._wait_thread.start()

    def _wait_for_results(self):
        """Wait for all remote results to complete. This is run in a separate thread to avoid blocking the main thread."""
        try:
            self._local_results = ray.get(self._remote_results)
        except Exception as e:
            print(
                f"Exception occurred while running {self._cls_name}'s function {self._func_name}: exception is {e}"
            )
            sys.stdout.flush()
            sys.stderr.flush()
            # Send suicide signal if one thread failed, the handler is registered in cluster
            os.kill(self._pid, signal.SIGUSR1)
            exit(-1)
        self._wait_done = True

    def consume_duration(self, reduction_type: str = "max"):
        """Get the max execution time of a function across different ranks of a group.

        This implicitly waits for the function to finish.

        Args:
            reduction_type (str): The type of reduction to apply. Can be "max", "min", or "mean".
        """
        self.wait()
        execution_times = self._worker_group.pop_execution_time(self._func_name).wait()
        reduction_func = getattr(np, reduction_type)
        return reduction_func(execution_times)

    def wait(self):
        """Wait for all remote results to complete and return the results."""
        if not self._wait_done:
            self._wait_thread.join()
        return self._local_results

    async def async_wait(self):
        """Asynchronously wait for all remote results to complete and return the results."""
        while not self._wait_done:
            await asyncio.sleep(0.1)
        return self._local_results
