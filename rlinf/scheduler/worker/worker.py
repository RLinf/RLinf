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

import functools
import inspect
import logging
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeVar,
)

import ray
import torch
from omegaconf import OmegaConf

from ..cluster import Cluster, ClusterEnvVar
from ..hardware import AcceleratorType, AcceleratorUtil, HardwareInfo
from ..manager import WorkerAddress

if TYPE_CHECKING:
    from .worker_group import WorkerGroup

WorkerClsType = TypeVar("WorkerClsType")


class WorkerMeta(type):
    """Metaclass to capture failures in worker classes."""

    def __new__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]):
        """Wrap the function to catch SystemExit exceptions."""
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = cls._catch_failure_for_cls_func(
                    name, attr_name, attr_value
                )
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def _catch_failure_for_cls_func(cls, cls_name, func_name: str, func: Callable):
        """Wrap a try...except SystemExit block around the class function calls."""
        # Get all callable methods of the WorkerGroup class and the Worker class
        if func_name.startswith("_") and func_name != "__init__":
            return func

        def func_wrapper(func: Callable):
            @functools.wraps(func)
            def sync_func(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except SystemExit:
                    # Catch SystemExit and log the error
                    raise RuntimeError(
                        f"SystemExit caught in {cls_name}'s function {func.__name__}, traceback is below: {traceback.format_exc()}"
                    )

            @functools.wraps(func)
            async def async_func(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except SystemExit:
                    # Catch SystemExit and log the error
                    raise RuntimeError(
                        f"SystemExit caught in {cls_name}'s function {func.__name__}, traceback is below: {traceback.format_exc()}"
                    )

            if inspect.iscoroutinefunction(func):
                return async_func
            elif inspect.isasyncgenfunction(func):
                raise NotImplementedError(
                    f"Async generator function {func.__name__} is not supported when CATCH_FAILURE is enabled."
                )
            else:
                return sync_func

        return func_wrapper(func)


class Worker(metaclass=WorkerMeta):
    """Class representing a remote process or worker.

    Inheriting `Worker` will grant your worker or processor class the ability to run remotely and communicate with other workers in the cluster.
    Also, essential environment variables like MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE will be set automatically.
    This allows easy creation of torch process groups and distributed training.

    The following example shows how to use the Worker class to create a simple distributed worker that can run on multiple GPUs and nodes.

    Example::

        >>> import torch
        >>> from rlinf.scheduler import Cluster, Worker
        >>>
        >>> class MyWorker(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def initialize(self):
        ...         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        ...         if not torch.distributed.is_initialized():
        ...             torch.distributed.init_process_group(backend="nccl")
        ...
        ...         test_tensor = torch.ones(
        ...             size=(1, 1), dtype=torch.float32, device=torch.cuda.current_device()
        ...         )
        ...         torch.distributed.all_reduce(test_tensor)
        ...         return test_tensor
        ...
        ...     def hello(self):
        ...         return self._rank
        >>>
        >>> cluster = Cluster(num_nodes=1)
        >>> my_worker_group = MyWorker.create_group().launch(cluster=cluster, name="my_worker_group")
        >>> my_worker_group.initialize().wait()[0]
        tensor([[4.]], device='cuda:0')
        >>> # This will execute the hello method only on ranks 0 and 1.
        >>> my_worker_group.execute_on(0, 3).hello().wait()
        [0, 3]

    The following example shows the communication capabilities of the Worker class.

    Example::

        >>> import asyncio
        >>> import torch
        >>> from rlinf.scheduler import Cluster, Worker
        >>> SEND_GROUP_NAME = "send_worker_group"
        >>> RECV_GROUP_NAME = "recv_worker_group"
        >>>
        >>> class SendWorker(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def hello_recv(self):
        ...         # 1. Send a message (string or any serializable object) to the RecvWorker group with the same rank as this SendWorker worker.
        ...         msg = f"Hello from SendWorker Rank {self._rank}!"
        ...         self.send(msg, dst_group_name=RECV_GROUP_NAME, dst_rank=self._rank)
        ...
        ...         # 2. Receive a reply from the RecvWorker group with the same rank.
        ...         reply = self.recv(
        ...             src_group_name=RECV_GROUP_NAME, src_rank=self._rank
        ...         )
        ...
        ...         # 3. The send/recv APIs can also handle tensor, list of tensors and dict of tensors.
        ...         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        ...         dst_rank = (
        ...             self._rank + 1
        ...         ) % self._world_size  # Send to the next rank in the group
        ...         tensor = torch.ones(
        ...             size=(1, 1),
        ...             dtype=torch.float32,
        ...             device=torch.cuda.current_device(),
        ...         )
        ...         self.send(tensor, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank)
        ...
        ...         tensor_list = [
        ...             torch.tensor(
        ...                 1.0, dtype=torch.float32, device=torch.cuda.current_device()
        ...             )
        ...             for _ in range(4)
        ...         ]
        ...         self.send(
        ...             tensor_list, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank
        ...         )
        ...
        ...         tensor_dict = {
        ...             "tensor1": torch.tensor(
        ...                 2.0, dtype=torch.float32, device=torch.cuda.current_device()
        ...             ),
        ...             "tensor2": torch.tensor(
        ...                 3.0, dtype=torch.float32, device=torch.cuda.current_device()
        ...             ),
        ...         }
        ...         self.send(
        ...             tensor_dict, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank
        ...         )
        ...
        ...         # 4. Send tensor directly without metadata overhead if you already know the tensor shape and dtype at the recv side
        ...         tensor = torch.ones(
        ...             size=(2, 1),
        ...             dtype=torch.float32,
        ...             device=torch.cuda.current_device(),
        ...         )
        ...         self.send_tensor(
        ...             tensor, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank
        ...         )
        ...
        ...     def hello_recv_async(self):
        ...         # 1. Send a tensor asynchronously to the RecvWorker group with the next rank.
        ...         dst_rank = (self._rank + 1) % self._world_size
        ...         tensor = torch.ones(
        ...             size=(3, 1),
        ...             dtype=torch.float32,
        ...             device=torch.cuda.current_device(),
        ...         )
        ...         async_send_work = self.send(
        ...             tensor,
        ...             dst_group_name=RECV_GROUP_NAME,
        ...             dst_rank=dst_rank,
        ...             async_op=True,
        ...         )
        ...         async_send_work.wait()  # Wait for the async send to complete
        ...
        ...         # 2. Send a tensor asynchronously and use asyncio to wait for the operation to complete.
        ...         async def send_tensor_async():
        ...             dst_rank = (self._rank + 1) % self._world_size
        ...             tensor = torch.ones(
        ...                 size=(4, 1),
        ...                 dtype=torch.float32,
        ...                 device=torch.cuda.current_device(),
        ...             )
        ...             async_send_work = self.send(
        ...                 tensor,
        ...                 dst_group_name=RECV_GROUP_NAME,
        ...                 dst_rank=dst_rank,
        ...                 async_op=True,
        ...             )
        ...             await async_send_work.async_wait()
        ...
        ...         asyncio.run(send_tensor_async())
        >>>
        >>> class RecvWorker(Worker):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def hello_recv(self):
        ...         # 1. Receive a message from the SendWorker worker group with the same rank.
        ...         msg = self.recv(src_group_name=SEND_GROUP_NAME, src_rank=self._rank)
        ...
        ...         # 2. Send a reply back to the SendWorker worker group with the same rank.
        ...         reply = f"Hello from RecvWorker Rank {self._rank}!"
        ...         self.send(
        ...             reply, dst_group_name=SEND_GROUP_NAME, dst_rank=self._rank
        ...         )
        ...
        ...         # 3. Receive a tensor, tensor list and tensor dict from the SendWorker worker group with the same rank.
        ...         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        ...         src_rank = (
        ...             self._rank - 1
        ...         ) % self._world_size  # Receive from the previous rank in the group
        ...         tensor = self.recv(
        ...             src_group_name=SEND_GROUP_NAME, src_rank=src_rank
        ...         )
        ...         tensor_list = self.recv(
        ...             src_group_name=SEND_GROUP_NAME, src_rank=src_rank
        ...         )
        ...         tensor_dict = self.recv(
        ...             src_group_name=SEND_GROUP_NAME, src_rank=src_rank
        ...         )
        ...
        ...         # 4. In-place receive tensor directly without metadata overhead
        ...         tensor = torch.empty(
        ...             size=(2, 1),
        ...             dtype=torch.float32,
        ...             device=torch.cuda.current_device(),
        ...         )
        ...         self.recv_tensor(
        ...             tensor, src_group_name=SEND_GROUP_NAME, src_rank=src_rank
        ...         )
        ...
        ...     def hello_recv_async(self):
        ...         # 1. Receive a tensor asynchronously from the SendWorker group with the next rank.
        ...         src_rank = (self._rank - 1) % self._world_size
        ...         async_recv_work = self.recv(
        ...             src_group_name=SEND_GROUP_NAME, src_rank=src_rank, async_op=True
        ...         )
        ...         tensor = async_recv_work.wait()
        ...
        ...         # 2. Receive a tensor asynchronously and use asyncio to wait for the operation to complete.
        ...         async def recv_tensor_async():
        ...             src_rank = (self._rank - 1) % self._world_size
        ...             async_recv_work = self.recv(
        ...                 src_group_name=SEND_GROUP_NAME,
        ...                 src_rank=src_rank,
        ...                 async_op=True,
        ...             )
        ...             tensor = await async_recv_work.async_wait()
        ...
        ...         asyncio.run(recv_tensor_async())
        >>>
        >>> cluster = Cluster(num_nodes=1)
        >>> send_group = SendWorker.create_group().launch(cluster=cluster, name=SEND_GROUP_NAME)
        >>> recv_group = RecvWorker.create_group().launch(cluster=cluster, name=RECV_GROUP_NAME)
        >>> res = send_group.hello_recv()
        >>> res = recv_group.hello_recv().wait()
        >>> res = send_group.hello_recv_async()
        >>> res = recv_group.hello_recv_async().wait()

    """

    PID = None
    current_worker = None
    logging.basicConfig()
    logger = logging.getLogger(Cluster.SYS_NAME)
    logger.setLevel(Cluster.LOGGING_LEVEL)
    torch_platform = torch.cuda
    torch_device_type = "cuda"

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the Worker class."""
        instance = super().__new__(cls)

        cluster_node_rank = os.environ.get("CLUSTER_NODE_RANK", None)

        # ray.remote initializes the class with the ActorClass wrapper locally first (not in a remote process),
        # which doesn't have the environment variables set.
        if cluster_node_rank is not None and "ActorClass(" not in cls.__name__:
            instance._env_setup_before_init()
            # Handle OS signals for better debuggability
            # Ray new the class in main thread but call __init__ in worker thread if it's an Actor with async functions
            # Since signal handlers must be registered in main thread, we call the registration in __new__
            instance._register_signal_handlers()

        return instance

    def _env_setup_before_init(self):
        """Set up distributed Worker environments."""
        # These are required env_vars necessary for both Ray Worker and non-Ray Worker
        # For non-ray workers, these are reset in the __init__ method
        self._rank = int(os.environ.get("RANK", "-1"))
        self._worker_name = os.environ.get("WORKER_NAME", None)
        self._world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        if self._worker_name is not None:
            self._worker_address = WorkerAddress.from_name(self._worker_name)

        # These are not required env_vars, but are set by Ray Worker for convenience
        self._cluster_node_rank = int(os.environ.get("CLUSTER_NODE_RANK", -1))
        self._accelerator_type = AcceleratorType(
            os.environ.get("ACCELERATOR_TYPE", str(AcceleratorType.NO_ACCEL.value))
        )
        self._local_accelerator_rank = int(os.environ.get("LOCAL_ACCELERATOR_RANK", -1))
        self._node_local_rank = int(os.environ.get("NODE_LOCAL_RANK", -1))
        self._node_local_world_size = int(os.environ.get("NODE_LOCAL_WORLD_SIZE", -1))
        Worker.torch_device_type = AcceleratorUtil.get_device_type(
            self._accelerator_type
        )
        Worker.torch_platform = AcceleratorUtil.get_torch_platform(
            self._accelerator_type
        )
        self.torch_device_type = Worker.torch_device_type
        self.torch_platform = Worker.torch_platform

        self._actor = None
        self._has_initialized = False
        self._timer_metrics: dict[str, float] = {}
        self._set_new_omegaconf_resolvers()

    def __init__(
        self,
        parent_address: Optional[WorkerAddress] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """Initialize the Worker with the given parent address and world size.

        Only non-Ray workers should provide parent_address, world_size and rank. For example, when a Worker is created via multiprocessing by another Worker, the parent address, world size and rank should be provided.

        Args:
            parent_address (Optional[WorkerAddress]): The address of the parent worker. This is used to set up the WorkerAddress for this worker.
            world_size (Optional[int]): The total number of workers in the group. If not provided, it will be set to the environment variable WORLD_SIZE.
            rank (Optional[int]): The rank of this worker in the group. If not provided, it will be set to the environment variable RANK.

        """
        if rank is not None and parent_address is not None and world_size is not None:
            # The Worker is not a Ray actor
            self._rank = rank
            self._worker_address = parent_address.get_child_address(rank)
            self._world_size = world_size
            self._worker_name = self._worker_address.get_name()
            # Forked process might inherit the environment variable RAY_ACTOR, but it is not a Ray actor.
            self._is_ray_actor = False
        else:
            self._is_ray_actor = True

        if self._is_ray_actor and not hasattr(self, "_local_accelerator_rank"):
            raise RuntimeError(
                "You may have mistakenly initialized the Worker class directly without `create_group` and `launch`. Please ensure a worker class is not instantiated on the main process directly like `Worker()`, but `Worker.create_group().launch()`."
            )

        Worker.PID = os.getpid()
        self._thread = threading.current_thread()
        self._stacklevel = 4 if self._is_ray_actor else 3

        # Reset Cluster.NAMESPACE for this Worker process according to the environment variable
        namespace = os.environ.get("CLUSTER_NAMESPACE", None)
        assert namespace is not None, (
            "CLUSTER_NAMESPACE environment variable must be set before initializing Worker."
        )
        Cluster.NAMESPACE = namespace

        if self._is_ray_actor and parent_address is not None:
            # The Worker is a Ray actor launched inside a Worker
            self._worker_address = parent_address.get_child_address(self._rank)
            self._worker_name = self._worker_address.get_name()
            os.environ["WORKER_NAME"] = self._worker_name
        self._group_name = self._worker_address.get_parent_address().get_name()

        # Setup local rank and world size
        self._setup_local_rank_world_size()

        # Setup accelerator ID
        self._setup_accelerator_info()

        # Configure logging
        self._setup_logging()

        # Init ray and managers
        self._manager_proxy = None
        self._collective = None
        self._init_ray_and_managers()

        # Setup MASTER_ADDR and MASTER_PORT
        self._setup_master_address_and_port()

        # Setup node group and hardware ranks
        self._setup_hardware()

        # Setup communication envs
        self._setup_comm_envs()

        self._lock = threading.Lock()

        from .lock import DeviceLock

        self._device_lock = DeviceLock(self)

        Worker.current_worker = self
        self._has_initialized = True

    @property
    def has_accelerator(self) -> bool:
        """Whether the worker has been allocated with accelerators."""
        return self._accelerator_type != AcceleratorType.NO_ACCEL

    @property
    def worker_address(self) -> WorkerAddress:
        """Get the WorkerAddress of the worker.

        This is used to identify the worker in the WorkerGroup.
        """
        return self._worker_address

    @property
    def manager_proxy(self):
        """Get the SchedulerProxy instance for this worker.

        This is used to interact with the scheduler and register the worker.
        """
        return self._manager_proxy

    @property
    def device_lock(self):
        """Get the DeviceLock instance for this worker."""
        return self._device_lock

    @property
    def hardware_type(self) -> str:
        """Get the hardware type of the current worker.

        Returns:
            str: The hardware type of the current worker.
        """
        return self._node_group.hardware_type

    @property
    def hardware_infos(self) -> list[HardwareInfo]:
        """Get the hardware information of the current worker.

        Returns:
            list[HardwareInfo]: The list hardware information assigned to the current worker.
        """
        infos = []
        for hw_rank in self._hardware_ranks:
            infos.append(self._node_group.local_hardware_infos[hw_rank])
        return infos

    @classmethod
    def create_group(
        cls: type[WorkerClsType], *args, **kwargs
    ) -> "WorkerGroup[WorkerClsType]":
        """Create a worker group with the class arguments.

        Args:
            args: The positional arguments of the class.
            kwargs: The keyword arguments of the class.
        """
        from .worker_group import WorkerGroup

        return WorkerGroup(cls, args, kwargs)

    def send(
        self,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        dst_group_name: str,
        dst_rank: int | list[int],
        async_op: bool = False,
    ):
        """Send an object to a specific worker address in the collective group.

        The function is specially optimized for torch.Tensor, List of torch.Tensor, Dict of torch.Tensor, which go through NCCL when the contained tensors are on GPU. Otherwise, all communications go through GLOO.

        .. note::
            Do not mix send with recv_tensor

        .. note::
            We only use NCCL primitives when the list or dict values only contain GPU tensors. We also see complex dicts with deep hierarchy as common Python objects, which will be serialized into a CPU tensor and sent through GLOO.

        .. note::
            When transferring GPU objects, the first send needs to be paired with a recv at the other end. Calling async send or recv first at both ends will result in communication hang, because NCCL communicators are established in a lazy manner when the first pair of send/recv is called.

        .. note::
            Do not mix CPU and GPU tensors in a list or dict.

        .. note::
            This method is not thread safe.

        Args:
            object (torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor] | Any): The object to send.
            dst_group_name (str): The name of the destination worker group.
            dst_rank (int | List[int]): The rank or list of ranks in the destination worker group to send the object to. For SPMD-like workers, this should be a single rank. For SPSD-like workers forked by parent workers, this can be a list of ranks that forms a path from the root worker to the target worker.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: An AsyncWork object if async_op is True, otherwise None.

        """
        dst_addr = WorkerAddress(dst_group_name, ranks=dst_rank)
        group = self._get_p2p_collective_group(dst_addr)
        return group.send(object=object, async_op=async_op)

    def recv(
        self, src_group_name: str, src_rank: int | list[int], async_op: bool = False
    ):
        """Out-of-place receive of an object from a specific worker address in the collective group.

        .. note::
            Do not mix recv with send_tensor

        .. note::
            When transferring GPU objects, the first send needs to be paired with a recv at the other end. Calling async send or recv first at both ends will result in communication hang, because NCCL communicators are established in a lazy manner when the first pair of send/recv is called.

        .. note::
            This method is not thread safe.

        Args:
            async_op (bool): Whether to perform the operation asynchronously.
            src_group_name (str): The name of the source worker group.
            src_rank (int | List[int]): The rank or list of ranks in the source worker group to receive the object from. For SPMD-like workers, this should be a single rank. For SPSD-like workers forked by parent workers, this can be a list of ranks that forms a path from the root worker to the target worker.

        Returns:
            AsyncWork | torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor] | Any: An AsyncWork object if async_op is True, otherwise the received object.

        """
        src_addr = WorkerAddress(src_group_name, ranks=src_rank)
        group = self._get_p2p_collective_group(src_addr)
        return group.recv(async_op=async_op)

    def send_tensor(
        self,
        tensor: torch.Tensor,
        dst_group_name: str,
        dst_rank: int | list[int],
        async_op: bool = False,
    ):
        """Send a tensor to a specific worker address in the collective group. This function is optimized for sending a single tensor and does not introduce metadata communication overhead like send. But it needs to be paired with the in-place recv_tensor function which requires apriori knowledge of the tensor shape and dtype.

        .. note::
            Do not mix send_tensor with recv

        .. note::
            When transferring GPU objects, the first send_tensor needs to be paired with a recv_tensor at the other end. Calling async send_tensor or recv_tensor first at both ends will result in communication hang, because NCCL communicators are established in a lazy manner when the first pair of send/recv is called.

        .. note::
            This method is not thread safe.

        Args:
            tensor (torch.Tensor): The tensor to send.
            dst_group_name (str): The name of the destination worker group.
            dst_rank (int | List[int]): The rank or list of ranks in the destination worker group to send the tensor to. For SPMD-like workers, this should be a single rank. For SPSD-like workers forked by parent workers, this can be a list of ranks that forms a path from the root worker to the target worker.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: An AsyncWork object if async_op is True, otherwise None.

        """
        dst_addr = WorkerAddress(dst_group_name, ranks=dst_rank)
        group = self._get_p2p_collective_group(dst_addr)
        return group.send_tensor(tensor=tensor, async_op=async_op)

    def recv_tensor(
        self,
        tensor: torch.Tensor,
        src_group_name: str,
        src_rank: int | list[int],
        async_op: bool = False,
    ):
        """In-place receive of a tensor from a specific worker address in the collective group. This function is optimized for receiving a single tensor and does not introduce metadata communication overhead like recv. But it requires preallocation of the tensor with the correct shape and dtype.

        .. note::
            Do not mix recv_tensor with send

        .. note::
            When transferring GPU objects, the first send_tensor needs to be paired with a recv_tensor at the other end. Calling async send_tensor or recv_tensor first at both ends will result in communication hang, because NCCL communicators are established in a lazy manner when the first pair of send/recv is called.

        .. note::
            This method is not thread safe.

        Args:
            tensor (torch.Tensor): The tensor to receive. It must be preallocated with the correct shape and dtype.
            src_group_name (str): The name of the source worker group.
            src_rank (int | List[int]): The rank or list of ranks in the source worker group to receive the tensor from. For SPMD-like workers, this should be a single rank. For SPSD-like workers forked by parent workers, this can be a list of ranks that forms a path from the root worker to the target worker.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: An AsyncWork object if async_op is True, otherwise None.

        """
        src_addr = WorkerAddress(src_group_name, ranks=src_rank)
        group = self._get_p2p_collective_group(src_addr)
        return group.recv_tensor(tensor=tensor, async_op=async_op)

    def create_channel(
        self,
        channel_name: str,
        node_rank: int = 0,
        maxsize: int = 0,
        local: bool = False,
    ):
        """Create a new channel with the specified placement rank and maximum size.

        Args:
            channel_name (str): The name of the channel.
            node_rank (int): The global rank of the node in the cluster where the channel will be created.
            maxsize (int): The maximum size of the channel queue. Defaults to 0 (unbounded).
            local (bool): Create the channel for intra-process communication. Cannot be connected by other workers.

        Returns:
            Channel: A new instance of the Channel class.

        """
        from ..channel.channel import Channel

        return Channel.create(
            name=channel_name, node_rank=node_rank, maxsize=maxsize, local=local
        )

    def connect_channel(self, channel_name: str):
        """Connect to an existing channel.

        Args:
            channel_name (str): The name of the channel to connect to.

        Returns:
            Channel: An instance of the Channel class connected to the specified channel.

        """
        from ..channel.channel import Channel

        return Channel.connect(channel_name=channel_name, current_worker=self)

    def broadcast(self, object: Optional[Any], ranks: list[int]):
        """Broadcast an object inside the current worker group.

        Args:
            object (Any): The object to broadcast. For non-src ranks, this is None.
            ranks (List[int]): The ranks of the workers to broadcast the object to. The first in the list is the source.
        """
        if not ranks:
            return object

        src_rank = ranks[0]
        if self._rank == src_rank:
            for rank in ranks[1:]:
                self.send(object, self._group_name, rank)
        else:
            object = self.recv(self._group_name, src_rank)
        return object

    def get_name(self) -> str:
        """Convert the WorkerAddress to a string representation.

        Returns:
            str: The string representation of the worker name.

        """
        return self._worker_address.get_name()

    def get_parent_rank(self) -> int:
        """Get the rank of the parent worker in the WorkerAddress.

        Returns:
            int: The rank of the parent worker, or 0 if this is the root worker.

        """
        return self._worker_address.get_parent_rank()

    def log_on_first_rank(self, msg):
        """Log a message only on the first rank of the worker group."""
        if self._rank == 0:
            self._logger.info(msg, stacklevel=self._stacklevel)

    def log_on_last_rank(self, msg):
        """Log a message only on the last rank of the worker group."""
        if self._rank == self._world_size - 1:
            self._logger.info(msg, stacklevel=self._stacklevel)

    def log_debug(self, msg):
        """Log at the debug level."""
        self._logger.debug(msg, stacklevel=self._stacklevel)

    def log_info(self, msg):
        """Log at the info level."""
        self._logger.info(msg, stacklevel=self._stacklevel)

    def log_warning(self, msg):
        """Log at the warning level."""
        self._logger.warning(msg, stacklevel=self._stacklevel)

    def log_error(self, msg):
        """Log at the error level."""
        self._logger.error(msg, stacklevel=self._stacklevel)

    def pop_execution_time(self, tag: str):
        """Retrieve the execution time of a function.

        Args:
            tag (str): The name of the timer to retrieve the execution time for.
        """
        if tag not in self._timer_metrics:
            raise ValueError(f"Timer '{tag}' has not been recorded.")
        return self._timer_metrics.pop(tag)

    @contextmanager
    def worker_timer(self, tag: Optional[str] = None):
        """Context manager to time the execution of a worker function.

        Args:
            tag (str): The name of the timer to record the execution time for. Default is the current function name.
        """
        if tag is None:
            frame_num = 2
            frame = inspect.stack()[frame_num]
            tag = frame.function
        assert tag is not None, "Timer tag must be provided."
        try:
            start_time = time.perf_counter()
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._timer_metrics[tag] = self._timer_metrics.get(tag, 0.0) + duration

    def _check_initialized(self):
        """Check if the Worker has been initialized.

        This is used to ensure that the Worker is ready to be used.
        """
        if not self._has_initialized:
            raise RuntimeError(
                "Worker has not been initialized. Please call Worker.__init__(self) in your class's __init__ method."
            )

    def _init_ray_and_managers(self):
        """When the Worker is not a Ray actor, we need to initialize Ray if it is not already initialized."""
        from ..collective import Collective
        from ..manager import WorkerManager

        if not ray.is_initialized():
            # Initialize Ray if not already initialized
            ray.init(
                address="auto",
                namespace=Cluster.NAMESPACE,
                logging_level=Cluster.LOGGING_LEVEL,
            )

        if (
            self._manager_proxy is None
            or self._collective is None
            or Worker.PID != os.getpid()
        ):
            self._manager_proxy = WorkerManager.get_proxy()
            self._manager_proxy.register_worker(
                self._worker_address, self._get_worker_info()
            )
            self._collective = Collective(self)

            Worker.PID = os.getpid()

    def _setup_local_rank_world_size(self):
        if self._is_ray_actor:
            if os.environ.get("ISOLATE_ACCELERATOR", "0") == "1":
                # Ray limits the number of accelerators per worker to 1, so when calling torch.cuda.set_device(), we must ensure that 0 is passed as the local rank.
                os.environ["LOCAL_RANK"] = "0"
                os.environ["LOCAL_WORLD_SIZE"] = "1"
                self._isolate_gpu = True
            else:
                os.environ["LOCAL_RANK"] = str(
                    self._local_accelerator_rank
                )  # Must use the actual device ID
                os.environ["LOCAL_WORLD_SIZE"] = str(self._node_local_world_size)
                self._isolate_gpu = False

            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        else:
            # These are not set for non-Ray workers
            self._local_rank = -1
            self._local_world_size = -1

    def _setup_master_address_and_port(self):
        # Executed after _init_ray_and_proxies
        from ..manager import WorkerInfo

        if self._is_ray_actor:
            master_worker_address = (
                self._worker_address.get_parent_address().get_child_address(0)
            )
            worker_info: WorkerInfo = None
            count = 0
            while worker_info is None:
                worker_info = self._manager_proxy.get_worker_info(master_worker_address)
                time.sleep(0.001)
                count += 1
                if count % Cluster.TIMEOUT_WARN_TIME == 0:
                    self._logger.warning(
                        f"Waiting for rank 0 of group {self._worker_address.root_group_name} to be up for {count // 1000} seconds"
                    )
            self._master_address = worker_info.node_ip
            self._master_port = worker_info.node_port
            os.environ["MASTER_ADDR"] = self._master_address
            os.environ["MASTER_PORT"] = str(self._master_port)

    def _setup_accelerator_info(self) -> int:
        cluster = Cluster()
        visible_devices = AcceleratorUtil.get_visible_devices(self._accelerator_type)
        node_accelerator_ranks = cluster.accelerator_ranks[self._cluster_node_rank]
        self.global_accelerator_ids = [
            node_accelerator_ranks[local_id] for local_id in visible_devices
        ]

        if not self._is_ray_actor:
            if len(visible_devices) > 0:
                self._local_accelerator_rank = visible_devices[0]
            else:
                self._local_accelerator_rank = -1

    def _setup_hardware(self):
        cluster = Cluster()
        hardware_ranks_str = os.environ.get("LOCAL_HARDWARE_RANKS", "")
        if hardware_ranks_str == "":
            self._hardware_ranks = []
        else:
            self._hardware_ranks = list(map(int, hardware_ranks_str.strip().split(",")))
        node_group_label = os.environ.get("NODE_GROUP_LABEL", None)
        self._node_group = cluster.get_node_group(node_group_label)
        assert self._node_group is not None, (
            f"Node group {node_group_label} not found in cluster. Available node groups: {[node_group.label for node_group in cluster._node_groups]}"
        )

    def _setup_comm_envs(self):
        # Communication devices
        self._comm_devices = Cluster.get_sys_env_var(
            ClusterEnvVar.COMM_NET_DEVICES, None
        )
        if self._comm_devices is not None:
            self.log_info(
                f"Using communication devices for worker {self._worker_name}: {self._comm_devices}"
            )
            # Validate the format of comm devices
            if os.getenv("GLOO_SOCKET_IFNAME") is None:
                os.environ["GLOO_SOCKET_IFNAME"] = self._comm_devices
            elif self._comm_devices != os.environ["GLOO_SOCKET_IFNAME"]:
                self.log_warning(
                    f"GLOO_SOCKET_IFNAME is already set to {os.environ['GLOO_SOCKET_IFNAME']}, ignoring {Cluster.get_full_env_var_name(ClusterEnvVar.COMM_NET_DEVICES)}={self._comm_devices}"
                )

            ccl_socket_env_var = AcceleratorUtil.get_ccl_socket_ifname_env_var(
                self._accelerator_type
            )
            if os.environ.get(ccl_socket_env_var) is None:
                os.environ[ccl_socket_env_var] = self._comm_devices
            elif self._comm_devices != os.environ[ccl_socket_env_var]:
                self.log_warning(
                    f"{ccl_socket_env_var} is already set to {os.environ[ccl_socket_env_var]}, ignoring {Cluster.get_full_env_var_name(ClusterEnvVar.COMM_NET_DEVICES)}={self._comm_devices}"
                )

    def _setup_logging(self):
        self._logger = logging.getLogger(self._worker_name)
        logging_level = Cluster.get_sys_env_var(ClusterEnvVar.LOG_LEVEL, "INFO").upper()
        if logging_level == "DEBUG":
            self._logging_level = logging.DEBUG
        elif logging_level == "INFO":
            self._logging_level = logging.INFO
        elif logging_level == "WARNING":
            self._logging_level = logging.WARNING
        elif logging_level == "ERROR":
            self._logging_level = logging.ERROR
        self._logger.setLevel(self._logging_level)

        self._logger.propagate = False
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=f"[%(levelname)s %(asctime)s {self._worker_address.get_parent_address().get_name()}-Rank-{self._rank}][%(filename)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        Worker.logger = self._logger

    def _register_signal_handlers(self):
        """Register signal handlers for this worker process for more informative debugging."""

        def signal_handler(signum, frame):
            self._logger.error(
                f"Received signal {signum} ({signal.strsignal(signum)}) in worker {self._worker_address}, traceback is below:"
            )
            traceback.print_stack(frame)

            if self._thread is not threading.main_thread():
                # If the Worker is running in a worker thread (e.g., Worker with async functions), print the stack trace of the worker thread
                self._logger.error("Worker thread traceback is below:")
                traceback.print_stack(sys._current_frames()[self._thread.ident])
            os.kill(os.getpid(), signal.SIGKILL)

        should_catch_system_failure = os.environ.get("CATCH_SYSTEM_FAILURE", "0") == "1"
        if not should_catch_system_failure:
            # If the environment variable CATCH_SYSTEM_FAILURE is "0", do not register signal handlers
            return

        try:
            # Register signal handlers for common signals
            signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
            signal.signal(signal.SIGSEGV, signal_handler)  # Handle segmentation fault
            signal.signal(signal.SIGABRT, signal_handler)  # Handle abort signal
            signal.signal(signal.SIGQUIT, signal_handler)  # Handle quit signal
            signal.signal(
                signal.SIGUSR1, signal_handler
            )  # Handle user-defined signal 1
            signal.signal(
                signal.SIGUSR2, signal_handler
            )  # Handle user-defined signal 2
        except ValueError:
            self._logger.warning(
                "Failed to register signal handlers. This may happen if the Worker is not running in the main thread."
            )

    def _set_new_omegaconf_resolvers(self):
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
        OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
        OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)
        OmegaConf.register_new_resolver(
            "torch.dtype", lambda dtype_name: getattr(torch, dtype_name), replace=True
        )

    def _get_p2p_collective_group(self, peer_addr: WorkerAddress):
        """Get a P2P collective group for communication with a peer worker."""
        workers = [self._worker_address, peer_addr]
        # Ensure the order is the same with the same two ranks
        workers = sorted(workers, key=lambda x: x.get_name())
        self._init_ray_and_managers()
        with self._lock:
            return self._collective.create_collective_group(workers)

    def _get_worker_info(self):
        """Get the worker information for local access.

        This method is used to retrieve the worker properties without calling remote functions.
        """
        if self._actor is None and self._is_ray_actor:
            self._actor = ray.get_actor(self._worker_name, namespace=Cluster.NAMESPACE)

        node_ip = ray.util.get_node_ip_address()
        node_port = Cluster.find_free_port()

        from ..manager import WorkerInfo

        return WorkerInfo(
            address=self._worker_address,
            rank=self._rank,
            cluster_node_rank=self._cluster_node_rank,
            accelerator_type=self._accelerator_type,
            accelerator_rank=self._local_accelerator_rank,
            node_ip=node_ip,
            node_port=node_port,
            available_accelerators=self.global_accelerator_ids,
        )
