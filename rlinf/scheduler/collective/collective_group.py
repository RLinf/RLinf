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

import io
import itertools
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, is_dataclass, replace
from pickle import Pickler, Unpickler
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Iterable, Optional

import torch
import torch.distributed as dist
from ray.cloudpickle import Pickler as CloudPickler
from torch.multiprocessing.reductions import reduce_tensor

from ..cluster.utils import (
    DataclassTensorFieldsMetadata,
    extract_dataclass_tensor_fields,
    unflatten_dataclass_tensor_fields,
)
from ..manager import CollectiveGroupInfo, CollectiveManager, WorkerInfo
from ..worker import Worker, WorkerAddress
from .async_work import AsyncFuncWork, AsyncWork

if TYPE_CHECKING:
    from .collective import Collective


@dataclass
class TensorData:
    """Metadata for tensor containers (list, dict, or dataclass with tensor fields).

    Used by TENSOR_LIST, TENSOR_DICT, and DATACLASS_WITH_TENSORS object types
    to pass precomputed device info and optional dataclass-specific fields.
    """

    cpu_tensor_mask: list[bool]
    """Per-tensor mask for CPU placement; used for wire metadata."""

    cpu_tensors: list[torch.Tensor]
    accel_tensors: list[torch.Tensor]
    """Pre-partitioned lists to avoid repeated extraction when sending."""

    # For dataclass
    tensor_fields: Optional[dict[str, Any]] = None
    metadata: Optional[DataclassTensorFieldsMetadata] = None
    tensors_list: Optional[list[torch.Tensor]] = None

    @property
    def has_cpu_tensor(self) -> bool:
        """Whether at least one tensor is on CPU."""
        return bool(self.cpu_tensors)

    @property
    def has_accel_tensor(self) -> bool:
        """Whether at least one tensor is on accelerator."""
        return bool(self.accel_tensors)


@dataclass
class CollectiveGroupOptions:
    """Options for the scheduler collective group.

    For accelerator communication options, see ProcessGroupNCCL.Options.
    """

    accel_cluster_size: Optional[int] = None
    """The cluster size for the accelerator communication."""

    accel_max_ctas: Optional[int] = None
    """The maximum number of collective threads to use for GPU communication via NCCL-like accelerator CCLs.
    Higher value of this option means more GPU computation resource (e.g., SM) consumption but better communication efficiency.
    Lower value of this option means less GPU computation resource (e.g., SM) consumption but worse communication efficiency."""

    accel_min_ctas: Optional[int] = None
    """The minimum number of collective threads to use for GPU communication via NCCL-like accelerator CCLs.
    Similar to accel_max_ctas, but with lower value means less GPU computation resource (e.g., SM) consumption but worse communication efficiency."""

    is_high_priority_stream: bool = False
    """Whether to use a high priority stream for GPU communication via NCCL-like accelerator CCLs."""

    use_ring_broadcast: bool = False
    """If True, force the broadcast tensor-list path to use the ring algorithm:
    the source sends each tensor to the first receiver, which then broadcasts
    it to the other receivers. While the first receiver async-broadcasts
    tensor K, it concurrently receives tensor K+1 from the source. When False
    (default), the ring algorithm is still auto-selected when all receivers
    share the same root WorkerGroup (a common pattern, e.g. actor -> all
    rollout ranks)."""

    def is_empty_options(self) -> bool:
        """Check if the options are empty."""
        empty_options = CollectiveGroupOptions()
        return self == empty_options


class CollectiveWorkQueue:
    """A queue for managing asynchronous collective operations."""

    SEND = 0
    RECV = 1
    BROADCAST = 2

    def __init__(self, comm_type: int, logger: logging.Logger):
        """Initialize the CollectiveWorkQueue.

        Args:
            comm_type (int): The type of the communication (SEND or RECV or BROADCAST).
            logger (logging.Logger): The logger to use for logging messages.

        """
        self._accel_stream = None
        self._stream_ctx = nullcontext()
        self._worker = Worker.current_worker
        self._work_queue: Queue[AsyncFuncWork] = Queue()
        self._work_done = True
        self._type = comm_type
        self._logger = logger
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_queue, daemon=True)
        self._thread.start()

    @property
    def done(self):
        """Check if the work queue is done."""
        return self._work_done

    def enqueue(
        self,
        work: AsyncFuncWork,
        comm_id: int,
        event: Optional[torch.Event] = None,
    ):
        """Enqueue a work to the queue."""
        with self._lock:
            self._work_done = False
            self._work_queue.put((work, comm_id, event))

    def _run_queue(self):
        while True:
            self._lock.acquire()
            lock_has_released = False
            try:
                work, comm_id, event = self._work_queue.get(block=False)
            except Empty:
                self._work_done = True
                lock_has_released = True
                self._lock.release()  # The blocking get should not hold the lock
                work, comm_id, event = self._work_queue.get()
            if not lock_has_released:
                self._lock.release()

            # Create CUDA stream if CUDA is initialized and not created yet
            if (
                self._worker.has_accelerator
                and Worker.torch_platform.is_initialized()
                and self._accel_stream is None
            ):
                self._accel_stream = Worker.torch_platform.Stream()
            if self._accel_stream is not None and isinstance(
                self._stream_ctx, nullcontext
            ):
                self._stream_ctx = Worker.torch_platform.stream(self._accel_stream)

            with self._stream_ctx:
                if event is not None:
                    event.wait(self._accel_stream)
                self._logger.debug(
                    f"Async {'send' if self._type == CollectiveWorkQueue.SEND else 'recv'} ID {comm_id} begins"
                )

                work(None)
                work = None  # The reference to work is released here to avoid potential memory leak

                self._logger.debug(
                    f"Async {'send' if self._type == CollectiveWorkQueue.SEND else 'recv'} ID {comm_id} done"
                )
                self._logger.debug(f"Done comm work {work}")


class CollectiveGroup:
    """Collective group for constructing and performing collective operations."""

    ACCEL: str = Worker.torch_device_type
    CPU: str = "cpu"
    TENSOR: int = 0
    TENSOR_LIST: int = 1
    TENSOR_DICT: int = 2
    OBJECT: int = 3
    DATACLASS_WITH_TENSORS: int = 4
    POOL_SIZE: int = 1

    def __init__(
        self,
        group_info: Optional[CollectiveGroupInfo],
        collective: "Collective",
        group_name: str,
        worker_addresses: list[WorkerAddress],
        cur_worker_address: WorkerAddress,
    ):
        """Initialize the CollectiveGroup.

        Args:
            group_info (CollectiveGroupInfo): The collective group information.
            collective (Collective): The collective instance that owns this group.
            group_name (str): The name of the collective group.
            worker_addresses (List[WorkerAddress]): The addresses of the workers in the group.
            cur_worker_address (WorkerAddress): The address of the current worker.

        """
        self._group_info = group_info
        self._collective = collective
        self._group_name = group_name
        self._worker_addresses = worker_addresses
        self._cur_worker_address = cur_worker_address
        self._mc_group = None
        self._worker = Worker.current_worker
        self._coll_manager = CollectiveManager.get_proxy()
        self._logger = logging.getLogger(cur_worker_address.get_name())
        self._lock = threading.Lock()
        # Lazily populated sub-groups for the hybrid broadcast path.
        # IPC sub-groups: one per (src_rank, same_device_dst_rank) pair.
        self._ipc_sub_groups: dict[tuple[int, int], "CollectiveGroup"] = {}
        # Broadcast sub-groups for different-device workers: keyed by src_rank.
        # Value is (sub_group, src_rank_within_sub_group).
        self._diff_dev_broadcast_sub_groups: dict[
            int, tuple["CollectiveGroup", int]
        ] = {}
        if self._group_info is not None:
            self._init_group()

        self._send_comm_id_iter = itertools.count()
        self._recv_comm_id_iter = itertools.count()
        self._broadcast_comm_id_iter = itertools.count()

        self._send_work_queues = [
            CollectiveWorkQueue(CollectiveWorkQueue.SEND, self._logger)
            for _ in range(CollectiveGroup.POOL_SIZE)
        ]
        self._recv_work_queues = [
            CollectiveWorkQueue(CollectiveWorkQueue.RECV, self._logger)
            for _ in range(CollectiveGroup.POOL_SIZE)
        ]
        self.collective_work_queues = [
            CollectiveWorkQueue(CollectiveWorkQueue.BROADCAST, self._logger)
            for _ in range(CollectiveGroup.POOL_SIZE)
        ]

    def _get_global_accelerator_id(self, device: torch.device | int) -> str:
        """Return the worker placement's global accelerator id for UUID-less platforms."""
        if isinstance(device, torch.device):
            device_idx = device.index
        else:
            device_idx = int(device)

        if device_idx is None:
            device_idx = Worker.torch_platform.current_device()

        global_accelerator_ids = getattr(self._worker, "global_accelerator_ids", [])
        if 0 <= int(device_idx) < len(global_accelerator_ids):
            return str(global_accelerator_ids[int(device_idx)])
        return str(device_idx)

    def _get_accelerator_device_identity(self, device: torch.device | int) -> str:
        """Return a stable identity for comparing accelerator devices across workers."""
        properties = Worker.torch_platform.get_device_properties(device)
        device_uuid = getattr(properties, "uuid", None)
        if device_uuid is not None:
            return str(device_uuid)
        return self._get_global_accelerator_id(device)

    def send(
        self,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        async_op: bool = False,
        options: Optional[CollectiveGroupOptions] = None,
        piggyback_payload: Optional[Any] = None,
    ) -> Optional[AsyncFuncWork]:
        """Implement the Worker's send method.

        The real communication implementation is in the _atomic_send below.

        This function calls _atomic_send in a way so that it can be chained with previous send operations in the same channel.
        Otherwise, async send operations in the same channel may become out-of-order and mismatch with recv.
        """
        # Only iter the channel here and pass the channel id along the way.
        # Because the _atomic_send and all the send in the way may be called asynchronously while the channel_id in the class may be different.
        send_comm_id = next(self._send_comm_id_iter)
        object_type, tensor_data = self._get_object_info(object)

        # Create AsyncFuncWork for the send operation
        send_work = AsyncFuncWork(
            self._atomic_send,
            object=object,
            comm_id=send_comm_id,
            object_type=object_type,
            tensor_data=tensor_data,
            options=options,
            piggyback_payload=piggyback_payload,
            pass_self=True,
        )

        # Capture CUDA event of the main stream if sending accelerator tensors
        if tensor_data.has_accel_tensor:
            send_event = Worker.torch_platform.Event()
            send_event.record()
        else:
            send_event = None

        # Put the send work into queue if the work is async
        # Otherwise, wait for all enqueued works to finish and call the send work synchronously
        work_queue = self._send_work_queues[send_comm_id % CollectiveGroup.POOL_SIZE]
        if async_op:
            work_queue.enqueue(send_work, send_comm_id, send_event)
            return send_work
        else:
            while not work_queue.done:
                continue
            send_work(None)
            self._logger.debug(f"Sync send ID {send_comm_id} done")
            return send_work.wait()

    def _atomic_send(
        self,
        work: AsyncFuncWork,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        comm_id: int,
        object_type: str,
        tensor_data: TensorData,
        options: Optional[CollectiveGroupOptions] = None,
        piggyback_payload: Optional[Any] = None,
    ) -> Optional[AsyncFuncWork]:
        """Send an object to a specific address in the collective group in an out-of-place manner.

        It runs in an atomic way, i.e., communications of two calls of _atomic_send are guaranteed to be in the same ordered as the send API is called.
        """
        self._init_process_group(options=options)
        # First send object type to the destination worker
        object_type_tensor = torch.tensor(object_type, dtype=torch.int, device="cpu")
        self._send(object_type_tensor, CollectiveGroup.CPU, comm_id)
        self._logger.debug(
            f"Sending object type {object_type} from {self._cur_worker_address.get_name()} in group {self._group_info.group_name}"
        )

        if object_type == CollectiveGroup.TENSOR:
            # Out-of-place tensor send/recv is done via tensor list send/recv with a list of one tensor
            return self._send_tensor_list(
                [object],
                comm_id,
                piggyback_payload=piggyback_payload,
                tensor_data=tensor_data,
                work=work,
            )
        elif object_type == CollectiveGroup.TENSOR_LIST:
            return self._send_tensor_list(
                object,
                comm_id,
                piggyback_payload=piggyback_payload,
                tensor_data=tensor_data,
                work=work,
            )
        elif object_type == CollectiveGroup.TENSOR_DICT:
            return self._send_tensor_dict(
                object,
                comm_id,
                tensor_data,
                piggyback_payload=piggyback_payload,
                work=work,
            )
        elif object_type == CollectiveGroup.DATACLASS_WITH_TENSORS:
            return self._send_tensor_dataclass(
                object,
                comm_id,
                tensor_data=tensor_data,
                piggyback_payload=piggyback_payload,
                work=work,
            )
        elif object_type == CollectiveGroup.OBJECT:
            return self._send_object(
                object, comm_id, piggyback_payload=piggyback_payload, work=work
            )
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

    def recv(
        self,
        async_op: bool = False,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> (
        AsyncFuncWork
        | torch.Tensor
        | list[torch.Tensor]
        | dict[str, torch.Tensor]
        | Any
    ):
        """Implement Worker's recv method.

        Similar as the send method above, it ensures the correct ordering of multiple communications of two recv calls.
        """
        recv_comm_id = next(self._recv_comm_id_iter)

        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            current_device = Worker.torch_platform.current_device()
        else:
            current_device = None

        recv_work = AsyncFuncWork(
            self._atomic_recv,
            comm_id=recv_comm_id,
            current_device=current_device,
            options=options,
            pass_self=True,
        )

        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            recv_event = Worker.torch_platform.Event()
            recv_event.record()
        else:
            recv_event = None

        work_queue = self._recv_work_queues[recv_comm_id % CollectiveGroup.POOL_SIZE]
        if async_op:
            work_queue.enqueue(recv_work, recv_comm_id, recv_event)
            return recv_work
        else:
            while not work_queue.done:
                continue
            recv_work(None)
            self._logger.debug(f"Sync recv ID {recv_comm_id} done")
            return recv_work.wait()

    def _atomic_recv(
        self,
        work: AsyncFuncWork,
        comm_id: int,
        current_device: Optional[int],
        options: Optional[CollectiveGroupOptions] = None,
    ) -> (
        AsyncFuncWork
        | torch.Tensor
        | list[torch.Tensor]
        | dict[str, torch.Tensor]
        | Any
    ):
        """Atomic recv implementation."""
        if current_device is not None:
            Worker.torch_platform.set_device(current_device)

        self._init_process_group(options=options)

        # First recv object type
        object_type_tensor = torch.empty(1, dtype=torch.int, device="cpu")
        self._recv(object_type_tensor, CollectiveGroup.CPU, comm_id)

        object_type = object_type_tensor.item()
        self._logger.debug(
            f"Receiving object type {object_type} from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        if object_type == CollectiveGroup.TENSOR:
            tensor, pb_data = self._recv_tensor_list(comm_id, work=work)
            assert len(tensor) == 1, (
                f"Expected to receive one tensor but got {len(tensor)} tensors from Rank {self._peer_rank} in group {self._group_info.group_name}"
            )
            data = tensor[0]
        elif object_type == CollectiveGroup.TENSOR_LIST:
            data, pb_data = self._recv_tensor_list(comm_id, work=work)
        elif object_type == CollectiveGroup.TENSOR_DICT:
            data, pb_data = self._recv_tensor_dict(comm_id, work=work)
        elif object_type == CollectiveGroup.DATACLASS_WITH_TENSORS:
            data, pb_data = self._recv_tensor_dataclass(comm_id, work=work)
        elif object_type == CollectiveGroup.OBJECT:
            data, pb_data = self._recv_object(comm_id, work=work)
        else:
            raise ValueError(f"Unsupported object type: {object_type}")
        if pb_data is not None:
            return data, pb_data
        else:
            return data

    def send_tensor(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> Optional[AsyncFuncWork]:
        """Implement the Worker's send_tensor method.

        It's also a wrapper of _atomic_send_tensor to ensure the correct ordering of multiple send_tensor calls in the same channel.
        """
        send_comm_id = next(self._send_comm_id_iter)
        object_type, tensor_data = self._get_object_info(tensor)

        send_work = AsyncFuncWork(
            self._atomic_send_tensor,
            tensor=tensor,
            comm_id=send_comm_id,
            object_type=object_type,
            tensor_data=tensor_data,
            options=options,
            pass_self=True,
        )

        if tensor_data.has_accel_tensor:
            send_event = Worker.torch_platform.Event()
            send_event.record()
        else:
            send_event = None

        work_queue = self._send_work_queues[send_comm_id % CollectiveGroup.POOL_SIZE]
        if async_op:
            work_queue.enqueue(send_work, send_comm_id, send_event)
            return send_work
        else:
            while not work_queue.done:
                continue
            send_work(None)
            self._logger.debug(f"Sync send_tensor ID {send_comm_id} done")
            return send_work.wait()

    def _atomic_send_tensor(
        self,
        work: AsyncFuncWork,
        tensor: torch.Tensor,
        comm_id: int,
        object_type: str,
        tensor_data: TensorData,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> None:
        """Atomic send_tensor implementation."""
        assert object_type == CollectiveGroup.TENSOR, (
            "The object must be a torch.Tensor when using send_tensor"
        )
        if not tensor.is_contiguous():
            raise ValueError(
                "All tensors must be contiguous when using P2P communication. Otherwise the recv side might recv wrong tensor data. Consider using .contiguous() to make the tensors contiguous."
            )

        self._init_process_group(options=options)
        self._logger.debug(
            f"Sending tensor to Rank {self._peer_rank} in group {self._group_info.group_name}"
        )

        device = (
            CollectiveGroup.ACCEL
            if tensor_data.has_accel_tensor
            else CollectiveGroup.CPU
        )
        with self._track_payload_time(work=work):
            # Handle CUDA tensor sending with IPC if the peer worker is on the same device
            if tensor_data.has_accel_tensor:
                check_cuda_device_result = self._check_same_device_with_peer()
                if check_cuda_device_result == 0:
                    return self._send_single_tensor_to_uncertain_peer(tensor, comm_id)
                elif check_cuda_device_result == 1:
                    return self._send_single_tensor_via_ipc(tensor, comm_id)

            return self._send(tensor, device=device, comm_id=comm_id)

    def recv_tensor(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> Optional[AsyncFuncWork]:
        """Implement Worker's recv_tensor method.

        It's also a wrapper of _atomic_recv_tensor to ensure the correct ordering of multiple recv_tensor calls in the same channel.
        """
        recv_comm_id = next(self._recv_comm_id_iter)

        recv_work = AsyncFuncWork(
            self._atomic_recv_tensor,
            tensor=tensor,
            comm_id=recv_comm_id,
            options=options,
            pass_self=True,
        )

        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            recv_event = Worker.torch_platform.Event()
            recv_event.record()
        else:
            recv_event = None

        work_queue = self._recv_work_queues[recv_comm_id % CollectiveGroup.POOL_SIZE]
        if async_op:
            work_queue.enqueue(recv_work, recv_comm_id, recv_event)
            return recv_work
        else:
            while not work_queue.done:
                continue
            recv_work(None)
            self._logger.debug(f"Sync recv_tensor ID {recv_comm_id} done")
            return recv_work.wait()

    def broadcast(
        self,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        src_addr: WorkerAddress,
        async_op: bool = False,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> (
        AsyncFuncWork
        | torch.Tensor
        | list[torch.Tensor]
        | dict[str, torch.Tensor]
        | Any
    ):
        """Broadcast an object to all workers in the collective group.

        The source rank is inferred as the first worker in the group. The source
        rank should provide the object, and all other ranks should pass None.
        """
        broadcast_comm_id = next(self._broadcast_comm_id_iter)
        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            current_device = Worker.torch_platform.current_device()
        else:
            current_device = None
        broadcast_work = AsyncFuncWork(
            self._atomic_broadcast,
            object=object,
            src_addr=src_addr,
            comm_id=broadcast_comm_id,
            current_device=current_device,
            options=options,
            pass_self=True,
        )

        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            broadcast_event = Worker.torch_platform.Event()
            broadcast_event.record()
        else:
            broadcast_event = None

        work_queue = self.collective_work_queues[
            broadcast_comm_id % CollectiveGroup.POOL_SIZE
        ]
        if async_op:
            work_queue.enqueue(broadcast_work, broadcast_comm_id, broadcast_event)
            return broadcast_work
        else:
            while not work_queue.done:
                continue
            broadcast_work(None)
            self._logger.debug(f"Sync broadcast ID {broadcast_comm_id} done")
            return broadcast_work.wait()

    def _atomic_broadcast(
        self,
        work: AsyncFuncWork,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        src_addr: WorkerAddress,
        comm_id: int,
        current_device: Optional[int],
        options: Optional[CollectiveGroupOptions] = None,
    ) -> torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any:
        if current_device is not None:
            Worker.torch_platform.set_device(current_device)

        self._init_process_group(options=options)
        src_rank = self._worker_addresses.index(src_addr)

        object_type_tensor = torch.empty(1, dtype=torch.int, device="cpu")
        if self._rank == src_rank:
            object_type, _ = self._get_object_info(object)
            object_type_tensor.fill_(object_type)

        self._broadcast(
            object_type_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            src_rank=src_rank,
        )
        object_type = object_type_tensor.item()

        if object_type == CollectiveGroup.TENSOR:
            tensor_list = [object] if self._rank == src_rank else None
            return self._broadcast_tensor_list(
                tensor_list,
                comm_id=comm_id,
                src_rank=src_rank,
                options=options,
                work=work,
            )[0]
        elif object_type == CollectiveGroup.TENSOR_LIST:
            tensor_list = object if self._rank == src_rank else None
            return self._broadcast_tensor_list(
                tensor_list,
                comm_id=comm_id,
                src_rank=src_rank,
                options=options,
                work=work,
            )
        elif object_type == CollectiveGroup.TENSOR_DICT:
            tensor_dict = object if self._rank == src_rank else None
            return self._broadcast_tensor_dict(
                tensor_dict,
                comm_id=comm_id,
                src_rank=src_rank,
                options=options,
                work=work,
            )
        elif object_type == CollectiveGroup.DATACLASS_WITH_TENSORS:
            tensor_dataclass = object if self._rank == src_rank else None
            return self._broadcast_tensor_dataclass(
                tensor_dataclass,
                comm_id=comm_id,
                src_rank=src_rank,
                options=options,
                work=work,
            )
        elif object_type == CollectiveGroup.OBJECT:
            return self._broadcast_object(
                object, comm_id=comm_id, src_rank=src_rank, work=work
            )
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

    def _broadcast_tensor_list(
        self,
        tensors: Optional[list[torch.Tensor]],
        comm_id: int,
        src_rank: int,
        options: Optional[CollectiveGroupOptions] = None,
        work: Optional[AsyncFuncWork] = None,
    ) -> list[torch.Tensor]:
        """Broadcast a list of tensors from src_rank to all ranks."""
        if self._rank == src_rank and tensors is None:
            raise ValueError("Source rank must provide tensors for broadcast.")
        metadata_size = torch.empty(1, dtype=torch.long, device="cpu")
        if self._rank == src_rank:
            cpu_tensor_mask = [tensor.device.type == "cpu" for tensor in tensors]
            tensor_shape_dtype = [(tensor.shape, tensor.dtype) for tensor in tensors]
            metadata = {
                "meta": tensor_shape_dtype,
                "cpu_tensor_mask": cpu_tensor_mask,
            }
            metadata_tensor, metadata_size = self._object_to_tensor(metadata, "cpu")
        self._broadcast(
            metadata_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            src_rank=src_rank,
        )
        metadata_tensor = (
            metadata_tensor
            if self._rank == src_rank
            else torch.empty(metadata_size.item(), dtype=torch.uint8, device="cpu")
        )
        self._broadcast(
            metadata_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            src_rank=src_rank,
        )
        metadata = self._tensor_to_object(metadata_tensor, metadata_size)

        tensor_shapes = metadata["meta"]
        cpu_tensor_mask = metadata["cpu_tensor_mask"]
        has_accel_tensor = any(not m for m in cpu_tensor_mask)

        if self._should_use_ring_broadcast(src_rank, options, has_accel_tensor):
            if self._rank == src_rank:
                ring_tensors = tensors
            else:
                ring_tensors = [
                    torch.empty(
                        shape,
                        dtype=dtype,
                        device=(
                            "cpu"
                            if is_cpu
                            else Worker.torch_platform.current_device()
                            if has_accel_tensor
                            else "cpu"
                        ),
                    )
                    for (shape, dtype), is_cpu in zip(tensor_shapes, cpu_tensor_mask)
                ]
            with self._track_payload_time(work=work):
                return self._ring_broadcast_tensor_list(
                    ring_tensors,
                    comm_id=comm_id,
                    src_rank=src_rank,
                    tensor_shapes=tensor_shapes,
                    cpu_tensor_mask=cpu_tensor_mask,
                    options=options,
                )

        # CPU-only payload, no ring: fan out async ``_send`` from src to every
        # receiver in parallel over per-receiver pair sub-groups. Receivers
        # issue sequential ``_recv`` on their own pair sub-group. For Gloo
        # this is typically faster than a full-group collective broadcast.
        if not has_accel_tensor:
            if self._rank == src_rank:
                broadcast_tensors = tensors
            else:
                broadcast_tensors = [
                    torch.empty(shape, dtype=dtype, device="cpu")
                    for (shape, dtype) in tensor_shapes
                ]
            num_workers = len(self._group_info.workers)
            receivers = [i for i in range(num_workers) if i != src_rank]
            with self._track_payload_time(work=work):
                if self._rank == src_rank:
                    pending: list = []
                    for r in receivers:
                        pg = self._get_or_create_ipc_sub_group(src_rank, r)
                        pg._init_process_group(options=options)
                        for tensor in broadcast_tensors:
                            handle = pg._send(
                                tensor,
                                device=CollectiveGroup.CPU,
                                comm_id=next(pg._send_comm_id_iter),
                                async_op=True,
                            )
                            if handle is not None:
                                pending.append(handle)
                    for h in pending:
                        h.wait()
                elif self._rank in receivers:
                    pg = self._get_or_create_ipc_sub_group(src_rank, self._rank)
                    pg._init_process_group(options=options)
                    for tensor in broadcast_tensors:
                        pg._recv(
                            tensor,
                            device=CollectiveGroup.CPU,
                            comm_id=next(pg._recv_comm_id_iter),
                        )
            return broadcast_tensors

        if has_accel_tensor:
            definitely_same_ranks, uncertain_ranks, diff_dev_ranks = (
                self._classify_broadcast_ranks(src_rank)
            )
            # Different-device receivers may still share an accelerator with each
            # other (just not with src). NCCL rejects a collective with two ranks
            # on one accelerator, so group them by device: only one representative
            # per distinct device joins the accelerator collective with src, then
            # re-broadcasts to its same-device peers via CUDA IPC.
            diff_dev_groups = self._group_ranks_by_device(diff_dev_ranks)
            diff_dev_repr_ranks = [group[0] for group in diff_dev_groups]
            diff_dev_ipc_recv_ranks = [
                rank for group in diff_dev_groups for rank in group[1:]
            ]
        else:
            definitely_same_ranks, uncertain_ranks, diff_dev_ranks = [], [], []
            diff_dev_groups = []
            diff_dev_repr_ranks = []
            diff_dev_ipc_recv_ranks = []

        # Non-src ranks that will receive accel tensors via the IPC hybrid path
        # (definitely-same-device, uncertain, or a different-device non-representative
        # peer) get freshly-allocated tensors back from the recv functions and
        # overwrite their slots in `broadcast_tensors`. Preallocating accel buffers
        # for those slots here would just churn device memory on large weight syncs,
        # so defer the allocation. CPU slots are still preallocated since the CPU
        # broadcast at the end of this function receives into them in-place.
        skip_accel_prealloc = self._rank != src_rank and (
            self._rank in definitely_same_ranks
            or self._rank in uncertain_ranks
            or self._rank in diff_dev_ipc_recv_ranks
        )

        if self._rank == src_rank:
            broadcast_tensors = tensors
        else:
            broadcast_tensors = [
                None
                if (not is_cpu and skip_accel_prealloc)
                else torch.empty(
                    shape,
                    dtype=dtype,
                    device=(
                        "cpu"
                        if is_cpu
                        else Worker.torch_platform.current_device()
                        if has_accel_tensor
                        else "cpu"
                    ),
                )
                for (shape, dtype), is_cpu in zip(tensor_shapes, cpu_tensor_mask)
            ]

        with self._track_payload_time(work=work):
            if (
                not definitely_same_ranks
                and not uncertain_ranks
                and not diff_dev_ipc_recv_ranks
            ):
                # Every receiver sits on a distinct accelerator from src and from
                # each other: a straightforward full-group collective is safe.
                for idx, tensor in enumerate(broadcast_tensors):
                    self._broadcast(
                        tensor,
                        device=CollectiveGroup.CPU
                        if cpu_tensor_mask[idx]
                        else CollectiveGroup.ACCEL,
                        comm_id=comm_id,
                        src_rank=src_rank,
                    )
            else:
                # Hybrid path:
                #   - definitely-same-device workers: P2P IPC from src
                #   - uncertain workers (overlapping multi-device sets): exchange current device
                #     at runtime, then IPC or accelerator P2P send/recv per pair
                #   - different-device workers: one representative per distinct accelerator joins
                #     an accelerator collective with src; that representative then re-broadcasts
                #     to the remaining same-device receivers via CUDA IPC. This guarantees the
                #     collective never contains two ranks on the same accelerator.
                accel_tensors = [
                    t
                    for t, is_cpu in zip(broadcast_tensors, cpu_tensor_mask)
                    if not is_cpu
                ]
                accel_tensor_shapes = [
                    (shape, dtype)
                    for (shape, dtype), is_cpu in zip(tensor_shapes, cpu_tensor_mask)
                    if not is_cpu
                ]

                if self._rank == src_rank:
                    # P2P IPC send to each definitely-same-device receiver.
                    for dst in definitely_same_ranks:
                        ipc_grp = self._get_or_create_ipc_sub_group(src_rank, dst)
                        ipc_grp._init_process_group(options=options)
                        ipc_grp._send_tensor_list_via_ipc(
                            accel_tensors, next(ipc_grp._send_comm_id_iter)
                        )
                    # Uncertain receivers: exchange current device then route per pair.
                    for dst in uncertain_ranks:
                        ipc_grp = self._get_or_create_ipc_sub_group(src_rank, dst)
                        ipc_grp._init_process_group(options=options)
                        ipc_grp._send_tensor_list_to_uncertain_peer(
                            accel_tensors, next(ipc_grp._send_comm_id_iter)
                        )
                    # Collective broadcast to one representative per distinct device.
                    if diff_dev_repr_ranks:
                        sub_grp, sub_src = (
                            self._get_or_create_diff_dev_broadcast_sub_group(
                                src_rank, diff_dev_repr_ranks
                            )
                        )
                        sub_grp._init_process_group(options=options)
                        sub_comm_id = next(sub_grp._broadcast_comm_id_iter)
                        for tensor in accel_tensors:
                            sub_grp._broadcast(
                                tensor, CollectiveGroup.ACCEL, sub_comm_id, sub_src
                            )
                elif self._rank in definitely_same_ranks:
                    # P2P IPC receive from source; replace pre-allocated slots.
                    ipc_grp = self._get_or_create_ipc_sub_group(src_rank, self._rank)
                    ipc_grp._init_process_group(options=options)
                    received = ipc_grp._recv_tensor_list_via_ipc(
                        next(ipc_grp._recv_comm_id_iter)
                    )
                    accel_iter = iter(received)
                    for i, is_cpu in enumerate(cpu_tensor_mask):
                        if not is_cpu:
                            broadcast_tensors[i] = next(accel_iter)
                elif self._rank in uncertain_ranks:
                    # Exchange current device with source, then receive via IPC or accelerator P2P.
                    ipc_grp = self._get_or_create_ipc_sub_group(src_rank, self._rank)
                    ipc_grp._init_process_group(options=options)
                    received = ipc_grp._recv_tensor_list_to_uncertain_peer(
                        accel_tensor_shapes, next(ipc_grp._recv_comm_id_iter)
                    )
                    accel_iter = iter(received)
                    for i, is_cpu in enumerate(cpu_tensor_mask):
                        if not is_cpu:
                            broadcast_tensors[i] = next(accel_iter)
                elif self._rank in diff_dev_repr_ranks:
                    # Different-device representative: receive via the collective
                    # sub-group, then re-broadcast to same-device peers via IPC.
                    sub_grp, sub_src = self._get_or_create_diff_dev_broadcast_sub_group(
                        src_rank, diff_dev_repr_ranks
                    )
                    sub_grp._init_process_group(options=options)
                    sub_comm_id = next(sub_grp._broadcast_comm_id_iter)
                    for tensor in accel_tensors:
                        sub_grp._broadcast(
                            tensor, CollectiveGroup.ACCEL, sub_comm_id, sub_src
                        )
                    # The accelerator collective only enqueues the broadcast on the
                    # stream; force completion before exposing the tensors over IPC
                    # so same-device peers don't read stale memory.
                    same_device_peers = next(
                        group[1:] for group in diff_dev_groups if group[0] == self._rank
                    )
                    if same_device_peers:
                        Worker.torch_platform.current_stream().synchronize()
                        for peer in same_device_peers:
                            ipc_grp = self._get_or_create_ipc_sub_group(
                                self._rank, peer
                            )
                            ipc_grp._init_process_group(options=options)
                            ipc_grp._send_tensor_list_via_ipc(
                                accel_tensors, next(ipc_grp._send_comm_id_iter)
                            )
                else:
                    # Different-device non-representative: receive via IPC from this
                    # accelerator's representative (no collective involvement).
                    repr_rank = next(
                        group[0] for group in diff_dev_groups if self._rank in group
                    )
                    ipc_grp = self._get_or_create_ipc_sub_group(repr_rank, self._rank)
                    ipc_grp._init_process_group(options=options)
                    received = ipc_grp._recv_tensor_list_via_ipc(
                        next(ipc_grp._recv_comm_id_iter)
                    )
                    accel_iter = iter(received)
                    for i, is_cpu in enumerate(cpu_tensor_mask):
                        if not is_cpu:
                            broadcast_tensors[i] = next(accel_iter)

                # CPU tensors still go through the full-group CPU collective.
                for idx, tensor in enumerate(broadcast_tensors):
                    if cpu_tensor_mask[idx]:
                        self._broadcast(tensor, CollectiveGroup.CPU, comm_id, src_rank)

        return broadcast_tensors

    def _broadcast_tensor_dict(
        self,
        tensor_dict: Optional[dict[str, torch.Tensor]],
        comm_id: int,
        src_rank: int,
        options: Optional[CollectiveGroupOptions] = None,
        work: Optional[AsyncFuncWork] = None,
    ) -> dict[str, torch.Tensor]:
        """Broadcast a dictionary of tensors from src_rank to all ranks."""
        keys = list(tensor_dict.keys()) if self._rank == src_rank else None
        keys = self._broadcast_object(keys, comm_id=comm_id, src_rank=src_rank)
        values = list(tensor_dict.values()) if self._rank == src_rank else None
        values = self._broadcast_tensor_list(
            values,
            comm_id=comm_id,
            src_rank=src_rank,
            options=options,
            work=work,
        )
        if len(keys) != len(values):
            raise RuntimeError(
                f"Broadcast received {len(values)} values but {len(keys)} keys from Rank {src_rank} in group {self._group_info.group_name}"
            )
        return dict(zip(keys, values))

    def _broadcast_object(
        self,
        object: Any,
        comm_id: int,
        src_rank: int,
        work: Optional[AsyncFuncWork] = None,
    ) -> Any:
        """Broadcast a Python object from src_rank to all ranks."""
        object_size = torch.empty(1, dtype=torch.long, device="cpu")
        if self._rank == src_rank:
            object_tensor, object_size = self._object_to_tensor(object, "cpu")
        self._broadcast(
            object_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            src_rank=src_rank,
        )
        object_tensor = (
            object_tensor
            if self._rank == src_rank
            else torch.empty(object_size.item(), dtype=torch.uint8, device="cpu")
        )
        with self._track_payload_time(work=work):
            self._broadcast(
                object_tensor,
                device=CollectiveGroup.CPU,
                comm_id=comm_id,
                src_rank=src_rank,
            )
        if self._rank == src_rank:
            return object
        return self._tensor_to_object(object_tensor, object_size)

    def _atomic_recv_tensor(
        self,
        work: AsyncFuncWork,
        tensor: torch.Tensor,
        comm_id: int,
        options: Optional[CollectiveGroupOptions] = None,
    ) -> None:
        """Atomic recv_tensor implementation."""
        object_type, tensor_data = self._get_object_info(tensor)
        assert object_type == CollectiveGroup.TENSOR, (
            "The object must be a torch.Tensor"
        )

        self._init_process_group(options=options)
        self._logger.debug(
            f"Receiving tensor from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        device = (
            CollectiveGroup.ACCEL
            if tensor_data.has_accel_tensor
            else CollectiveGroup.CPU
        )
        with self._track_payload_time(work=work):
            if tensor_data.has_accel_tensor:
                check_cuda_device_result = self._check_same_device_with_peer()
                if check_cuda_device_result == 0:
                    return self._recv_single_tensor_to_uncertain_peer(tensor, comm_id)
                elif check_cuda_device_result == 1:
                    # The peer worker is on the same device, so we need to use CUDA IPC to receive the tensors
                    return self._recv_single_tensor_via_ipc(tensor, comm_id)
            return self._recv(tensor, device=device, comm_id=comm_id)

    def _send(
        self, tensor: torch.Tensor, device: str, comm_id: int, async_op: bool = False
    ):
        """Wrap the actual send operation to hide internal API changes."""
        channel_id = comm_id % CollectiveGroup.POOL_SIZE
        return self._mc_group.send(
            tensor=tensor, device=device, channel_id=channel_id, async_op=async_op
        )

    def _recv(
        self, tensor: torch.Tensor, device: str, comm_id: int, async_op: bool = False
    ):
        """Wrap the actual recv operation to hide internal API changes."""
        channel_id = comm_id % CollectiveGroup.POOL_SIZE
        return self._mc_group.recv(
            tensor=tensor, device=device, channel_id=channel_id, async_op=async_op
        )

    def _broadcast(
        self,
        tensor: torch.Tensor,
        device: str,
        comm_id: int,
        src_rank: int,
        async_op: bool = False,
    ):
        """Wrap the actual broadcast operation to hide internal API changes."""
        channel_id = comm_id % CollectiveGroup.POOL_SIZE
        return self._mc_group.broadcast(
            tensor=tensor,
            device=device,
            channel_id=channel_id,
            src=src_rank,
            async_op=async_op,
        )

    def _init_group(self):
        if self._group_info is None:
            master_worker_address = self._worker_addresses[0]
            if self._cur_worker_address == master_worker_address:
                # Create the group if I'm the master worker
                workers: list[WorkerInfo] = []
                for address in self._worker_addresses:
                    worker_info = self._collective._get_worker_info_safe(address)
                    workers.append(worker_info)

                master_addr = workers[0].node_ip

                group_info = CollectiveGroupInfo(
                    group_name=self._group_name,
                    workers=workers,
                    master_addr=master_addr,
                )

                self._coll_manager.register_collective_group(group_info)
                self._logger.debug(
                    f"Collective group {self._group_name} created with workers: {[worker.get_name() for worker in self._worker_addresses]}"
                )
            else:
                # Wait for the master worker to create the group
                group_info = self._collective._get_group_info_safe(self._group_name)
                self._logger.debug(
                    f"Collective group {self._group_name} found with workers: {[worker.get_name() for worker in self._worker_addresses]}"
                )

            self._group_info = group_info

        if self._mc_group is None:
            self._rank = -1
            for i, worker in enumerate(self._group_info.workers):
                if worker.address == self._cur_worker_address:
                    self._rank = i
                    break
            self._peer_rank = 1 if self._rank == 0 else 0

            from .multi_channel_pg import MultiChannelProcessGroup

            self._mc_group: MultiChannelProcessGroup = MultiChannelProcessGroup(
                cur_rank=self._rank,
                num_channels=CollectiveGroup.POOL_SIZE,
                group_info=self._group_info,
                logger=self._logger,
            )

    def _init_process_group(
        self, options: Optional[CollectiveGroupOptions] = None
    ) -> dist.ProcessGroup:
        """Initialize the process group for collective operations."""
        with self._lock:
            self._init_group()
            if self._mc_group.is_initialized:
                return

            from ..cluster import Cluster

            if self._rank == 0:
                master_port = self._worker.acquire_free_port()
                self._coll_manager.set_master_port_info(
                    self._group_info.group_name, master_port
                )
            else:
                master_port = None
                count = 0
                while master_port is None:
                    master_port = self._coll_manager.get_master_port_info(
                        self._group_info.group_name
                    )
                    time.sleep(0.001)
                    count += 1
                    if count % Cluster.TIMEOUT_WARN_TIME == 0:
                        self._logger.warning(
                            f"Waiting for master port for collective group {self._group_info.group_name} to be set for {count // 1000} seconds"
                        )

            self._logger.debug(
                f"Initializing process group for collective group {self._group_info.group_name}, master address {self._group_info.master_addr}, master port {master_port}, world size {self._group_info.world_size}, rank {self._rank}"
            )

            self._mc_group.init(
                init_method=f"tcp://{self._group_info.master_addr}:{master_port}",
                world_size=self._group_info.world_size,
                rank=self._rank,
                group_name=self._group_info.group_name,
                options=options,
            )

            self._logger.debug(
                f"Process group {self._group_info.group_name} initialized successfully."
            )

            if self._rank == 0:
                # Avoid using the same master port for the next group
                self._coll_manager.reset_master_port_info(self._group_info.group_name)

    @contextmanager
    def _track_payload_time(self, work: Optional[AsyncFuncWork]):
        """Attribute payload-transfer time to ``work`` (no-op if ``work`` is None).

        Wrap only the payload (tensor data) sends/receives/broadcasts -- callers
        must exchange any metadata *before* entering this block so the measured
        window is exactly ``end-of-metadata -> end-of-payload``. CPU-side only:
        for accelerator transfers this captures submit/enqueue cost, not GPU
        transfer wall time, by design (no stream synchronization).
        """
        if work is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            work.add_perf_time(time.perf_counter() - start)

    def _partition_tensors(
        self, tensors: list[torch.Tensor]
    ) -> tuple[list[bool], list[torch.Tensor], list[torch.Tensor]]:
        """Partition tensors by device.

        Returns:
            (cpu_tensor_mask, cpu_tensors, accel_tensors).
        """
        cpu_tensor_mask: list[bool] = []
        cpu_tensors: list[torch.Tensor] = []
        accel_tensors: list[torch.Tensor] = []
        for t in tensors:
            cpu_tensor_mask.append(t.is_cpu)
            if t.is_cpu:
                cpu_tensors.append(t)
            else:
                accel_tensors.append(t)
        return cpu_tensor_mask, cpu_tensors, accel_tensors

    def _get_object_info(self, object: torch.Tensor | Any) -> tuple[int, TensorData]:
        """Classify the object and build precomputed tensor metadata.

        Returns:
            (object_type, tensor_data). tensor_data is always set; for OBJECT
            it has empty cpu/accel lists.
        """
        object_type = CollectiveGroup.OBJECT
        tensor_data = TensorData(
            cpu_tensor_mask=[],
            cpu_tensors=[],
            accel_tensors=[],
        )

        if isinstance(object, torch.Tensor):
            cpu_tensor_mask, cpu_tensors, accel_tensors = self._partition_tensors(
                [object]
            )
            self._check_tensor_contiguous(accel_tensors + cpu_tensors)
            object_type = CollectiveGroup.TENSOR
            tensor_data = TensorData(
                cpu_tensor_mask=cpu_tensor_mask,
                cpu_tensors=cpu_tensors,
                accel_tensors=accel_tensors,
            )

        elif (isinstance(object, list) or isinstance(object, tuple)) and all(
            isinstance(item, torch.Tensor) for item in object
        ):
            cpu_tensor_mask, cpu_tensors, accel_tensors = self._partition_tensors(
                list(object)
            )
            self._check_tensor_contiguous(accel_tensors + cpu_tensors)
            object_type = CollectiveGroup.TENSOR_LIST
            tensor_data = TensorData(
                cpu_tensor_mask=cpu_tensor_mask,
                cpu_tensors=cpu_tensors,
                accel_tensors=accel_tensors,
            )

        elif isinstance(object, dict) and all(
            isinstance(item, torch.Tensor) for item in object.values()
        ):
            values = list(object.values())
            cpu_tensor_mask, cpu_tensors, accel_tensors = self._partition_tensors(
                values
            )
            self._check_tensor_contiguous(accel_tensors + cpu_tensors)
            object_type = CollectiveGroup.TENSOR_DICT
            tensor_data = TensorData(
                cpu_tensor_mask=cpu_tensor_mask,
                cpu_tensors=cpu_tensors,
                accel_tensors=accel_tensors,
            )

        elif is_dataclass(object):
            tensor_fields, tensors_list, metadata = extract_dataclass_tensor_fields(
                object
            )
            if tensor_fields:
                (
                    cpu_tensor_mask,
                    cpu_tensors,
                    accel_tensors,
                ) = self._partition_tensors(tensors_list)
                self._check_tensor_contiguous(accel_tensors + cpu_tensors)
                object_type = CollectiveGroup.DATACLASS_WITH_TENSORS
                tensor_data = TensorData(
                    cpu_tensor_mask=cpu_tensor_mask,
                    cpu_tensors=cpu_tensors,
                    accel_tensors=accel_tensors,
                    tensor_fields=tensor_fields,
                    metadata=metadata,
                    tensors_list=tensors_list,
                )

        return object_type, tensor_data

    def _check_tensor_contiguous(self, tensors: Iterable[torch.Tensor]):
        """Check if the tensors are contiguous."""
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError(
                "All tensors must be contiguous when using P2P communication. Otherwise the recv side might recv wrong tensor data. Consider using .contiguous() to make the tensors contiguous."
            )

    def _check_same_device_with_peer(self):
        """Check if the current worker and the peer worker are on the same device.

        Returns:
            int: -1 means no common device; 0 means have common devices, but not sure if the tensor will be on the same device (the worker has multiple devices); 1 means the two workers are on the same device.

        """
        # Always use NCCL instead of CUDA IPC, even for colocated workers
        return -1

    def _classify_broadcast_ranks(
        self, src_rank: int
    ) -> tuple[list[int], list[int], list[int]]:
        """Classify non-src ranks by their device relationship to src_rank.

        Returns:
            Tuple of (definitely_same, uncertain, definitely_diff):
            - definitely_same: same node, both have exactly one accelerator, identical
              device → use P2P IPC directly.
            - uncertain: same node, accelerator sets overlap but at least one side has
              multiple accelerators → exchange current-device at runtime to decide.
            - definitely_diff: different node or no accelerator overlap → accelerator collective.
        """
        src = self._group_info.workers[src_rank]
        src_devices = set(src.available_accelerators)
        definitely_same: list[int] = []
        uncertain: list[int] = []
        definitely_diff: list[int] = []
        for i, worker in enumerate(self._group_info.workers):
            if i == src_rank:
                continue
            if worker.cluster_node_rank != src.cluster_node_rank or not (
                src_devices & set(worker.available_accelerators)
            ):
                definitely_diff.append(i)
            elif (
                len(src.available_accelerators) == 1
                and len(worker.available_accelerators) == 1
            ):
                definitely_same.append(i)
            else:
                uncertain.append(i)
        return definitely_same, uncertain, definitely_diff

    def _group_ranks_by_device(self, ranks: list[int]) -> list[list[int]]:
        """Group ranks that are guaranteed to share the same accelerator.

        Two ranks share an accelerator when they sit on the same cluster node and
        each exposes exactly one (identical) accelerator id. Multi-accelerator
        workers cannot be matched statically -- their runtime device is unknown
        until the tensor is produced -- so each forms its own singleton group.

        Args:
            ranks: Ranks (indices within this collective group) to partition.

        Returns:
            A list of groups; within each group the first rank is the device
            representative. Group order and membership are deterministic across
            workers (they derive solely from ``group_info``), so every process
            computes the same partition.
        """
        groups: dict[tuple[int, int], list[int]] = {}
        singletons: list[list[int]] = []
        for rank in ranks:
            worker = self._group_info.workers[rank]
            accelerators = worker.available_accelerators
            if len(accelerators) == 1:
                key = (worker.cluster_node_rank, accelerators[0])
                groups.setdefault(key, []).append(rank)
            else:
                singletons.append([rank])
        return list(groups.values()) + singletons

    def _get_or_create_ipc_sub_group(
        self, src_rank: int, dst_rank: int
    ) -> "CollectiveGroup":
        """Return (creating if needed) the 2-worker CollectiveGroup for IPC.

        The group spans src_rank and dst_rank within the broadcast group.
        """
        key = (src_rank, dst_rank)
        if key not in self._ipc_sub_groups:
            src_address = self._group_info.workers[src_rank].address
            dst_address = self._group_info.workers[dst_rank].address
            self._ipc_sub_groups[key] = self._collective.create_collective_group(
                sorted([src_address, dst_address])
            )
        return self._ipc_sub_groups[key]

    def _get_or_create_diff_dev_broadcast_sub_group(
        self, src_rank: int, diff_dev_repr_ranks: list[int]
    ) -> tuple["CollectiveGroup", int]:
        """Return (creating if needed) the accelerator collective sub-group for different-device workers.

        The group spans src_rank and one representative per distinct accelerator
        among the different-device receivers, so no two members share an
        accelerator. Returns the sub-group and src's rank index within it.
        """
        if src_rank not in self._diff_dev_broadcast_sub_groups:
            src_address = self._group_info.workers[src_rank].address
            diff_addresses = [
                self._group_info.workers[r].address for r in diff_dev_repr_ranks
            ]
            sub_addresses = sorted([src_address] + diff_addresses)
            sub_src_rank = sub_addresses.index(src_address)
            sub_grp = self._collective.create_collective_group(sub_addresses)
            self._diff_dev_broadcast_sub_groups[src_rank] = (sub_grp, sub_src_rank)
        return self._diff_dev_broadcast_sub_groups[src_rank]

    def _should_use_ring_broadcast(
        self,
        src_rank: int,
        options: Optional[CollectiveGroupOptions],
        has_accel_tensor: bool,
    ) -> bool:
        """Decide whether to use the ring broadcast algorithm for a tensor-list broadcast.

        Ring is used when explicitly requested via ``options.use_ring_broadcast`` or when
        the receivers (non-src ranks) all belong to the same root WorkerGroup. The latter
        is the common cross-WorkerGroup pattern (e.g. actor rank 0 -> all rollout ranks)
        and benefits from a single cross-group hop followed by an intra-group broadcast.

        Ring is force-disabled in two cases:

        - The src shares a device with any receiver: the src->first_receiver hop
          relies on plain ``_send``/``_recv`` over a 2-worker sub-group, and NCCL
          rejects same-device send/recv. In that case the existing hybrid path
          (which routes same-device pairs through IPC) is the correct choice.
        - The payload contains accelerator tensors *and* every participant has the
          same accelerator type and model. Such a homogeneous cluster supports a
          native NCCL/CCL collective broadcast, which is at least as good as the
          ring hop+broadcast pair. Ring is intended for the heterogeneous case
          (mixed GPU models, mixed accelerator vendors, or no shared CCL).
        """
        if self._group_info is None:
            return False
        num_workers = len(self._group_info.workers)
        receivers = [i for i in range(num_workers) if i != src_rank]
        if len(receivers) < 2:
            return False  # nothing to gain over a direct send/broadcast

        same, uncertain, diff = self._classify_broadcast_ranks(src_rank)
        if same or uncertain:
            return False

        # Receiver-to-receiver same-device sharing would force the ring fan-out to
        # issue same-device P2P send/recv (which NCCL rejects); fall back to the
        # hybrid IPC path, which routes such pairs through CUDA IPC.
        if has_accel_tensor and any(
            len(group) > 1 for group in self._group_ranks_by_device(diff)
        ):
            return False

        if has_accel_tensor:
            participants = [self._group_info.workers[i] for i in [src_rank, *receivers]]
            ref_type = participants[0].accelerator_type
            ref_model = participants[0].accelerator_model
            homogeneous = all(
                w.accelerator_type == ref_type and w.accelerator_model == ref_model
                for w in participants
            )
            if homogeneous:
                return False

        if options is not None and options.use_ring_broadcast:
            return True
        return False

    def _ring_broadcast_tensor_list(
        self,
        broadcast_tensors: list[torch.Tensor],
        comm_id: int,
        src_rank: int,
        tensor_shapes: list[tuple[torch.Size, torch.dtype]],
        cpu_tensor_mask: list[bool],
        options: Optional[CollectiveGroupOptions],
    ) -> list[torch.Tensor]:
        """Two-hop fan-out of a tensor list, one tensor at a time.

        1. ``src_rank`` sends each tensor in order to the first receiver
           over a 2-worker hop sub-group.
        2. The first receiver, after receiving tensor K, fires async
           ``_send`` to every other receiver in parallel (one per-pair
           sub-group, so they share no communicator and run on independent
           streams), then immediately loops back to receive tensor K+1 from
           the source. The host-side recv of K+1 overlaps with the
           in-flight async sends of K.
        3. Other receivers each issue a sequential ``_recv`` from the first
           receiver on their own pair sub-group.

        The two-hop structure runs regardless of tensor device; the
        non-ring CPU-only path in :py:meth:`_broadcast_tensor_list` handles
        the "just fan out from src directly" case.
        """
        del comm_id, tensor_shapes  # both unused; we draw fresh per-sub-group ids below

        num_workers = len(self._group_info.workers)
        receivers = [i for i in range(num_workers) if i != src_rank]

        def _tensor_device(idx: int) -> str:
            return (
                CollectiveGroup.CPU if cpu_tensor_mask[idx] else CollectiveGroup.ACCEL
            )

        first_receiver = receivers[0]
        other_receivers = receivers[1:]

        hop_grp: Optional[CollectiveGroup] = None
        if self._rank == src_rank or self._rank == first_receiver:
            hop_grp = self._get_or_create_ipc_sub_group(src_rank, first_receiver)
            hop_grp._init_process_group(options=options)

        # Per-pair sub-groups between first_receiver and each other receiver.
        # Each pair has its own communicator/stream, so async sends from the
        # first receiver to multiple other receivers really do run in
        # parallel, and they also run in parallel with the next hop recv.
        fanout_grps: dict[int, "CollectiveGroup"] = {}
        if self._rank == first_receiver:
            for r in other_receivers:
                pg = self._get_or_create_ipc_sub_group(first_receiver, r)
                pg._init_process_group(options=options)
                fanout_grps[r] = pg
        elif self._rank in other_receivers:
            pg = self._get_or_create_ipc_sub_group(first_receiver, self._rank)
            pg._init_process_group(options=options)
            fanout_grps[self._rank] = pg

        if self._rank == src_rank:
            for idx, tensor in enumerate(broadcast_tensors):
                hop_grp._send(
                    tensor,
                    device=_tensor_device(idx),
                    comm_id=next(hop_grp._send_comm_id_iter),
                )
        elif self._rank == first_receiver:
            prev_send_works: list = []
            for idx, tensor in enumerate(broadcast_tensors):
                # Sync recv tensor `idx` from src. While the host is blocked
                # here, the previous iteration's async fan-out sends are
                # still in flight on the other receivers' pair sub-groups,
                # so the two operations overlap.
                hop_grp._recv(
                    tensor,
                    device=_tensor_device(idx),
                    comm_id=next(hop_grp._recv_comm_id_iter),
                )
                # Drain the previous iter's sends before queuing more on the
                # same per-pair channels.
                for w in prev_send_works:
                    w.wait()
                prev_send_works = []
                # Async-send tensor `idx` to every other receiver in
                # parallel. Distinct pair sub-groups => distinct streams,
                # so these N-1 sends really fire concurrently.
                for r in other_receivers:
                    pg = fanout_grps[r]
                    work = pg._send(
                        tensor,
                        device=_tensor_device(idx),
                        comm_id=next(pg._send_comm_id_iter),
                        async_op=True,
                    )
                    if work is not None:
                        prev_send_works.append(work)
            for w in prev_send_works:
                w.wait()
        else:
            pg = fanout_grps[self._rank]
            for idx, tensor in enumerate(broadcast_tensors):
                pg._recv(
                    tensor,
                    device=_tensor_device(idx),
                    comm_id=next(pg._recv_comm_id_iter),
                )

        return broadcast_tensors

    def _object_to_tensor(self, obj: Any, device: str):
        """Convert an object to tensor.

        This is modified version of dist.distributed_c10d._object_to_tensor that removes the group argument.
        """
        f = io.BytesIO()
        try:
            Pickler(f).dump(obj)
        except Exception:
            CloudPickler(f).dump(obj)
        byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage).to(device)
        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size

    def _tensor_to_object(self, tensor: torch.Tensor, tensor_size: torch.Tensor):
        """Convert a tensor back to the object.

        This is modified version of dist.distributed_c10d._tensor_to_object that removes the group argument.
        """
        tensor = tensor.cpu()
        buf = tensor.numpy().tobytes()[:tensor_size]
        return Unpickler(io.BytesIO(buf)).load()

    def _send_single_tensor_via_ipc(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling same device send/recv in send_tensor."""
        handle = reduce_tensor(tensor)
        self._logger.debug(
            f"Sending tensor via IPC from worker {self._cur_worker_address.get_name()}"
        )
        handle_tensor, handle_tensor_size = self._object_to_tensor(handle, "cpu")
        self._send(
            handle_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        self._send(
            handle_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        self._send(
            torch.tensor(0, dtype=torch.long, device="cpu"),
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        Worker.torch_platform.ipc_collect()

    def _recv_single_tensor_via_ipc(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling same device send/recv in recv_tensor."""
        self._logger.debug(
            f"Receiving tensor via IPC in worker {self._cur_worker_address.get_name()}"
        )
        handle_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        recv_work = self._recv(
            handle_tensor_size,
            CollectiveGroup.CPU,
            comm_id,
            async_op=async_op,
        )

        def recv_and_copy(handle_tensor_size: torch.Tensor):
            handle_tensor = torch.empty(
                handle_tensor_size.item(), dtype=torch.uint8, device="cpu"
            )
            self._recv(handle_tensor, CollectiveGroup.CPU, comm_id)
            handle = self._tensor_to_object(handle_tensor, handle_tensor_size)
            remote_tensor_func, remote_tensor_args = handle
            remote_tensor = remote_tensor_func(*remote_tensor_args)
            tensor.copy_(remote_tensor)
            Worker.torch_platform.current_stream().synchronize()
            del remote_tensor
            zero_tensor = torch.tensor(0, dtype=torch.long, device="cpu")
            self._recv(zero_tensor, CollectiveGroup.CPU, comm_id)
            Worker.torch_platform.ipc_collect()
            return None

        if async_op:
            return recv_work.then(recv_and_copy, handle_tensor_size)
        else:
            recv_and_copy(handle_tensor_size)

    def _send_single_tensor_to_uncertain_peer(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling possible same devices send/recv in send_tensor."""
        # Exchange tensor device info
        tensor_device = self._get_accelerator_device_identity(tensor.device)
        device_tensor, device_tensor_size = self._object_to_tensor(tensor_device, "cpu")
        send_work = self._send(
            device_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

        def check_and_send():
            self._send(device_tensor, CollectiveGroup.CPU, comm_id)
            peer_device_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
            self._recv(
                peer_device_tensor_size,
                CollectiveGroup.CPU,
                comm_id,
            )
            peer_device_tensor = torch.empty(
                peer_device_tensor_size.item(), dtype=torch.uint8, device="cpu"
            )
            self._recv(
                peer_device_tensor,
                CollectiveGroup.CPU,
                comm_id,
            )
            peer_device = self._tensor_to_object(
                peer_device_tensor, peer_device_tensor_size
            )
            if peer_device == tensor_device:
                # The peer worker is on the same device, so we need to use CUDA IPC to send the tensors
                handle = reduce_tensor(tensor)
                self._send_object(
                    handle,
                    comm_id=comm_id,
                    async_op=False,
                )
                self._send(
                    torch.tensor(0, dtype=torch.long, device="cpu"),
                    CollectiveGroup.CPU,
                    comm_id=comm_id,
                )
                Worker.torch_platform.ipc_collect()
            else:
                self._send(tensor, CollectiveGroup.ACCEL, comm_id=comm_id)

        if async_op:
            return send_work.then(check_and_send)
        else:
            check_and_send()

    def _recv_single_tensor_to_uncertain_peer(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling possible same devices send/recv in recv_tensor."""
        # Exchange tensor device info
        tensor_device = self._get_accelerator_device_identity(tensor.device)
        device_tensor, device_tensor_size = self._object_to_tensor(tensor_device, "cpu")

        peer_device_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        recv_work = self._recv(
            peer_device_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

        def check_and_recv(peer_device_tensor_size: torch.Tensor):
            peer_device_tensor = torch.empty(
                peer_device_tensor_size.item(), dtype=torch.uint8, device="cpu"
            )
            self._recv(peer_device_tensor, device=CollectiveGroup.CPU, comm_id=comm_id)
            self._send(device_tensor_size, CollectiveGroup.CPU, comm_id=comm_id)
            self._send(device_tensor, CollectiveGroup.CPU, comm_id=comm_id)
            peer_device = self._tensor_to_object(
                peer_device_tensor, peer_device_tensor_size
            )
            if peer_device == tensor_device:
                # The peer worker is on the same device, so we need to use CUDA IPC to send the tensors
                handle = self._recv_object(comm_id)
                remote_tensor_func, remote_tensor_args = handle
                remote_tensor = remote_tensor_func(*remote_tensor_args)
                tensor.copy_(remote_tensor)
                Worker.torch_platform.current_stream().synchronize()
                del remote_tensor
                zero_tensor = torch.tensor(0, dtype=torch.long, device="cpu")
                self._recv(zero_tensor, CollectiveGroup.CPU, comm_id)
                Worker.torch_platform.ipc_collect()
                return None
            else:
                return self._recv(tensor, CollectiveGroup.ACCEL, comm_id)

        if async_op:
            return recv_work.then(check_and_recv, peer_device_tensor_size)
        else:
            check_and_recv(peer_device_tensor_size)

    def _send_tensor_list_via_ipc(
        self,
        tensors: list[torch.Tensor],
        comm_id: int,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Handle same device send/recv in _send_tensor_list."""
        tensor_handles = [reduce_tensor(tensor) for tensor in tensors]
        self._logger.debug(
            f"Sending {len(tensor_handles)} tensors via IPC from worker {self._cur_worker_address.get_name()}"
        )
        handles_tensor, handles_tensor_size = self._object_to_tensor(
            tensor_handles, "cpu"
        )

        self._send(
            handles_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        work = self._send(
            handles_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

        self._send(
            torch.tensor(0, dtype=torch.long, device="cpu"),
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        Worker.torch_platform.ipc_collect()

        if async_op:
            return work

    def _recv_tensor_list_via_ipc(self, comm_id: int) -> list[torch.Tensor]:
        self._logger.debug(
            f"Receiving tensors via IPC in worker {self._cur_worker_address.get_name()}"
        )
        handles_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(
            handles_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        handles_tensor = torch.empty(
            handles_tensor_size.item(), dtype=torch.uint8, device="cpu"
        )
        self._recv(
            handles_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        tensor_handles = self._tensor_to_object(handles_tensor, handles_tensor_size)

        remote_tensors = [
            rebuild_func(*rebuild_args)
            for (rebuild_func, rebuild_args) in tensor_handles
        ]
        tensors = [
            tensor.clone().detach().to(Worker.torch_platform.current_device())
            for tensor in remote_tensors
        ]

        Worker.torch_platform.current_stream().synchronize()
        remote_tensors.clear()
        zero_tensor = torch.tensor(0, dtype=torch.long, device="cpu")
        self._recv(zero_tensor, CollectiveGroup.CPU, comm_id)
        Worker.torch_platform.ipc_collect()

        return tensors

    def _send_tensor_list_to_uncertain_peer(
        self,
        tensors: list[torch.Tensor],
        comm_id: int,
        async_op: bool = False,
    ):
        """For handling same device send/recv in _send_tensor_list."""
        # Exchange tensor device info
        devices = [
            self._get_accelerator_device_identity(tensor.device) for tensor in tensors
        ]

        devices_tensor, devices_tensor_size = self._object_to_tensor(devices, "cpu")
        send_work = self._send(
            devices_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

        def send_tensors_with_peer_device_info():
            self._send(devices_tensor, device=CollectiveGroup.CPU, comm_id=comm_id)
            peer_device_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
            self._recv(
                peer_device_tensor_size,
                device=CollectiveGroup.CPU,
                comm_id=comm_id,
            )
            peer_device_tensor = torch.empty(
                peer_device_tensor_size.item(), dtype=torch.uint8, device="cpu"
            )
            self._recv(peer_device_tensor, device=CollectiveGroup.CPU, comm_id=comm_id)
            peer_device = self._tensor_to_object(
                peer_device_tensor, peer_device_tensor_size
            )

            tensors_via_ipc = []
            tensors_via_nccl = []
            for tensor, tensor_device in zip(tensors, devices):
                if tensor_device == peer_device:
                    tensors_via_ipc.append(tensor)
                else:
                    tensors_via_nccl.append(tensor)

            if len(tensors_via_ipc) > 0:
                self._send_tensor_list_via_ipc(tensors_via_ipc, comm_id)
            if len(tensors_via_nccl) > 0:
                self._logger.debug(f"Sending {len(tensors_via_nccl)} tensors via NCCL")
                for tensor in tensors_via_nccl:
                    self._send(
                        tensor=tensor,
                        device=CollectiveGroup.ACCEL,
                        comm_id=comm_id,
                    )

        if async_op:
            return send_work.then(send_tensors_with_peer_device_info)
        else:
            send_tensors_with_peer_device_info()

    def _recv_tensor_list_to_uncertain_peer(
        self, tensor_shapes: torch.Size, comm_id: int
    ):
        """For handling same device send/recv in _recv_tensor_list."""
        peer_tensor_devices_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(
            peer_tensor_devices_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        peer_tensor_devices_tensor = torch.empty(
            peer_tensor_devices_tensor_size.item(), dtype=torch.uint8, device="cpu"
        )
        self._recv(
            peer_tensor_devices_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
        )
        peer_tensor_devices = self._tensor_to_object(
            peer_tensor_devices_tensor, peer_tensor_devices_tensor_size
        )

        current_device = self._get_accelerator_device_identity(
            Worker.torch_platform.current_device()
        )
        device_tensor, device_tensor_size = self._object_to_tensor(
            current_device, "cpu"
        )
        self._send(device_tensor_size, device=CollectiveGroup.CPU, comm_id=comm_id)
        self._send(device_tensor, device=CollectiveGroup.CPU, comm_id=comm_id)

        ipc_tensor_indices = [
            i
            for i, device in enumerate(peer_tensor_devices)
            if device == current_device
        ]
        nccl_tensor_indices = [
            i
            for i, device in enumerate(peer_tensor_devices)
            if device != current_device
        ]
        self._logger.debug(
            f"Receiving tensors with {len(ipc_tensor_indices)} tensors via IPC and {len(nccl_tensor_indices)} tensors via NCCL"
        )

        tensors = [None for _ in range(len(tensor_shapes))]
        if len(ipc_tensor_indices) > 0:
            ipc_tensors = self._recv_tensor_list_via_ipc(comm_id)
            for i, tensor in zip(ipc_tensor_indices, ipc_tensors):
                tensors[i] = tensor
        if len(nccl_tensor_indices) > 0:
            for i in nccl_tensor_indices:
                shape, dtype = tensor_shapes[i]
                tensors[i] = torch.empty(
                    shape, dtype=dtype, device=Worker.torch_platform.current_device()
                )
                self._recv(
                    tensor=tensors[i],
                    device=CollectiveGroup.ACCEL,
                    comm_id=comm_id,
                )
        return tensors

    def _send_tensor_list(
        self,
        tensors: list[torch.Tensor],
        comm_id: int,
        tensor_data: TensorData,
        async_op: bool = False,
        piggyback_payload: Optional[Any] = None,
        work: Optional[AsyncFuncWork] = None,
    ) -> Optional[AsyncWork]:
        """Send a list of tensors to the specified destination address in the collective group.

        Args:
            tensors (List[torch.Tensor]): The list of tensors to send.
            comm_id (int): The ID for the send operation.
            tensor_data (TensorData): Pre-computed metadata from `_get_object_device_type`.
            async_op (bool): Whether to perform the operation asynchronously.
            piggyback_payload (Optional[Any]): The payload to piggyback on the send operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work via :meth:`_track_payload_time`.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        cpu_tensor_mask = tensor_data.cpu_tensor_mask
        cpu_tensors = tensor_data.cpu_tensors
        accel_tensors = tensor_data.accel_tensors

        dst_rank_in_group = self._peer_rank
        last_work: Optional[dist.Work] = None

        # First send tensor size list
        tensor_shape_dtype = [(tensor.shape, tensor.dtype) for tensor in tensors]
        assert len(cpu_tensor_mask) == len(tensors), (
            f"Length mismatch for precomputed tensor device flags: expected {len(tensors)}, got {len(cpu_tensor_mask)}"
        )
        metadata = {
            "meta": tensor_shape_dtype,
            "pb": piggyback_payload,
            "cpu_tensor_mask": cpu_tensor_mask,
        }
        self._logger.debug(
            f"Sending tensor metadata {metadata} to Rank {dst_rank_in_group} in group {self._group_info.group_name}"
        )
        metadata_tensor, metadata_tensor_size = self._object_to_tensor(metadata, "cpu")

        self._send(
            metadata_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=False,
        )
        self._send(
            metadata_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=False,
        )

        self._logger.debug(
            f"Sending list of {len(tensors)} tensors to Rank {dst_rank_in_group} in group {self._group_info.group_name}"
        )

        with self._track_payload_time(work=work):
            for tensor in cpu_tensors:
                last_work = self._send(
                    tensor,
                    device=CollectiveGroup.CPU,
                    comm_id=comm_id,
                    async_op=async_op,
                )
            if accel_tensors:
                # Handle CUDA tensor sending with IPC if the peer worker is on the same device
                check_cuda_device_result = self._check_same_device_with_peer()
                if check_cuda_device_result == 0:
                    last_work = self._send_tensor_list_to_uncertain_peer(
                        accel_tensors, comm_id, async_op
                    )
                elif check_cuda_device_result == 1:
                    last_work = self._send_tensor_list_via_ipc(
                        accel_tensors, comm_id, async_op
                    )
                else:
                    for tensor in accel_tensors:
                        last_work = self._send(
                            tensor,
                            device=CollectiveGroup.ACCEL,
                            comm_id=comm_id,
                            async_op=async_op,
                        )

        if async_op:
            return last_work

    def _recv_tensor_list(
        self,
        comm_id: int,
        work: Optional[AsyncFuncWork] = None,
    ) -> tuple[list[torch.Tensor], Any]:
        """Receive a list of tensors from the specified source address in the collective group.

        Args:
            comm_id (int): The ID for the recv operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            tuple[List[torch.Tensor], Any]: A tuple of the received list of tensors and the piggyback payload.

        """
        # Recv metadata of the list
        self._logger.debug(
            f"Receiving tensor list metadata from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        metadata_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(metadata_size, CollectiveGroup.CPU, comm_id)
        metadata_tensor = torch.empty(
            metadata_size.item(), dtype=torch.uint8, device="cpu"
        )
        self._recv(metadata_tensor, CollectiveGroup.CPU, comm_id)
        metadata = self._tensor_to_object(metadata_tensor, metadata_size)
        self._logger.debug(
            f"Received metadata: {metadata} from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )

        # Construct the tensors based on the metadata
        tensor_shapes = metadata["meta"]
        pb_data = metadata["pb"]
        cpu_tensor_mask = metadata["cpu_tensor_mask"]
        has_accel_tensor = any(not m for m in cpu_tensor_mask)

        tensors = [
            torch.empty(
                shape,
                dtype=dtype,
                device=(
                    "cpu"
                    if is_cpu
                    else Worker.torch_platform.current_device()
                    if has_accel_tensor and Worker.torch_platform is not None
                    else "cpu"
                ),
            )
            for (shape, dtype), is_cpu in zip(tensor_shapes, cpu_tensor_mask)
        ]

        # Recv the tensors
        self._logger.debug(
            f"Receiving {len(tensors)} tensors from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        cpu_tensors: list[torch.Tensor] = []
        accel_entries: list[
            tuple[int, torch.Tensor, tuple[torch.Size, torch.dtype]]
        ] = []
        for idx, (tensor, is_cpu, shape_dtype) in enumerate(
            zip(tensors, cpu_tensor_mask, tensor_shapes)
        ):
            if is_cpu:
                cpu_tensors.append(tensor)
            else:
                accel_entries.append((idx, tensor, shape_dtype))

        with self._track_payload_time(work=work):
            for tensor in cpu_tensors:
                self._recv(tensor, CollectiveGroup.CPU, comm_id)
            if has_accel_tensor:
                check_cuda_device_result = self._check_same_device_with_peer()
                if check_cuda_device_result == 0:
                    accel_shapes = [shape_dtype for _, _, shape_dtype in accel_entries]
                    received_accel_tensors = self._recv_tensor_list_to_uncertain_peer(
                        accel_shapes, comm_id
                    )
                    for (idx, _, _), tensor in zip(
                        accel_entries, received_accel_tensors
                    ):
                        tensors[idx] = tensor
                elif check_cuda_device_result == 1:
                    received_accel_tensors = self._recv_tensor_list_via_ipc(comm_id)
                    for (idx, _, _), tensor in zip(
                        accel_entries, received_accel_tensors
                    ):
                        tensors[idx] = tensor
                else:
                    for _, tensor, _ in accel_entries:
                        self._recv(tensor, CollectiveGroup.ACCEL, comm_id)
        return tensors, pb_data

    def _send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor],
        comm_id: int,
        tensor_data: TensorData,
        async_op: bool = False,
        piggyback_payload: Optional[Any] = None,
        work: Optional[AsyncFuncWork] = None,
    ) -> Optional[AsyncWork]:
        """Send a dictionary of tensors to the specified destination address in the collective group.

        Args:
            tensor_dict (Dict[str, torch.Tensor]): The dictionary of tensors to send.
            comm_id (int): The ID for the send operation.
            tensor_data (TensorData): Pre-computed metadata from `_get_object_device_type`.
            async_op (bool): Whether to perform the operation asynchronously.
            piggyback_payload (Optional[Any]): The payload to piggyback on the send operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        # Send keys
        keys = list(tensor_dict.keys())
        values = list(tensor_dict.values())
        keys = (keys, piggyback_payload)
        keys_tensor, key_tensor_size = self._object_to_tensor(keys, "cpu")
        self._logger.debug(
            f"Sending {len(keys)} keys to Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        self._send(
            key_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        self._send(
            keys_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

        # Send values
        value_work = self._send_tensor_list(
            values,
            comm_id,
            tensor_data=tensor_data,
            async_op=async_op,
            work=work,
        )

        if async_op:
            return value_work

    def _recv_tensor_dict(
        self,
        comm_id: int,
        work: Optional[AsyncFuncWork] = None,
    ) -> tuple[dict[str, torch.Tensor], Any]:
        """Receive a dictionary of tensors from the specified source address in the collective group.

        Args:
            comm_id (int): The ID for the recv operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            tuple[Dict[str, torch.Tensor], Any]: A tuple of the received dictionary of tensors and the piggyback payload.

        """
        src_rank_in_group = self._peer_rank

        # Recv keys
        key_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(key_tensor_size, CollectiveGroup.CPU, comm_id)
        keys_tensor = torch.empty(
            key_tensor_size.item(), dtype=torch.uint8, device="cpu"
        )
        self._recv(keys_tensor, CollectiveGroup.CPU, comm_id)
        keys, pb_data = self._tensor_to_object(keys_tensor, key_tensor_size)
        self._logger.debug(
            f"Received {len(keys)} keys from Rank {src_rank_in_group} in group {self._group_info.group_name}"
        )

        # Recv values
        values, _ = self._recv_tensor_list(comm_id, work=work)
        assert len(keys) == len(values), (
            f"Received {len(values)} values but expected {len(keys)} keys from Rank {src_rank_in_group} in group {self._group_info.group_name}"
        )
        return dict(zip(keys, values)), pb_data

    def _send_tensor_dataclass(
        self,
        tensor_dataclass: Any,
        comm_id: int,
        tensor_data: TensorData,
        async_op: bool = False,
        piggyback_payload: Optional[Any] = None,
        work: Optional[AsyncFuncWork] = None,
    ):
        """Send a dataclass with tensor fields (tensor, list of tensors, or dict of tensors) to the destination.

        Args:
            tensor_dataclass (Any): The dataclass with tensor fields to send.
            comm_id (int): The ID for the send operation.
            tensor_data (TensorData): Pre-computed metadata from `_get_object_device_type` (must have tensor_fields, metadata, tensors_list set).
            async_op (bool): Whether to perform the operation asynchronously.
            piggyback_payload (Optional[Any]): Payload to piggyback with the skeleton.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork; otherwise None.
        """
        assert tensor_data.tensor_fields is not None
        assert tensor_data.metadata is not None
        assert tensor_data.tensors_list is not None
        metadata = tensor_data.metadata
        flat_tensors = tensor_data.tensors_list
        tensor_fields = tensor_data.tensor_fields

        # Send flat tensor list with metadata as piggyback, then skeleton + piggyback.
        self._send_tensor_list(
            flat_tensors,
            comm_id,
            tensor_data=tensor_data,
            async_op=async_op,
            piggyback_payload=metadata,
            work=work,
        )
        tensor_field_names = set(tensor_fields.keys())
        overwrite_kwargs = dict.fromkeys(tensor_field_names, None)
        skeleton = replace(tensor_dataclass, **overwrite_kwargs)
        return self._send_object(
            skeleton,
            comm_id=comm_id,
            async_op=async_op,
            piggyback_payload=piggyback_payload,
            work=work,
        )

    def _recv_tensor_dataclass(
        self,
        comm_id: int,
        work: Optional[AsyncFuncWork] = None,
    ) -> tuple[Any, Any]:
        r"""Receive a dataclass with tensor fields (tensor, list, or dict of tensors).

        Mirrors `_send_tensor_dataclass`:
        1) Receive flat tensor list (metadata comes as piggyback_payload).
        2) Receive skeleton dataclass and reconstruct by refilling tensor fields.
        """
        flat_tensors, metadata = self._recv_tensor_list(comm_id, work=work)
        tensor_dict = unflatten_dataclass_tensor_fields(metadata, flat_tensors)
        skeleton, pb_data = self._recv_object(comm_id, work=work)
        dataclass_obj = replace(skeleton, **tensor_dict)
        return dataclass_obj, pb_data

    def _broadcast_tensor_dataclass(
        self,
        tensor_dataclass: Optional[Any],
        comm_id: int,
        src_rank: int,
        options: Optional[CollectiveGroupOptions] = None,
        work: Optional[AsyncFuncWork] = None,
    ) -> Any:
        """Broadcast a dataclass with tensor fields (tensor, list, or dict of tensors) from src_rank to all ranks.

        On the source rank:
            - `tensor_dataclass` must be the actual dataclass instance.
        On other ranks:
            - `tensor_dataclass` must be None.
        """
        if self._rank == src_rank:
            tensor_dict, flat_tensors, metadata = extract_dataclass_tensor_fields(
                tensor_dataclass
            )
            tensor_field_names = set(tensor_dict.keys())
            overwrite_kwargs = dict.fromkeys(tensor_field_names, None)
            skeleton = replace(tensor_dataclass, **overwrite_kwargs)
        else:
            metadata = None
            flat_tensors = None
            skeleton = None

        recv_metadata = self._broadcast_object(
            metadata, comm_id=comm_id, src_rank=src_rank
        )
        recv_flat_tensors = self._broadcast_tensor_list(
            flat_tensors,
            comm_id=comm_id,
            src_rank=src_rank,
            options=options,
            work=work,
        )
        recv_tensor_dict = unflatten_dataclass_tensor_fields(
            recv_metadata, recv_flat_tensors
        )
        recv_skeleton = self._broadcast_object(
            skeleton, comm_id=comm_id, src_rank=src_rank, work=work
        )
        return replace(recv_skeleton, **recv_tensor_dict)

    def _send_object(
        self,
        object: Any,
        comm_id: int = 0,
        async_op: bool = False,
        piggyback_payload: Optional[Any] = None,
        work: Optional[AsyncFuncWork] = None,
    ):
        """Send an object to the specified destination address in the collective group. The object can be any Python object that can be serialized into a tensor. Objects are always sent via CPU tensor.

        Args:
            object (Any): The object to send.
            comm_id (int): The ID for the send operation.
            async_op (bool): Whether to perform the operation asynchronously.
            piggyback_payload (Optional[Any]): The payload to piggyback on the send operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        self._logger.debug(
            f"Sending object to Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        object = (object, piggyback_payload)
        object_tensor, object_tensor_size = self._object_to_tensor(object, "cpu")
        self._send(
            object_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        with self._track_payload_time(work=work):
            object_work = self._send(
                object_tensor,
                device=CollectiveGroup.CPU,
                comm_id=comm_id,
                async_op=async_op,
            )
        if async_op:
            return object_work

    def _recv_object(
        self,
        comm_id: int,
        work: Optional[AsyncFuncWork] = None,
    ) -> tuple[Any, Any]:
        """Receive an object from the specified source address in the collective group.

        Args:
            comm_id (int): The ID for the recv operation.
            work (Optional[AsyncFuncWork]): If provided, payload-transfer time is
                attributed to this work.

        Returns:
            tuple[Any, Any]: A tuple of the received object and the piggyback payload.

        """
        object_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(object_size, CollectiveGroup.CPU, comm_id)
        object_tensor = torch.empty(object_size.item(), dtype=torch.uint8, device="cpu")
        with self._track_payload_time(work=work):
            self._recv(object_tensor, CollectiveGroup.CPU, comm_id)
        return self._tensor_to_object(object_tensor, object_size)
