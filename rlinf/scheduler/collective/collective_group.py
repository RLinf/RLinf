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
from contextlib import nullcontext
from pickle import Pickler, Unpickler
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

from ..manager import CollectiveGroupInfo, CollectiveManager, WorkerInfo
from ..worker import Worker, WorkerAddress
from .async_work import AsyncFuncWork, AsyncWork

if TYPE_CHECKING:
    from .collective import Collective


class CollectiveWorkQueue:
    """A queue for managing asynchronous collective operations."""

    SEND = 0
    RECV = 1

    def __init__(self, comm_type: int, logger: logging.Logger):
        """Initialize the CollectiveWorkQueue.

        Args:
            comm_type (int): The type of the communication (SEND or RECV).
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
        event: Optional["torch.cuda.Event"] = None,
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
                self._logger.debug(
                    f"Async {'send' if self._type == CollectiveWorkQueue.SEND else 'recv'} ID {comm_id} done"
                )
                self._logger.debug(f"Done comm work {work}")


class CollectiveGroup:
    """Collective group for constructing and performing collective operations."""

    ACCEL: str = "cuda"
    CPU: str = "cpu"
    TENSOR: int = 0
    TENSOR_LIST: int = 1
    TENSOR_DICT: int = 2
    OBJECT: int = 3
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

        if self._group_info is not None:
            self._init_group()

        self._send_comm_id_iter = itertools.count()
        self._recv_comm_id_iter = itertools.count()

        self._send_work_queues = [
            CollectiveWorkQueue(CollectiveWorkQueue.SEND, self._logger)
            for _ in range(CollectiveGroup.POOL_SIZE)
        ]
        self._recv_work_queues = [
            CollectiveWorkQueue(CollectiveWorkQueue.RECV, self._logger)
            for _ in range(CollectiveGroup.POOL_SIZE)
        ]

    def send(
        self,
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Implement the Worker's send method.

        The real communication implementation is in the _atomic_send below.

        This function calls _atomic_send in a way so that it can be chained with previous send operations in the same channel.
        Otherwise, async send operations in the same channel may become out-of-order and mismatch with recv.
        """
        # Only iter the channel here and pass the channel id along the way.
        # Because the _atomic_send and all the send in the way may be called asynchronously while the channel_id in the class may be different.
        send_comm_id = next(self._send_comm_id_iter)
        device_type, object_type = self._get_object_device_type(object)

        # Create AsyncFuncWork for the send operation
        send_work = AsyncFuncWork(
            self._atomic_send,
            object=object,
            comm_id=send_comm_id,
            device_type=device_type,
            object_type=object_type,
        )

        # Capture CUDA event of the main stream if the device type is CUDA
        if device_type == CollectiveGroup.ACCEL:
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
        object: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any,
        comm_id: int,
        device_type: str,
        object_type: str,
    ) -> Optional[AsyncWork]:
        """Send an object to a specific address in the collective group in an out-of-place manner.

        It runs in an atomic way, i.e., communications of two calls of _atomic_send are guaranteed to be in the same ordered as the send API is called.
        """
        self._init_p2p_process_group()
        # First send object type to the destination worker
        object_type_tensor = torch.tensor(object_type, dtype=torch.int, device="cpu")
        self._send(object_type_tensor, CollectiveGroup.CPU, comm_id)
        self._logger.debug(
            f"Sending object type {object_type} from {self._cur_worker_address.get_name()} in group {self._group_info.group_name}"
        )

        if object_type == CollectiveGroup.TENSOR:
            # Out-of-place tensor send/recv is done via tensor list send/recv with a list of one tensor
            return self._send_tensor_list([object], device_type, comm_id)
        elif object_type == CollectiveGroup.TENSOR_LIST:
            return self._send_tensor_list(object, device_type, comm_id)
        elif object_type == CollectiveGroup.TENSOR_DICT:
            return self._send_tensor_dict(object, device_type, comm_id)
        elif object_type == CollectiveGroup.OBJECT:
            return self._send_object(object, device_type, comm_id)
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

    def recv(
        self, async_op: bool = False
    ) -> AsyncWork | torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any:
        """Implement Worker's recv method.

        Similar as the send method above, it ensures the correct ordering of multiple communications of two recv calls.
        """
        recv_comm_id = next(self._recv_comm_id_iter)

        if self._worker.has_accelerator and Worker.torch_platform.is_initialized():
            current_device = Worker.torch_platform.current_device()
        else:
            current_device = None

        recv_work = AsyncFuncWork(
            self._atomic_recv, comm_id=recv_comm_id, current_device=current_device
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
        self, comm_id: int, current_device: Optional[int]
    ) -> AsyncWork | torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | Any:
        """Atomic recv implementation."""
        if current_device is not None:
            Worker.torch_platform.set_device(current_device)

        self._init_p2p_process_group()

        # First recv object type
        object_type_tensor = torch.empty(1, dtype=torch.int, device="cpu")
        self._recv(object_type_tensor, CollectiveGroup.CPU, comm_id)

        object_type = object_type_tensor.item()
        self._logger.debug(
            f"Receiving object type {object_type} from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        if object_type == CollectiveGroup.TENSOR:
            tensor = self._recv_tensor_list(comm_id)
            assert len(tensor) == 1, (
                f"Expected to receive one tensor but got {len(tensor)} tensors from Rank {self._peer_rank} in group {self._group_info.group_name}"
            )
            return tensor[0]
        elif object_type == CollectiveGroup.TENSOR_LIST:
            return self._recv_tensor_list(comm_id)
        elif object_type == CollectiveGroup.TENSOR_DICT:
            return self._recv_tensor_dict(comm_id)
        elif object_type == CollectiveGroup.OBJECT:
            return self._recv_object(comm_id)

    def send_tensor(
        self, tensor: torch.Tensor, async_op: bool = False
    ) -> Optional[AsyncWork]:
        """Implement the Worker's send_tensor method.

        It's also a wrapper of _atomic_send_tensor to ensure the correct ordering of multiple send_tensor calls in the same channel.
        """
        send_comm_id = next(self._send_comm_id_iter)
        device_type, object_type = self._get_object_device_type(tensor)

        send_work = AsyncFuncWork(
            self._atomic_send_tensor,
            tensor=tensor,
            comm_id=send_comm_id,
            device_type=device_type,
            object_type=object_type,
        )

        if device_type == CollectiveGroup.ACCEL:
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
        self, tensor: torch.Tensor, comm_id: int, device_type: str, object_type: str
    ) -> None:
        """Atomic send_tensor implementation."""
        assert object_type == CollectiveGroup.TENSOR, (
            "The object must be a torch.Tensor when using send_tensor"
        )
        if device_type == CollectiveGroup.ACCEL and not tensor.is_contiguous():
            raise ValueError(
                "All CUDA tensors must be contiguous when using P2P communication. Otherwise the recv side might recv wrong tensor data. Consider using .contiguous() to make the tensors contiguous."
            )

        self._init_p2p_process_group()
        self._logger.debug(
            f"Sending tensor to Rank {self._peer_rank} in group {self._group_info.group_name}"
        )

        # Handle CUDA tensor sending with IPC if the peer worker is on the same device
        if device_type == CollectiveGroup.ACCEL:
            check_cuda_device_result = self._check_same_device_with_peer()
            if check_cuda_device_result == 0:
                return self._send_single_cuda_tensor_to_uncertain_peer(tensor, comm_id)
            elif check_cuda_device_result == 1:
                return self._send_single_cuda_tensor_via_ipc(tensor, comm_id)

        return self._send(tensor, device=device_type, comm_id=comm_id)

    def recv_tensor(
        self, tensor: torch.Tensor, async_op: bool = False
    ) -> Optional[AsyncWork]:
        """Implement Worker's recv_tensor method.

        It's also a wrapper of _atomic_recv_tensor to ensure the correct ordering of multiple recv_tensor calls in the same channel.
        """
        recv_comm_id = next(self._recv_comm_id_iter)

        recv_work = AsyncFuncWork(
            self._atomic_recv_tensor, tensor=tensor, comm_id=recv_comm_id
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

    def _atomic_recv_tensor(self, tensor: torch.Tensor, comm_id: int) -> None:
        """Atomic recv_tensor implementation."""
        device_type, object_type = self._get_object_device_type(tensor)
        assert object_type == CollectiveGroup.TENSOR, (
            "The object must be a torch.Tensor"
        )

        self._init_p2p_process_group()
        self._logger.debug(
            f"Receiving tensor from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        if device_type == CollectiveGroup.ACCEL:
            check_cuda_device_result = self._check_same_device_with_peer()
            if check_cuda_device_result == 0:
                return self._recv_single_cuda_tensor_to_uncertain_peer(tensor, comm_id)
            elif check_cuda_device_result == 1:
                # The peer worker is on the same device, so we need to use CUDA IPC to receive the tensors
                return self._recv_single_cuda_tensor_via_ipc(tensor, comm_id)
        return self._recv(tensor, device=device_type, comm_id=comm_id)

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

    def _init_p2p_process_group(self) -> dist.ProcessGroup:
        """Initialize the process group for collective operations."""
        with self._lock:
            self._init_group()
            if self._mc_group.is_initialized:
                return

            from ..cluster import Cluster

            if self._rank == 0:
                master_port = Cluster.find_free_port()
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
            )

            self._logger.debug(
                f"Process group {self._group_info.group_name} initialized successfully."
            )

            if self._rank == 0:
                # Avoid using the same master port for the next group
                self._coll_manager.reset_master_port_info(self._group_info.group_name)

    def _get_object_device_type(self, object: torch.Tensor | Any) -> tuple[str, int]:
        """Check the device type of the object. We also handle List of tensors, tuple of tensors, and Dict of tensors (all values must be tensors)."""
        device_type = CollectiveGroup.CPU
        object_type = CollectiveGroup.OBJECT
        if isinstance(object, torch.Tensor):
            device_type = (
                CollectiveGroup.ACCEL
                if object.device.type == Worker.torch_device_type
                else CollectiveGroup.CPU
            )
            object_type = CollectiveGroup.TENSOR
        elif (isinstance(object, list) or isinstance(object, tuple)) and all(
            isinstance(item, torch.Tensor) for item in object
        ):
            device_type = (
                CollectiveGroup.ACCEL
                if all(item.device.type == Worker.torch_device_type for item in object)
                else CollectiveGroup.CPU
            )
            if device_type == CollectiveGroup.CPU:
                assert all(item.device.type == "cpu" for item in object), (
                    "All tensors in the list or tuple must be on the same device"
                )
            object_type = CollectiveGroup.TENSOR_LIST
        elif isinstance(object, dict) and all(
            isinstance(item, torch.Tensor) for item in object.values()
        ):
            device_type = (
                CollectiveGroup.ACCEL
                if all(
                    item.device.type == Worker.torch_device_type
                    for item in object.values()
                )
                else CollectiveGroup.CPU
            )
            if device_type == CollectiveGroup.CPU:
                assert all(item.device.type == "cpu" for item in object.values()), (
                    "All tensors in the dictionary must be on the same device"
                )
            object_type = CollectiveGroup.TENSOR_DICT

        try:
            if device_type == CollectiveGroup.ACCEL:
                if object_type == CollectiveGroup.TENSOR and not object.is_contiguous():
                    raise ValueError
                elif object_type == CollectiveGroup.TENSOR_LIST and not all(
                    item.is_contiguous() for item in object
                ):
                    raise ValueError
                elif object_type == CollectiveGroup.TENSOR_DICT and not all(
                    item.is_contiguous() for item in object.values()
                ):
                    raise ValueError
        except ValueError:
            raise ValueError(
                "All CUDA/Accelerator tensors must be contiguous when using P2P communication. Otherwise the recv side might recv wrong tensor data. Consider using .contiguous() to make the tensors contiguous."
            )

        return device_type, object_type

    def _check_same_device_with_peer(self):
        """Check if the current worker and the peer worker are on the same device.

        Returns:
            int: -1 means no common device; 0 means have common devices, but not sure if the tensor will be on the same device (the worker has multiple devices); 1 means the two workers are on the same device.

        """
        peer_devices = self._group_info.workers[self._peer_rank].available_accelerators
        my_devices = self._group_info.workers[self._rank].available_accelerators

        # Check if the peer is on the same node
        if (
            self._group_info.workers[self._peer_rank].cluster_node_rank
            != self._group_info.workers[self._rank].cluster_node_rank
        ):
            return -1

        # Check if the two device list has intersection
        if not set(peer_devices).intersection(set(my_devices)):
            return -1
        if len(peer_devices) == 1 and len(my_devices) == 1:
            return 1
        return 0

    def _object_to_tensor(self, obj: Any, device: str):
        """Convert an object to tensor.

        This is modified version of dist.distributed_c10d._object_to_tensor that removes the group argument.
        """
        f = io.BytesIO()
        Pickler(f).dump(obj)
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

    def _send_single_cuda_tensor_via_ipc(
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
        return self._send(
            handle_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )

    def _recv_single_cuda_tensor_via_ipc(
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
            return None

        if async_op:
            return recv_work.then(recv_and_copy, handle_tensor_size)
        else:
            recv_and_copy(handle_tensor_size)

    def _send_single_cuda_tensor_to_uncertain_peer(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling possible same devices send/recv in send_tensor."""
        # Exchange tensor device info
        tensor_device = str(
            Worker.torch_platform.get_device_properties(tensor.device).uuid
        )
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
                    device_type=CollectiveGroup.CPU,
                    comm_id=comm_id,
                    async_op=False,
                )
            else:
                self._send(tensor, CollectiveGroup.ACCEL, comm_id=comm_id)

        if async_op:
            return send_work.then(check_and_send)
        else:
            check_and_send()

    def _recv_single_cuda_tensor_to_uncertain_peer(
        self, tensor: torch.Tensor, comm_id: int, async_op: bool = False
    ):
        """For handling possible same devices send/recv in recv_tensor."""
        # Exchange tensor device info
        tensor_device = str(
            Worker.torch_platform.get_device_properties(tensor.device).uuid
        )
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
                return None
            else:
                return self._recv(tensor, CollectiveGroup.ACCEL, comm_id)

        if async_op:
            return recv_work.then(check_and_recv, peer_device_tensor_size)
        else:
            check_and_recv(peer_device_tensor_size)

    def _send_cuda_tensor_list_via_ipc(
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

        if async_op:
            return work

    def _recv_cuda_tensor_list_via_ipc(self, comm_id: int) -> list[torch.Tensor]:
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

        return tensors

    def _send_cuda_tensor_list_to_uncertain_peer(
        self,
        tensors: list[torch.Tensor],
        comm_id: int,
        async_op: bool = False,
    ):
        """For handling same device send/recv in _send_tensor_list."""
        # Exchange tensor device info
        devices = [
            str(Worker.torch_platform.get_device_properties(tensor.device).uuid)
            for tensor in tensors
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
                self._send_cuda_tensor_list_via_ipc(tensors_via_ipc, comm_id)
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

    def _recv_cuda_tensor_list_to_uncertain_peer(
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

        current_device = str(
            Worker.torch_platform.get_device_properties(
                Worker.torch_platform.current_device()
            ).uuid
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
            ipc_tensors = self._recv_cuda_tensor_list_via_ipc(comm_id)
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
        device_type: str,
        comm_id: int,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Send a list of tensors to the specified destination address in the collective group.

        Args:
            tensors (List[torch.Tensor]): The list of tensors to send.
            device_type (str): The device type of the tensors, either 'cuda' or 'cpu'.
            comm_id (int): The ID for the send operation.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        dst_rank_in_group = self._peer_rank
        work: dist.Work = None

        # First send tensor size list
        tensor_shape_dtype = [(tensor.shape, tensor.dtype) for tensor in tensors]
        metadata = {"type": device_type, "meta": tensor_shape_dtype}
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

        if device_type == CollectiveGroup.CPU:
            for tensor in tensors:
                work = self._send(tensor, device_type, comm_id, async_op=async_op)
        else:
            # Handle CUDA tensor sending with IPC if the peer worker is on the same device
            check_cuda_device_result = self._check_same_device_with_peer()
            if check_cuda_device_result == 0:
                return self._send_cuda_tensor_list_to_uncertain_peer(
                    tensors, comm_id, async_op
                )
            elif check_cuda_device_result == 1:
                return self._send_cuda_tensor_list_via_ipc(tensors, comm_id, async_op)
            else:
                for tensor in tensors:
                    work = self._send(
                        tensor,
                        device=CollectiveGroup.ACCEL,
                        comm_id=comm_id,
                        async_op=async_op,
                    )

        if async_op:
            return work

    def _recv_tensor_list(self, comm_id: int) -> list[torch.Tensor]:
        """Receive a list of tensors from the specified source address in the collective group.

            NOTE: Do not mix CPU and GPU tensors in the same list.

        Args:
            comm_id (int): The ID for the recv operation.

        Returns:
            List[torch.Tensor]: A list of received tensors.

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
        device_type = metadata["type"]
        tensor_shapes = metadata["meta"]

        tensors = []
        if device_type == CollectiveGroup.ACCEL:
            check_cuda_device_result = self._check_same_device_with_peer()
            # Find a suitable device for each tensor that is not the same device as the peer
            if check_cuda_device_result == 0:
                return self._recv_cuda_tensor_list_to_uncertain_peer(
                    tensor_shapes, comm_id
                )
            elif check_cuda_device_result == 1:
                return self._recv_cuda_tensor_list_via_ipc(comm_id)

        tensors = [
            torch.empty(
                shape,
                dtype=dtype,
                device=Worker.torch_platform.current_device()
                if device_type == CollectiveGroup.ACCEL
                else "cpu",
            )
            for (shape, dtype) in tensor_shapes
        ]

        # Recv the tensors
        self._logger.debug(
            f"Receiving {len(tensors)} tensors from Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        for tensor in tensors:
            assert tensor.device.type == device_type, (
                f"Received tensor on {tensor.device.type} but expected {device_type}"
            )
            self._recv(tensor, device_type, comm_id)
        return tensors

    def _send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor],
        device_type: str,
        comm_id: int,
        async_op: bool = False,
    ) -> Optional[AsyncWork]:
        """Send a dictionary of tensors to the specified destination address in the collective group.

        Args:
            tensor_dict (Dict[str, torch.Tensor]): The dictionary of tensors to send.
            device_type (str): The device type of the tensors, either 'cuda' or 'cpu'.
            comm_id (int): The ID for the send operation.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        # Send keys
        keys = list(tensor_dict.keys())
        values = list(tensor_dict.values())
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
            values, device_type, comm_id, async_op=async_op
        )

        if async_op:
            return value_work

    def _recv_tensor_dict(self, comm_id: int) -> dict[str, torch.Tensor]:
        """Receive a dictionary of tensors from the specified source address in the collective group.

        Args:
            comm_id (int): The ID for the recv operation.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of received tensors.

        """
        src_rank_in_group = self._peer_rank

        # Recv keys
        key_tensor_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(key_tensor_size, CollectiveGroup.CPU, comm_id)
        keys_tensor = torch.empty(
            key_tensor_size.item(), dtype=torch.uint8, device="cpu"
        )
        self._recv(keys_tensor, CollectiveGroup.CPU, comm_id)
        keys = self._tensor_to_object(keys_tensor, key_tensor_size)
        self._logger.debug(
            f"Received {len(keys)} keys from Rank {src_rank_in_group} in group {self._group_info.group_name}"
        )

        # Recv values
        values = self._recv_tensor_list(comm_id)
        assert len(keys) == len(values), (
            f"Received {len(values)} values but expected {len(keys)} keys from Rank {src_rank_in_group} in group {self._group_info.group_name}"
        )
        return dict(zip(keys, values))

    def _send_object(
        self,
        object: Any,
        device_type: str,
        comm_id: int = 0,
        async_op: bool = False,
    ):
        """Send an object to the specified destination address in the collective group. The object can be any Python object that can be serialized into a tensor.

        Args:
            object (Any): The object to send.
            device_type (str): The device type of the object, either 'cuda' or 'cpu'.
            comm_id (int): The ID for the send operation.
            async_op (bool): Whether to perform the operation asynchronously.

        Returns:
            Optional[AsyncWork]: If async_op is True, returns an AsyncWork object for the asynchronous operation. If async_op is False, returns None.

        """
        assert device_type == CollectiveGroup.CPU, (
            "The object must be sent through CPU tensor"
        )
        # Always use CPU tensor to send an object

        self._logger.debug(
            f"Sending object to Rank {self._peer_rank} in group {self._group_info.group_name}"
        )
        object_tensor, object_tensor_size = self._object_to_tensor(object, "cpu")
        self._send(
            object_tensor_size,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        object_work = self._send(
            object_tensor,
            device=CollectiveGroup.CPU,
            comm_id=comm_id,
            async_op=async_op,
        )
        if async_op:
            return object_work

    def _recv_object(self, comm_id: int) -> Any:
        """Receive an object from the specified source address in the collective group.

        Args:
            comm_id (int): The ID for the recv operation.

        Returns:
            Any: The received object, which can be any Python object that was serialized into a tensor.

        """
        object_size = torch.empty(1, dtype=torch.long, device="cpu")
        self._recv(object_size, CollectiveGroup.CPU, comm_id)
        object_tensor = torch.empty(object_size.item(), dtype=torch.uint8, device="cpu")
        self._recv(object_tensor, CollectiveGroup.CPU, comm_id)
        return self._tensor_to_object(object_tensor, object_size)
