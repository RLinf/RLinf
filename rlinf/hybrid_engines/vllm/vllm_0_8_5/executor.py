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
import threading
import weakref
from typing import Optional

import psutil
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.executor.multiproc_worker_utils import (
    _add_prefix,
    set_multiprocessing_worker_envs,
)
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_mp_context, get_open_port
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)
from vllm.worker.worker_base import WorkerWrapperBase

from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement

logger = init_logger(__name__)


class VLLMExecutor(MultiprocExecutor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rlinf_config: DictConfig,
        dp_rank: int,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
    ):
        self.rlinf_config = rlinf_config
        self.parent_address = parent_address
        self.placement = placement
        self.dp_rank = dp_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        """
        This function is copied and modified from
        vllm.v1.executor.multiproc_executor.MultiprocExecutor._init_executor.
        because it hardcodes the worker class which does not allow us to use
        our customized worker class, we changed it to use our customized
        worker class VLLMWorkerProc.
        """
        self._finalizer = weakref.finalize(self, self.shutdown)

        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue."
            )
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1"
        )

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                worker = VLLMWorkerProc.make_worker_process(
                    local_rank=rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    parent_address=self.parent_address,
                    placement=self.placement,
                    vllm_config=self.vllm_config,
                    rlinf_config=self.rlinf_config,
                    input_shm_handle=scheduler_output_handle,
                )
                unready_workers.append(worker)
            self.workers = VLLMWorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()
            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                self._ensure_worker_termination([w.proc for w in unready_workers])


class VLLMWorkerProc(WorkerProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        parent_address: WorkerAddress,
        rlinf_config: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        self.rank = rank  # global rank in 2D parallel (tp,pp)

        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            # vllm former args
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": rank == 0,
            # rlinf specific args
            "parent_address": parent_address,
            "rlinf_config": rlinf_config,
            "placement": placement,
        }
        wrapper.init_worker(all_kwargs=all_kwargs)

        self.worker = wrapper.worker

        pid = os.getpid()
        _add_prefix(
            sys.stdout,
            f"VllmWorkerProc[dp_rank={self.worker.get_dp_rank()},tp_rank={self.rank}]",
            pid,
        )
        _add_prefix(
            sys.stderr,
            f"VllmWorkerProc[dp_rank={self.worker.get_dp_rank()},tp_rank={self.rank}]",
            pid,
        )
        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)

        self.worker.init_device()
        self.worker.load_model()
        # after load_model, we should save it's named_buffers to implement sync weight
        self.worker.use_sharded_weights()

    @staticmethod
    def make_worker_process(
        # vllm former args
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        # rlinf specific args
        parent_address: WorkerAddress,
        rlinf_config: DictConfig,
        placement: ModelParallelComponentPlacement,
    ) -> WorkerProcHandle:
        """
        Note:
        This function is modified from vllm's raw implementation. because it in vllm hardcodes that
        it will launch `WorkerProc` which can't be selected, we can only copy and modify it here.
        """
        context = get_mp_context()
        reader, writer = context.Pipe(duplex=False)

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        process_kwargs = {
            # vllm former args
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            # rlinf specific args
            "parent_address": parent_address,
            "rlinf_config": rlinf_config,
            "placement": placement,
        }

        proc = context.Process(
            target=VLLMWorkerProc.worker_main, kwargs=process_kwargs, daemon=True
        )

        proc.start()
        writer.close()
        return UnreadyWorkerProcHandle(proc, rank, reader)

    @staticmethod
    def worker_main(*args, **kwargs):
        """
        Note:
        This function is modified from vllm's raw implementation. because it in vllm hardcodes that
        it will launch `WorkerProc` which can't be selected, we can only copy and modify it here.
        """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        try:
            reader.close()
            worker = VLLMWorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                }
            )
            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True
        finally:
            if ready_writer is not None:
                ready_writer.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
