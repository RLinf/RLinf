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

import signal
import threading
import weakref
from typing import List

import psutil
import zmq
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs,
)
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProcHandle

from rlinf.hybrid_engines.vllm.vllm_0_7_1.worker import VLLMWorkerProc
from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.vllm.io_struct import (
    OffloadModelWeightCommand,
    OffloadModelWeightResponse,
    SyncHFWeightCommand,
    SyncHFWeightResponse,
    VLLMCommand,
)

logger = init_logger(__name__)


class VLLMExecutor(MultiprocExecutor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        dp_rank: int,
        executor_ipc_input_name: str,
        executor_ipc_output_name: str,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
    ):
        self.executor_ipc_input_name = executor_ipc_input_name
        self.executor_ipc_output_name = executor_ipc_output_name
        context = zmq.Context()
        self.recv_from_engine = context.socket(zmq.PULL)
        self.recv_from_engine.connect(self.executor_ipc_input_name)

        self.send_to_engine = context.socket(zmq.PUSH)
        self.send_to_engine.connect(self.executor_ipc_output_name)

        self.parent_address = parent_address
        self.placement = placement

        self._rpc_lock = threading.RLock()

        self.dp_rank = dp_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        self.command_handlers = {
            "sync_hf_weight": self.sync_hf_weight,
            "offload_model_weight": self.offload_model_weights,
        }
        self.listen_engine_handle = threading.Thread(
            target=self._listen_engine, daemon=True
        )
        self.listen_engine_handle.start()
        super().__init__(vllm_config)
        logger.info("VLLMExecutor initialized.")

    def _listen_engine(self):
        while True:
            if self.recv_from_engine.poll():
                try:
                    command: VLLMCommand = self.recv_from_engine.recv_pyobj()
                    handler = self.command_handlers.get(command.command_type)
                    if handler:
                        logger.debug(f"Handling command: {type(command)}")
                        response = handler(command)
                    else:
                        logger.warning(f"No handler for command type: {type(command)}")
                        response = None
                    self.send_to_engine.send_pyobj(response)
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error(f"Failed to handle requests from engine: {e}")

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
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
        self.workers: List[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = VLLMWorkerProc.make_worker_process(
                local_rank=rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                parent_address=self.parent_address,
                placement=self.placement,
                vllm_config=self.vllm_config,
                input_shm_handle=scheduler_output_handle,
            )
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()

    def offload_model_weights(self, command: OffloadModelWeightCommand):
        self.collective_rpc("offload_model_weights", args=(command,))
        return OffloadModelWeightResponse(command_id=command.command_id)

    def sync_hf_weight(self, command: SyncHFWeightCommand):
        self.collective_rpc("sync_hf_weight", args=(command,))
        return SyncHFWeightResponse(command_id=command.command_id)
