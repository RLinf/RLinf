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

import pickle
import signal
import tempfile
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cloudpickle
import psutil
import zmq
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import WorkerProcHandle

from rlinf.hybrid_engines.vllm.vllm_0_7_1.worker import ZmqWorkerProc
from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.vllm.io_struct import (
    CollectiveRpcCommand,
    CollectiveRpcResponse,
    OffloadModelWeightCommand,
    OffloadModelWeightResponse,
    SyncHFWeightCommand,
    SyncHFWeightResponse,
    VLLMCommand,
    WorkerReadyResponse,
)

logger = init_logger(__name__)


class VLLMExecutor(Executor):
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

        self.dp_rank = dp_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        # used by LLMengine to communicate straightly with executor
        self.command_handlers = {
            "sync_hf_weight": self.sync_hf_weight,
            "offload_model_weight": self.offload_model_weights,
        }

        self._listen_engine_handle = threading.Thread(
            target=self._listen_engine, daemon=True
        )
        self._listen_engine_handle.start()

        super().__init__(vllm_config)
        logger.info("VLLMExecutor initialized.", flush=True)

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

    def _init_executor(self):
        self._finalizer = weakref.finalize(self, self.shutdown)

        def sigusr1_handler(signum, frame):
            logger.fatal(f"Vllm Executor got signal: {signum} from worker processes")
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size

        context = zmq.Context()
        self.worker_input_ipc_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )
        self.worker_output_ipc_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )

        self.send_to_workers = context.socket(zmq.PUB)
        self.send_to_workers.bind(self.worker_input_ipc_name)

        self.recv_from_workers = context.socket(zmq.PULL)
        self.recv_from_workers.bind(self.worker_output_ipc_name)

        self.worker_response_poller = zmq.Poller()
        self.worker_response_poller.register(self.recv_from_workers, zmq.POLLIN)

        self.workers: List[WorkerProcHandle] = []

        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )
        print(f"placement rollout gpus: {self.placement._rollout_gpus}")
        for rank in range(self.tp_size):
            print(
                f"vllm_executor: local rank is : {self.placement._rollout_gpus[self.dp_rank * self.tp_size + rank]}"
            )
            worker = ZmqWorkerProc.make_worker_process(
                local_rank=self.placement._rollout_gpus[
                    self.dp_rank * self.tp_size + rank
                ],
                rank=rank,
                worker_input_ipc_name=self.worker_input_ipc_name,
                worker_output_ipc_name=self.worker_output_ipc_name,
                distributed_init_method=distributed_init_method,
                parent_address=self.parent_address,
                placement=self.placement,
                vllm_config=self.vllm_config,
            )
            logger.info(f"worker process created for rank:{rank}", flush=True)
            self.workers.append(worker)
            logger.info("all worker created!", flush=True)

        ready_workers: Set[int] = set()

        startup_timeout_seconds = 60  # TIMEOUT THRES
        end_time = time.monotonic() + startup_timeout_seconds

        while len(ready_workers) < self.tp_size:
            remaining_time = (end_time - time.monotonic()) * 1000
            if remaining_time <= 0:
                raise TimeoutError(
                    f"Executor initialization failed. Only {len(ready_workers)}/{self.tp_size} "
                    f"workers reported ready within {startup_timeout_seconds} seconds. "
                    f"Missing ranks: {set(range(self.tp_size)) - ready_workers}"
                )

            socks = dict(self.worker_response_poller.poll(timeout=int(remaining_time)))

            if self.recv_from_workers in socks:
                response: WorkerReadyResponse = self.recv_from_workers.recv_pyobj()
                try:
                    if isinstance(response, WorkerReadyResponse):
                        if response.rank not in ready_workers:
                            logger.info(f"Worker {response.rank} is ready.")
                            ready_workers.add(response.rank)
                        else:
                            logger.warning(
                                f"Received duplicate ready signal from worker {response.rank}."
                            )
                    else:
                        logger.warning(
                            f"Received unexpected response of type '{type(response)}' "
                            "during initialization. Expecting 'worker_ready'."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to parse response from a worker: {e}. Raw response: '{response}'"
                    )
        logger.info("All workers are ready. Executor initialization is complete.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        kwargs = kwargs or {}
        timeout_ms = -1 if timeout is None else int(timeout * 1000)
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL
                )
            logger.info(f"Sending RPC request: {send_method}", flush=True)
            command = CollectiveRpcCommand(method=send_method, args=args, kwargs=kwargs)

            self.send_to_workers.send_pyobj(command)

            responses_received = 0
            responses = [None] * self.world_size
            while responses_received < self.world_size:
                socks = dict(self.worker_response_poller.poll(timeout=timeout_ms))
                if self.recv_from_workers in socks:
                    response: CollectiveRpcResponse = (
                        self.recv_from_workers.recv_pyobj()
                    )
                    print(
                        f"!!!! recevied collective rpc response : {response}",
                        flush=True,
                    )
                    assert isinstance(response, CollectiveRpcResponse), (
                        f"Expected CollectiveRpcResponse, got {type(response)}"
                    )
                    if response.command_id != command.command_id:
                        logger.info(
                            f"Received a stale RPC response for command {response.command_id}, expecting {command.command_id}. Discarding.",
                            flush=True,
                        )
                        continue

                    if responses[response.rank] is None:
                        responses_received += 1

                    responses[response.rank] = response.data

                else:
                    raise TimeoutError
            return responses
        except TimeoutError:
            logger.error(
                f"RPC call timed out after {timeout}s. "
                f"Received {responses_received}/{self.world_size} responses.",
                flush=True,
            )
            return responses
        except Exception as e:
            logger.error(f"Error occurred during RPC call: {e}", flush=True)
            return responses

    def check_health(self):
        self.collective_rpc("check_health")

    def offload_model_weights(self, command: OffloadModelWeightCommand):
        # TODO(daibo): add return value check
        self.collective_rpc("offload_model_weights", args=(command,))
        return OffloadModelWeightResponse(command_id=command.command_id)

    def sync_hf_weight(self, command: SyncHFWeightCommand):
        # TODO(daibo): add return value check
        self.collective_rpc("sync_hf_weight", args=(command,))
        return SyncHFWeightResponse(command_id=command.command_id)

    def _ensure_worker_termination(self):
        def wait_for_termination(procs, timeout):
            if not time:
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        if getattr(self, "shutting_down", False):
            self.shutting_down = True

            try:
                self.send_to_engine.close()
                self.recv_from_engine.close()
                self._ensure_worker_termination()
                self.send_to_workers.close()
                self.recv_from_workers.close()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
