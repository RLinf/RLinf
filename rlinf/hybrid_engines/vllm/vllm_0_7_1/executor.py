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

import json
import signal
import tempfile
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import zmq
from pydantic import TypeAdapter, ValidationError
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor

from rlinf.workers.rollout.vllm.io_struct import (
    OffloadModelWeightCommand,
    SyncHFWeightCommand,
    VLLMCommand,
)

logger = init_logger(__name__)


class VLLMExecutor(Executor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_ipc_input_name: str,
        executor_ipc_output_name: str,
    ):
        super().__init__(vllm_config)
        self.executor_ipc_input_name = executor_ipc_input_name
        self.executor_ipc_output_name = executor_ipc_output_name

        context = zmq.Context()
        self.recv_from_engine = context.socket(zmq.PULL)
        self.recv_from_engine.connect(self.executor_ipc_input_name)

        self.send_to_engine = context.socket(zmq.PUSH)
        self.send_to_engine.connect(self.executor_ipc_output_name)

        self.command_parser = TypeAdapter(VLLMCommand)

        self.command_handlers = {
            "sync_hf_weight": self.sync_hf_weight,
            "offload_model_weights": self.offload_model_weights,
        }

        self._listen_engine_handle = threading.Thread(
            target=self._listen_engine, daemon=True
        )
        self._listen_engine_handle.start()

    def _listen_engine(self):
        while True:
            if self.recv_from_engine.poll(100):
                try:
                    json_str = self.recv_from_engine.recv_string(flags=zmq.NOBLOCK)
                    parsed_command = self.command_parser.validate_json(json_str)
                    handler = self.command_handlers.get(parsed_command.command_type)
                    if handler:
                        logger.debug(f"Handling command: {parsed_command}")
                        handler(parsed_command)
                    else:
                        logger.warning(
                            f"No handler for command type: {parsed_command.command_type}"
                        )
                except zmq.Again:
                    continue
                except ValidationError as e:
                    print(f"Failed to parse command: {e}")
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")

    def _init_executor(self):
        self._finalizer = weakref.finalize(self, self.shutdown)

        def sigusr1_handler(signum, frame):
            logger.fatal(f"Vllm Executor got signal: {signum} from worker processes")
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size: {self.world_size} is not equal to "
            f"tensor_parallel_size: {tensor_parallel_size}"
        )

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

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> List[Any]:
        raise NotImplementedError("VLLMExecutor.collective_rpc is not implemented yet.")

    def check_health(self):
        self.collective_rpc("check_health")

    def offload_model_weights(self, command: OffloadModelWeightCommand):
        self.collective_rpc("offload_model_weights", args=(command,))

    def sync_hf_weight(self, command: SyncHFWeightCommand):
        self.collective_rpc("sync_hf_weight", args=(command,))
