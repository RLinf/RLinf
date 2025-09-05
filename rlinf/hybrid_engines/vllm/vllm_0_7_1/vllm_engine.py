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
import tempfile
from functools import partial
from typing import List, Optional, Union

import cloudpickle
import zmq
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.llm_engine import LLMEngine as _LLMEngine

from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.vllm.io_struct import (
    OffloadModelWeightCommand,
    OffloadModelWeightResponse,
    SyncHFWeightCommand,
    SyncHFWeightResponse,
)


class VLLMEngine:
    def __init__(
        self,
        rlinf_config: DictConfig,
        vllm_config: VllmConfig,
        log_stats: bool,
        dp_rank: int,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        multiprocess_model: bool = False,
    ):
        self.executor_ipc_input_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )
        self.executor_ipc_output_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )

        context = zmq.Context()
        self.send_to_executor = context.socket(zmq.PUSH)
        self.send_to_executor.bind(self.executor_ipc_input_name)

        self.recv_from_executor = context.socket(zmq.PULL)
        self.recv_from_executor.bind(self.executor_ipc_output_name)

        from rlinf.hybrid_engines.vllm.vllm_0_7_1.worker import VLLMWorker

        vllm_worker_cls = partial(VLLMWorker, rlinf_config=rlinf_config)
        vllm_config.parallel_config.worker_cls = cloudpickle.dumps(vllm_worker_cls)

        from rlinf.hybrid_engines.vllm.vllm_0_7_1.executor import VLLMExecutor

        executor_factory = partial(
            VLLMExecutor,
            executor_ipc_input_name=self.executor_ipc_input_name,
            executor_ipc_output_name=self.executor_ipc_output_name,
            parent_address=parent_address,
            placement=placement,
            dp_rank=dp_rank,
        )

        self._engine = _LLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_factory,
            log_stats=log_stats,
            multiprocess_mode=multiprocess_model,
        )
        self.request_counter = Counter()

    def generate(
        self,
        input_ids: Union[List[List[int]], List[int]],
        sampling_params: Optional[SamplingParams] = None,
        return_logprobs: bool = False,
    ) -> List[RequestOutput]:
        self._add_requests(input_ids, sampling_params)
        results: List[RequestOutput] = self._run_engine()
        return results

    def _add_requests(
        self,
        input_ids: Union[List[List[int]], List[int]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> None:
        assert isinstance(input_ids, list), (
            f"Expected list for input_ids, got {type(input_ids)}"
        )
        if not isinstance(input_ids[0], list):
            input_ids = [input_ids]
        for input_id in input_ids:
            request_id = str(next(self.request_counter))
            tokens_prompt = TokensPrompt(prompt_token_ids=input_id)
            self._engine.add_request(
                request_id=request_id,
                prompt=tokens_prompt,
                params=sampling_params,
            )

    def _run_engine(self) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []

        while self._engine.has_unfinished_requests():
            step_outputs = self._engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))

    def offload_model_weights(self) -> None:
        command = OffloadModelWeightCommand()
        self.send_to_executor.send_pyobj(command)
        response: OffloadModelWeightResponse = self.recv_from_executor.recv_pyobj()
        assert isinstance(response, OffloadModelWeightResponse), (
            f"Expected OffloadModelWeightResponse, got {type(response)}"
        )
        assert response.command_id == command.command_id, (
            f"Expected get {command.command_id}'s response, got, {response.command_id}"
        )

    def sync_hf_weight(self) -> None:
        command = SyncHFWeightCommand()
        self.send_to_executor.send_pyobj(command)
        response: SyncHFWeightResponse = self.recv_from_executor.recv_pyobj()
        assert isinstance(response, SyncHFWeightResponse), (
            f"Expected SyncHFWeightResponse, got {type(response)}"
        )
        assert response.command_id == command.command_id, (
            f"Expected get {command.command_id}'s response, got, {response.command_id}"
        )
        self._engine.reset_prefix_cache()
