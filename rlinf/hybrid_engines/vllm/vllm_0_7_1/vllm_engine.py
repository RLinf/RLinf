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


from typing import List, Optional, Union

from vllm import RequestOutput
from vllm.config import VllmConfig
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.llm_engine import LLMEngine as _LLMEngine

from .executor import VLLMExecutor


class VLLMEngine:
    def __init__(
        self, vllm_config: VllmConfig, log_stats: bool, multiprocess_model: bool = False
    ):
        self._engine = _LLMEngine(
            vllm_config=vllm_config,
            executor_class=VLLMExecutor,
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

    def _run_engine(self):
        outputs: List[RequestOutput] = []

        while self._engine.has_unfinished_requests():
            step_outputs = self._engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))

    def offload_model_weights(self) -> None:
        raise NotImplementedError(
            "VLLMEngine.offload_model_weights is not implemented yet."
        )

    def sync_hf_weight(self) -> None:
        raise NotImplementedError("VLLMEngine.sync_hf_weight is not implemented yet.")
