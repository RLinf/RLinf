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
import io
import os
from functools import partial
from typing import AsyncGenerator, List, Optional, Union

import requests
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs.data import PromptType, TextPrompt, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.utils import print_vllm_outputs

from . import VLLMExecutor


class VLLMWorker(Worker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)
        self._cfg = config
        self._placement = placement

        self._prepare_vllm_environment()
        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = self._get_sampling_params_from_config()
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.rollout.model_dir)
        self._vllm_engine = None

        if self._cfg.algorithm.rollout_batch_size_per_gpu is None:
            self._rollout_batch_size = None
        else:
            self._rollout_batch_size = (
                self._cfg.algorithm.rollout_batch_size_per_gpu
                * self._cfg.rollout.tensor_parallel_size
                * self._cfg.rollout.pipeline_parallel_size
            )

        self._validate_sampling_params = SamplingParams(temperature=0, max_tokens=32)
        self._validate_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        self.request_counter = Counter()

    def _prepare_vllm_environment(self) -> None:
        """
        Set up environment variables for VLLM.
        """
        # use v1 engine
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
            "1" if self._cfg.rollout.vllm.enable_flash_infer_sampler else "0"
        )
        # use spawn to avoid fork issues with CUDA
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_ATTENTION_BACKEND"] = self._cfg.rollout.vllm.attention_backend
        # set True to use AsyncMPClient, which uses async calls.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
        if self._cfg.rollout.vllm.torch_profiler_dir is not None:
            os.environ["VLLM_TORCH_PROFILER_DIR"] = (
                self._cfg.rollout.vllm.torch_profiler_dir
            )
            if not os.path.exists(self._cfg.rollout.vllm.torch_profiler_dir):
                os.makedirs(self._cfg.rollout.vllm.torch_profiler_dir)

    def _get_sampling_params_from_config(self) -> SamplingParams:
        """
        Get sampling parameters built from the configuration.
        """
        cfg_sampling_params = self._cfg.algorithm.sampling_params
        if cfg_sampling_params.use_greedy:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=cfg_sampling_params.max_new_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
                n=self._cfg.algorithm.group_size,
                logprobs=0 if self._return_logprobs else None,
            )
        else:
            sampling_params = SamplingParams(
                temperature=cfg_sampling_params.temperature,
                top_k=cfg_sampling_params.top_k,
                top_p=cfg_sampling_params.top_p,
                repetition_penalty=cfg_sampling_params.repetition_penalty,
                max_tokens=cfg_sampling_params.max_new_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
                n=self._cfg.algorithm.group_size,
                logprobs=0 if self._return_logprobs else None,
            )
        return sampling_params

    def _process_image_data(
        self, image_data: Optional[List[Union[bytes, str]]]
    ) -> Optional[List[Image]]:
        """
        Process the batch image data which can be bytes or image paths.

        Args:
            batch_image_data (Optional[List[List[Union[bytes,str]]]]): A batch of
                image data, each item can be bytes or image path (local or URL).
        Returns:
            Optional[List[List[Image]]]: A batch of list of PIL Image. If input
                is None, return None.
        """
        if image_data is None:
            return None
        if not isinstance(image_data, list):
            raise ValueError("image_data should be a list of list of image data.")
        image_list = []
        for img in image_data:
            if isinstance(img, bytes):
                image = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                if img.startswith("http://") or img.startswith("https://"):
                    response = requests.get(img)
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(img)
            else:
                raise ValueError("Unsupported image data type.")
            image_list.append(image)
        return image_list

    async def _validate_weight_at_first(self) -> None:
        """
        Validate the model weights before starting to rollout formally.
        """
        if self._cfg.rollout.detokenize:
            vllm_outputs = await self.generate(
                input_ids=None,
                sampling_params=self._validate_sampling_params,
                prompt_texts=self._validate_prompts,
            )
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            vllm_outputs = await self.generate(
                input_ids=prompt_ids,
                sampling_params=self._validate_sampling_params,
            )
        for request_output in vllm_outputs:
            print_vllm_outputs(request_output, self._tokenizer)

    async def offload_model_weights(self) -> None:
        """
        Use async_engine to offload model weights/kv cache.
        """
        await self._async_engine.reset_prefix_cache()
        await self._async_engine.collective_rpc("offload_model_weights")

    async def sync_model_from_actor(self) -> None:
        """
        Sync model weights from actor to the vllm workers.
        """
        await self._async_engine.collective_rpc("sync_hf_weight")
        await self._async_engine.reset_prefix_cache()

    async def _get_output_from_async_generator(
        self, async_generator: AsyncGenerator[RequestOutput, None]
    ) -> RequestOutput:
        """
        Helper function to get the final output from an async generator.
        """
        output: RequestOutput = None
        async for out in async_generator:
            output = out
        assert output is not None, "Async generator returned no output."
        return output

    def _pre_process_rollout_request(
        self,
        request: RolloutRequest,
    ) -> List[List[RolloutRequest]]:
        if self._rollout_batch_size is not None:
            # NOTE:
            # it's different from sglang, here a request's sample count
            # instead of sample count x group_size  should be divisible by rollout_batch_size
            assert len(request.input_ids) % self._rollout_batch_size == 0, (
                f"rollout_batch_size {self._rollout_batch_size} must divide the total number of requests {len(request.input_ids)}"
            )
            num_batch = len(request.input_ids) // self._rollout_batch_size
        else:
            num_batch = 1

        split_requests = request.split(num_batch)
        if self._placement.is_disaggregated:
            num_prompts_per_request = len(split_requests[0].input_ids)
            return [r.split(num_prompts_per_request) for r in split_requests]
        else:
            return [r.split(1) for r in split_requests]

    async def generate(
        self,
        input_ids: Union[List[List[int]], List[int]],
        sampling_params: SamplingParams,
        prompt_texts: Optional[Union[List[str], str]] = None,
        image_data: Optional[
            Union[List[List[Union[bytes, str]]], List[Union[bytes, str]]]
        ] = None,
    ) -> List[RequestOutput]:
        """
        Do Generate Task using the vllm async engine.

        Args:
            input_ids: The input token ids to generate. It can be a list of list of int,
                or a list of int (single prompt).
            sampling_params: The sampling parameters to use for generation.
            prompt_texts: The input prompt texts to generate. It can be a list of strings
                or a single string. If provided, it will be used instead of input_ids.
            image_data: The input multi-modal data to generate. It can be a list of list
                of bytes or image paths (local or URL), or a list of bytes or image paths
                (single prompt).

        Returns:
            List[RequestOutput]: A list of RequestOutput from vllm engine.
        """

        def check_input_ids() -> List[List[int]]:
            assert isinstance(input_ids, list), (
                "input_ids should be a list or list of list of int."
            )
            assert len(input_ids) > 0, "input_ids should not be empty."
            if isinstance(input_ids[0], int):
                return [input_ids]
            else:
                return input_ids

        def check_prompt_text() -> Optional[List[str]]:
            if prompt_texts is None:
                return None
            assert isinstance(prompt_texts, list) or isinstance(prompt_texts, str), (
                "prompt_text should be a string or list of strings."
            )
            if isinstance(prompt_texts, str):
                return [prompt_texts]
            else:
                assert len(prompt_texts) > 0, "prompt_text should not be empty."
                return prompt_texts

        def check_image_data() -> Optional[List[List[Image.Image]]]:
            if image_data is None or not any(image_data):
                return None
            assert isinstance(image_data, list), "image_data should be a list."
            if isinstance(image_data[0], list):
                return image_data
            else:
                return [image_data]

        input_ids = check_input_ids()
        prompt_texts = check_prompt_text()
        image_list = check_image_data()

        inputs: List[PromptType] = []
        outputs: List[RequestOutput] = []
        if prompt_texts is not None:
            for i, prompt_text in enumerate(prompt_texts):
                if image_list is not None:
                    images = self._process_image_data(image_data=image_list[i])
                    inputs.append(
                        TextPrompt(
                            prompt=prompt_text, multi_modal_data={"image": images}
                        )
                    )
                else:
                    inputs.append(TextPrompt(prompt=prompt_text))
        else:
            for i, input_id in enumerate(input_ids):
                if image_list is not None:
                    images = self._process_image_data(image_data=image_list[i])
                    inputs.append(
                        TokensPrompt(
                            prompt_token_ids=input_id,
                            multi_modal_data={"image": images},
                        )
                    )
                else:
                    inputs.append(TokensPrompt(prompt_token_ids=input_id))

        outputs = await asyncio.gather(
            *[
                self._get_output_from_async_generator(
                    self._async_engine.generate(
                        prompt=inp,
                        sampling_params=sampling_params,
                        request_id=str(next(self.request_counter)),
                    )
                )
                for inp in inputs
            ]
        )

        return outputs

    async def init_worker(self) -> None:
        """
        Use EngineArgs and VllmConfig to initialize VLLM async engine.
        If mode is collocated, it will additionally offload model weights,
        ready to use parameters sent from actor.
        """
        engine_args: EngineArgs = EngineArgs(
            model=self._cfg.rollout.model_dir,
            tensor_parallel_size=self._cfg.rollout.tensor_parallel_size,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            gpu_memory_utilization=self._cfg.rollout.gpu_memory_utilization,
            enforce_eager=self._cfg.rollout.enforce_eager,
            enable_chunked_prefill=self._cfg.rollout.vllm.enable_chunked_prefill,
            enable_prefix_caching=self._cfg.rollout.vllm.enable_prefix_caching,
            max_num_batched_tokens=self._cfg.rollout.vllm.max_num_batched_tokens,
            task="generate",
            load_format="dummy" if not self._cfg.rollout.validate_weight else "auto",
            trust_remote_code=self._cfg.actor.tokenizer.trust_remote_code,
            max_model_len=self._cfg.runner.seq_length,
            max_num_seqs=self._cfg.rollout.max_running_requests,
            enable_sleep_mode=True,  # it enables offload weights
        )
        vllm_config: VllmConfig = engine_args.create_engine_config()

        # here to set the customed worker class for VLLM engine
        vllm_worker_cls = "rlinf.hybrid_engines.vllm.vllm_0_8_5.worker.VLLMWorker"
        vllm_config.parallel_config.worker_cls = vllm_worker_cls

        self.log_info(f"vllm_config is {vllm_config}")

        executor_class = partial(
            VLLMExecutor,
            rlinf_config=self._cfg,
            parent_address=self.worker_address,
            placement=self._placement,
            dp_rank=self._rank,
        )

        self._async_engine = AsyncLLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not self._cfg.rollout.disable_log_stats,
            log_requests=False,  # do not need to log each request
        )

        self.log_info(f"[LLM dp {self._rank}] VLLM engine initialized.")

        if not self._placement.is_disaggregated:
            await self.offload_model_weights()

    async def _put_result(self, result: RolloutResult, output_channel: Channel) -> None:
        """
        Helper function to put the result to output channel.

        Args:
            result: The RolloutResult to put to the channel.
            output_channel: The output channel to send results to.
        """
        # NOTE:
        # To fit reward worker and actor workers' expected input count,
        # currently we can only split result into groups.
        splited_results = RolloutResult.split_result_list_by_group([result])
        put_tasks = [
            output_channel.put(r, async_op=True).async_wait() for r in splited_results
        ]
        await asyncio.gather(*put_tasks)

    async def _stop(self) -> None:
        """
        Helper function to stop the VLLM engine and offload model weights.
        This should only be called when vllm engine has no more requests to process.
        """
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        if not self._placement.is_disaggregated:
            await self.offload_model_weights()

    async def rollout_and_return(
        self, request: RolloutRequest, output_channel: Channel
    ):
        """
        Helper function to rollout for a single RolloutRequest and build RolloutResult then
        put it to output channel.

        Args:
            request: The RolloutRequest to process.
            output_channel: The output channel to send results to.
        """
        vllm_results: List[RequestOutput] = await self.generate(
            input_ids=request.input_ids,
            image_data=request.image_data,
            sampling_params=self._sampling_params,
        )
        rollout_result: RolloutResult = RolloutResult.from_vllm_results(
            group_size=self._cfg.algorithm.group_size,
            results=vllm_results,
            answers=request.answers,
            multi_modal_inputs=request.multi_modal_inputs,
            return_logprobs=self._return_logprobs,
        )
        if self._cfg.rollout.print_outputs:
            print_vllm_outputs(outputs=vllm_results)
        await self._put_result(result=rollout_result, output_channel=output_channel)

    async def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        """
            Perform rollout using vllm engine.
            It will read `RolloutRequest` from input_channel and put `RolloutResult` to output_channel.
            If the input request is None, it will stop the rollout.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        output_channel.device_lock.acquire()
        batched_requests = self._pre_process_rollout_request(rollout_request)
        with self.worker_timer():
            for requests in batched_requests:
                rollout_tasks: List[asyncio.Task] = []
                for request in requests:
                    rollout_tasks.append(
                        asyncio.create_task(
                            self.rollout_and_return(
                                request=request, output_channel=output_channel
                            )
                        )
                    )
                await asyncio.gather(*rollout_tasks)
            await self._stop()
        output_channel.device_lock.release()
