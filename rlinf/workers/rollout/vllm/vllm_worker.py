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
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from omegaconf import DictConfig
from PIL.Image import Image
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs.data import TextPrompt, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine

from rlinf.algorithms.math.verifier.verify import MathRewardModel, math_verify_call
from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.utils import print_vllm_outputs

from . import VLLMEngine, VLLMExecutor


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
        # set False to use Inproclient, which uses sync calls.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_ATTENTION_BACKEND"] = self._cfg.rollout.vllm.attention_backend

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

    def _validate_weight_at_first(self) -> None:
        """
        Validate the model weights before starting to rollout formally.
        """
        if self._cfg.rollout.detokenize:
            outputs = self._vllm_engine.generate(
                input_ids=None,
                sampling_params=self._validate_sampling_params,
                return_logprobs=False,
                prompt_texts=self._validate_prompts,
            )
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            outputs = self._vllm_engine.generate(
                input_ids=prompt_ids,
                sampling_params=self._validate_sampling_params,
                return_logprobs=False,
            )
        print_vllm_outputs(outputs=outputs)
        print("===============================", flush=True)

    def sync_model_from_actor(self) -> None:
        """
        Use vllm_engine to sync model weights from the actor.
        """
        self._vllm_engine.sync_hf_weight()

    def init_worker(self) -> None:
        """
        Use EngineArgs and VllmConfig to initialize the VLLM engine.
        Then offload the model weights, ready to use weights sent from actor.
        """
        engine_args: EngineArgs = EngineArgs(
            model=self._cfg.rollout.model_dir,
            tensor_parallel_size=self._cfg.rollout.tensor_parallel_size,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            gpu_memory_utilization=self._cfg.rollout.gpu_memory_utilization,
            enforce_eager=self._cfg.rollout.enforce_eager,
            enable_chunked_prefill=self._cfg.rollout.vllm.enable_chunked_prefill,
            enable_prefix_caching=self._cfg.rollout.vllm.enable_prefix_caching,
            task="generate",
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
        self.log_info(f"[LLM dp {self._rank}] start to initialize VLLM engine")
        self._vllm_engine = VLLMEngine(
            rlinf_config=self._cfg,
            vllm_config=vllm_config,
            log_stats=not self._cfg.rollout.disable_log_stats,
            multiprocess_model=False,  # use Inproclient
            parent_address=self.worker_address,
            placement=self._placement,
            dp_rank=self._rank,
        )
        self.log_info(f"[LLM dp {self._rank}] VLLM engine initialized.")
        self._vllm_engine.offload_model_weights()

    def _stop(self) -> None:
        """
        Helper function to stop the VLLM engine and offload model weights.
        This should only be called when vllm engine has no more requests to process.
        """
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        self._vllm_engine.offload_model_weights()

    def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        """
        The main rollout function to interact with the VLLM engine.
        It receives RolloutRequest from input_channel, and sends back RolloutResult
        to output_channel.

        Args:
            input_channel (Channel): The channel to receive RolloutRequest.
            output_channel (Channel): The channel to send RolloutResult.
        """
        request: RolloutRequest = input_channel.get()

        requests: List[RolloutRequest] = request.repeat_and_split(
            self._rollout_batch_size
        )

        # Acquire the GPUs to ensure no one is using them during rollout
        output_channel.device_lock.acquire()

        rollout_results: List[RolloutResult] = []
        for request in requests:
            with self.worker_timer():
                vllm_results = self._vllm_engine.generate(
                    input_ids=request.input_ids,
                    sampling_params=self._sampling_params,
                    return_logprobs=self._return_logprobs,
                )
            # should be converted by _vllm_engine side.
            results = RolloutResult.from_vllm_results(
                group_size=self._cfg.algorithm.group_size,
                results=vllm_results,
                answers=request.answers,
                return_logprobs=self._return_logprobs,
            )
            rollout_results.append(results)
            if self._cfg.rollout.print_outputs:
                print_vllm_outputs(outputs=vllm_results)

        # Stop and offload SGLang first before putting into channel
        # This avoids running SGLang and Megatron simultaneously
        self._stop()
        # Release the GPUs once the engine has offloaded
        output_channel.device_lock.release()
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        output_channel.put(rollout_result)


def all_floats_equal(float_list: list[float], epsilon: float = 1e-9) -> bool:
    if len(float_list) <= 1:
        return True
    return np.std(float_list) < epsilon


class AsyncVLLMWorker(VLLMWorker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        super().__init__(config, placement)

        self._reward_model = MathRewardModel(self._cfg.reward.reward_scale)
        self.request_counter = Counter()

    def _prepare_vllm_environment(self) -> None:
        """
        Set up environment variables for VLLM.
        """
        super()._prepare_vllm_environment()
        # set True to use AsyncMPClient, which uses async calls.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

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
        raise NotImplementedError("Async validate is not implemented yet.")

    async def offload_model_weights(self) -> None:
        """
        Use async_engine to offload model weights/kv cache.
        """
        await self._async_engine.collective_rpc("offload_model_weights")

    async def sync_model_from_actor(self) -> None:
        """
        Sync model weights from actor to the vllm workers.
        """
        await self._async_engine.collective_rpc("sync_hf_weight")

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

    async def _generate(
        self,
        request_id: str,
        input_ids: List[int],
        sampling_params: SamplingParams,
        prompt_text: Optional[str] = None,
        image_data: Optional[List[Union[bytes, str]]] = None,
    ) -> Tuple[str, RequestOutput]:
        """
        An async wrapper of async_engine.generate, which only returns AsyncGenerator.
        args are the same as async_engine.generate.

        Args:
            input_ids (List[List[int]]): A batch of input token ids.
            sampling_params (SamplingParams): Sampling parameters for generation.
            prompt_texts (Optional[List[str]]): A batch of input prompt texts. If not None,
                input_ids will be ignored.
            image_data (Optional[List[List[Union[bytes,str]]]]): A batch of image data,
                each item can be bytes or image path (local or URL). The outer list should
                have the same length as input_ids or prompt_texts. The inner list contains
                multiple images for each input.

        Returns:
            List[RequestOutput]: A list of RequestOutput from vllm.
            str: The request id of this generation.
        """
        image_list = self._process_image_data(image_data=image_data)
        if prompt_text is not None:
            prompt = TextPrompt(prompt=prompt_text, multi_modal_data=image_list)
        else:
            prompt = TokensPrompt(
                prompt_token_ids=input_ids, multi_modal_data=image_list
            )
        async_generator = self._async_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        output = await self._get_output_from_async_generator(async_generator)
        return request_id, output

    async def init_worker(self) -> None:
        """
        Use EngineArgs and VllmConfig to initialize VLLM async engine.
        Then offload the model weights, ready to use weights sent from actor.
        """
        engine_args: EngineArgs = EngineArgs(
            model=self._cfg.rollout.model_dir,
            tensor_parallel_size=self._cfg.rollout.tensor_parallel_size,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            gpu_memory_utilization=self._cfg.rollout.gpu_memory_utilization,
            enforce_eager=self._cfg.rollout.enforce_eager,
            enable_chunked_prefill=self._cfg.rollout.vllm.enable_chunked_prefill,
            enable_prefix_caching=self._cfg.rollout.vllm.enable_prefix_caching,
            task="generate",
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

    async def _put_result(self, result: RolloutResult, output_channel: Channel) -> None:
        await output_channel.put(result, async_op=True).async_wait()

    async def _stop(self) -> None:
        """
        Helper function to stop the VLLM engine and offload model weights.
        This should only be called when vllm engine has no more requests to process.
        """
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        await self.offload_model_weights()

    async def _compute_reward_and_advantage(
        self, output: RequestOutput, answer: str
    ) -> Tuple[List[float], List[float]]:
        output_lengths = len(output.outputs)
        assert output_lengths == self._cfg.algorithm.group_size, (
            f"Output lengths {output_lengths} != group size {self._cfg.algorithm.group_size}"
        )
        answer = [answer] * output_lengths
        texts = [out.text for out in output.outputs]
        results = math_verify_call(texts, answer)
        rewards = [(1 if r else -1) * self._reward_model.scale for r in results]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)

        mean = rewards_tensor.mean()
        std = rewards_tensor.std(unbiased=False)
        advantages = (rewards_tensor - mean) / (std + 1e-6)

        return rewards, advantages.tolist()

    async def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        answers: Dict[str, str] = {}
        with self.worker_timer():
            rollout_tasks = []
            for input_id, answer in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                request_id = str(next(self.request_counter))
                rollout_tasks.append(
                    asyncio.create_task(
                        self._generate(
                            request_id=request_id,
                            input_ids=input_id,
                            sampling_params=self._sampling_params,
                        )
                    )
                )
                answers[request_id] = answer

            return_tasks = []
            total_reqs = len(rollout_tasks) * rollout_request.n
            required_reqs = total_reqs // self._cfg.algorithm.max_num_gen_batches

            dropped_reqs = 0
            finished_reqs = 0
            abort_flag = False

            for future in asyncio.as_completed(rollout_tasks):
                request_id, request_output = await future
                rewards, advantages = await self._compute_reward_and_advantage(
                    output=request_output, answer=answers[request_id]
                )

                if (
                    all_floats_equal(rewards)
                    and self._cfg.algorithm.get("max_num_gen_batches", 1) > 1
                ):
                    if (total_reqs - dropped_reqs) > required_reqs:
                        dropped_reqs += rollout_request.n
                        continue

                answer = answers[request_id]
                rollout_result: RolloutResult = RolloutResult.from_vllm_result(
                    group_size=rollout_request.n,
                    vllm_result=request_output,
                    answer=answer,
                    return_logprobs=self._return_logprobs,
                )
                rollout_result.rewards = torch.tensor(
                    rewards, dtype=torch.float32
                ).reshape(-1, 1)
                rollout_result.advantages = advantages
                return_tasks.append(
                    asyncio.create_task(
                        self._put_result(rollout_result, output_channel)
                    )
                )
                finished_reqs += rollout_request.n
                if finished_reqs >= required_reqs:
                    abort_flag = True
                    break
            if abort_flag:
                await self._async_engine.abort("")
            await asyncio.gather(*return_tasks)
