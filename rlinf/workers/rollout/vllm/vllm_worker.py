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
import copy
import os
from functools import partial
from typing import AsyncGenerator, Optional, Union, cast

from omegaconf import DictConfig
from PIL import Image
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs.data import PromptType, TextPrompt, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult, SeqGroupInfo
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.dynamic_scheduler.manager import RolloutScalingScheduler
from rlinf.scheduler.dynamic_scheduler.utils import get_scheduler_channel
from rlinf.utils.data_process import process_image_data
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.utils import RunningStatusManager, print_vllm_outputs

from . import VLLMExecutor


class VLLMWorker(Worker):
    def __init__(self, config: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self._cfg = config
        self._placement = placement

        self._prepare_vllm_environment()
        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = self._get_sampling_params_from_config()
        self._tokenizer = self._load_tokenizer()
        self._vllm_engine = None

        self._validate_sampling_params = SamplingParams(temperature=0, max_tokens=32)
        self._validate_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        self._request_counter = Counter()

        # NOTE(daibo):
        # because 0.8.5 vLLM can not return outputs when generation
        # request is aborted, we need to track the running generate tasks
        # in vLLM instance and cancel them in asyncio level and get
        # the final outputs from async generator as aborted generation
        # output. If newer vLLM version supports returning outputs
        # everything about _running_generate_tasks can be removed.
        self._running_generate_tasks: dict[str, asyncio.Task] = {}
        self.status_manager = RunningStatusManager()
        self._use_auto_scheduler = self._placement.is_auto

        if self._use_auto_scheduler:
            self._init_scheduler()

    def _init_scheduler(self) -> None:
        self.schedule_channel = self.connect_channel(
            get_scheduler_channel("rollout", self._rank)
        )

        self._scheduler = RolloutScalingScheduler(
            self._rank, self.schedule_channel, self
        )

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

    def _load_tokenizer(self):
        model_path = self._cfg.rollout.model.model_path
        trust_remote_code = self._cfg.actor.tokenizer.get("trust_remote_code", False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError):
            self.log_warning(
                "Fast tokenizer unavailable; falling back to the slow tokenizer."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )
        return tokenizer

    def _get_sampling_params_from_config(self) -> SamplingParams:
        """
        Get sampling parameters built from the configuration.
        """
        cfg_sampling_params = self._cfg.algorithm.sampling_params
        if not cfg_sampling_params.do_sample:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=cfg_sampling_params.max_new_tokens,
                logprobs=0 if self._return_logprobs else None,
            )
        else:
            sampling_params = SamplingParams(
                temperature=cfg_sampling_params.temperature,
                top_k=cfg_sampling_params.top_k,
                top_p=cfg_sampling_params.top_p,
                repetition_penalty=cfg_sampling_params.repetition_penalty,
                max_tokens=cfg_sampling_params.max_new_tokens,
                logprobs=0 if self._return_logprobs else None,
            )
        return sampling_params

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

    async def offload_engine(self) -> None:
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
        try:
            async for out in async_generator:
                output = out
            assert output is not None, "Async generator returned no output."
        except asyncio.CancelledError:
            pass
        return output

    async def generate(
        self,
        input_ids: Union[list[list[int]], list[int]],
        sampling_params: SamplingParams,
        prompt_texts: Optional[Union[list[str], str]] = None,
        image_data: Optional[
            Union[list[list[Union[bytes, str]]], list[Union[bytes, str]]]
        ] = None,
    ) -> list[RequestOutput]:
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

        def check_input_ids() -> list[list[int]]:
            assert isinstance(input_ids, list), (
                "input_ids should be a list or list of list of int."
            )
            assert len(input_ids) > 0, "input_ids should not be empty."
            if isinstance(input_ids[0], int):
                return [input_ids]
            else:
                return input_ids

        def check_prompt_text() -> Optional[list[str]]:
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

        def check_image_data() -> Optional[list[list[Union[bytes, str]]]]:
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
        processed_images: Optional[list[list[Image.Image]]] = None
        if image_list is not None:
            gathered_images = await asyncio.gather(
                *(process_image_data(images) for images in image_list)
            )
            processed_images = [
                cast(list[Image.Image], images) for images in gathered_images
            ]
        inputs: list[PromptType] = []
        if prompt_texts is not None:
            for i, prompt_text in enumerate(prompt_texts):
                if processed_images is not None:
                    images = processed_images[i]
                    inputs.append(
                        TextPrompt(
                            prompt=prompt_text, multi_modal_data={"image": images}
                        )
                    )
                else:
                    inputs.append(TextPrompt(prompt=prompt_text))
        else:
            for i, input_id in enumerate(input_ids):
                if processed_images is not None:
                    images = processed_images[i]
                    inputs.append(
                        TokensPrompt(
                            prompt_token_ids=input_id,
                            multi_modal_data={"image": images},
                        )
                    )
                else:
                    inputs.append(TokensPrompt(prompt_token_ids=input_id))

        # use local_tasks to track current generation request's tasks
        # which is subset of self._running_generate_tasks, after this
        # generate is done, we will remove them from self._running_generate_tasks
        local_tasks: dict[str, asyncio.Task] = {}
        for inp in inputs:
            request_id = str(next(self._request_counter))
            task = asyncio.create_task(
                self._get_output_from_async_generator(
                    self._async_engine.generate(
                        prompt=inp,
                        sampling_params=sampling_params,
                        request_id=request_id,
                    )
                )
            )
            self._running_generate_tasks[request_id] = task
            local_tasks[request_id] = task
        outputs: list[RequestOutput] = await asyncio.gather(*local_tasks.values())
        # clean up the local tasks from the global running tasks
        # after generation is done
        for request_id in local_tasks.keys():
            self._running_generate_tasks.pop(request_id, None)

        return outputs

    async def abort_generation(self) -> None:
        """
        Abort all ongoing and waiting generations in the vllm async engine.
        """
        # we first cancel all running asyncio Task in a sync way
        # then send abort signal to vLLM engine to make sure
        # the generation requests are aborted in vLLM side
        current_request_ids = list(self._running_generate_tasks.keys())
        for request_id in current_request_ids:
            task: asyncio.Task = self._running_generate_tasks.pop(request_id, None)
            if task is not None:
                task.cancel()

    async def init_worker(self) -> None:
        """
        Use EngineArgs and VllmConfig to initialize VLLM async engine.
        If mode is collocated, it will additionally offload model weights,
        ready to use parameters sent from actor.
        """
        engine_args: EngineArgs = EngineArgs(
            model=self._cfg.rollout.model.model_path,
            tensor_parallel_size=self._cfg.rollout.tensor_parallel_size,
            dtype=torch_dtype_from_precision(self._cfg.rollout.model.precision),
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

        if self._placement.is_collocated:
            await self.offload_engine()
        if self._use_auto_scheduler:
            asyncio.create_task(self._scheduler.main_loop())

    async def _async_generate_group(self, seq_group_info: SeqGroupInfo) -> SeqGroupInfo:
        async def generate_with_idx(
            idx: int,
            input_ids: list[int],
            image_data: Optional[list[Union[bytes, str]]],
            sampling_params: SamplingParams,
        ) -> tuple[int, RequestOutput]:
            outputs = await self.generate(
                input_ids=input_ids,
                image_data=image_data,
                sampling_params=sampling_params,
            )
            return idx, outputs[0]

        if seq_group_info.num_aborted == 0:
            assert seq_group_info.num_returned == 0
            seq_idx_list = list(range(seq_group_info.group_size))
            input_batch = [
                list(seq_group_info.input_ids) for _ in range(seq_group_info.group_size)
            ]
            if seq_group_info.image_data is None:
                image_data_list = [None] * seq_group_info.group_size
            else:
                image_data_list = [
                    list(seq_group_info.image_data)
                    for _ in range(seq_group_info.group_size)
                ]
            sampling_params_list: list[SamplingParams] = [
                self._sampling_params for _ in range(seq_group_info.group_size)
            ]
        else:
            idx_aborted = seq_group_info.idx_aborted.copy()
            seq_idx_list: list[int] = []
            seq_group_info.idx_aborted.clear()
            input_batch: list[list[int]] = []
            image_data_list: list = []
            sampling_params_list: list[SamplingParams] = []
            for idx in idx_aborted:
                generated_ids: list[int] = (
                    seq_group_info.results[idx].outputs[0].token_ids
                )
                if len(generated_ids) >= self._sampling_params.max_tokens:
                    result = seq_group_info.results[idx]
                    result.outputs[0].finish_reason = "length"
                    seq_group_info.idx_aborted.remove(idx)
                    seq_group_info.idx_completed.add(idx)
                    continue
                seq_idx_list.append(idx)
                input_batch.append(seq_group_info.input_ids + generated_ids)
                if seq_group_info.image_data is None:
                    image_data_list.append(None)
                else:
                    image_data_list.append(list(seq_group_info.image_data))
                sampling_params = copy.deepcopy(self._sampling_params)
                sampling_params.max_tokens -= len(generated_ids)
                sampling_params_list.append(sampling_params)
        tasks = [
            asyncio.create_task(
                generate_with_idx(idx, input_ids, image_data, sampling_params)
            )
            for idx, input_ids, image_data, sampling_params in zip(
                seq_idx_list,
                input_batch,
                image_data_list,
                sampling_params_list,
                strict=True,
            )
        ]
        for future in asyncio.as_completed(tasks):
            idx, request_output = await future
            seq_group_info.record_vllm_result(idx, request_output, self._logger)

        return seq_group_info

    async def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        groups = rollout_request.to_seq_group_infos()
        async_wait_type = (
            asyncio.FIRST_COMPLETED
            if self._placement.is_pipeline
            else asyncio.ALL_COMPLETED
        )
        with self.device_lock, self.worker_timer():
            num_residual = self.status_manager.num_seq_group
            assert num_residual == 0, (
                f"There are {num_residual} "
                f"sequence group{'' if num_residual == 1 else 's'} before rollout."
            )

            for group in groups:
                task = asyncio.create_task(self._async_generate_group(group))
                self.status_manager.add_task(group, task)

            while pending := self.status_manager.get_running_tasks():
                done, pending = await asyncio.wait(pending, return_when=async_wait_type)
                returned_seq_groups: list[SeqGroupInfo] = [
                    task.result() for task in done
                ]
                for group in returned_seq_groups:
                    if group.all_completed:
                        rollout_result = RolloutResult.from_vllm_seq_group(
                            group,
                            self._return_logprobs,
                        )
                        await output_channel.put(
                            item=rollout_result, async_op=True
                        ).async_wait()
                        self.status_manager.mark_done(group)
                    else:
                        self.status_manager.mark_aborted(group)

                if (
                    self._use_auto_scheduler
                    and self.status_manager.num_seq_group_running == 0
                ):
                    # rollout should not exit immediately when using auto scheduler
                    # because there might be migrations
                    # if so, `pending` will not be empty in while loop condition
                    await self.status_manager.wait_notification()

            self.status_manager.clear()

            if self._placement.is_collocated or self._placement.is_auto:
                await self.offload_engine()
                if self._use_auto_scheduler:
                    await self._scheduler.report_offloaded()
