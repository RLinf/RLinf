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
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    CompletionInfo,
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler.dynamic_scheduler.utils import (
    RolloutAction,
    RolloutMigrateBatch,
    RolloutReport,
    RolloutScheduleInfo,
    get_scheduler_channel,
    get_scheduler_request_queue,
    get_scheduler_response_queue,
)
from rlinf.utils.placement import ComponentPlacement, PlacementMode
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    print_sglang_outputs,
)
from toolkits.math_verifier.verify import MathRewardModel, math_verify_call


class MetaInfoStatsCollector:
    """Collector for SGLang meta_info statistics

    This collector is only initialized when enabled via configuration.
    Add the following parameters to your generation config section:

    generation:
      collect_meta_stats: true  # Enable meta_info statistics collection
      meta_stats_file: "custom_meta_stats.jsonl"  # Optional: custom output file
      async_meta_stats_file: "custom_async_meta_stats.jsonl"  # Optional: custom async output file
      schedule_meta_stats_file: "custom_schedule_meta_stats.jsonl"  # Optional: custom schedule output file
    """

    def __init__(self, output_file: str = "sglang_meta_stats.jsonl"):
        self.output_file = output_file
        self.stats_buffer = []
        self.buffer_size = 100  # Write to file every 100 records

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else ".", exist_ok=True)

        # Initialize file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                f.write("")  # Create empty file

    def collect_batch_stats(self, outputs: List[Dict], batch_id: int) -> None:
        """Collect statistics from a batch of SGLang outputs

        Args:
            outputs: List of SGLang output dictionaries
            batch_id: Unique identifier for this batch
        """
        current_time = time.time()

        for req_idx, output in enumerate(outputs):
            try:
                # Extract meta_info
                meta_info = output.get('meta_info', {})

                # Extract the specific metrics you requested
                stats_record = {
                    'timestamp': current_time,
                    'batch_id': batch_id,
                    'request_id': f"batch_{batch_id}_req_{req_idx}",
                    'prompt_tokens': meta_info.get('prompt_tokens', None),
                    'completion_tokens': meta_info.get('completion_tokens', None),
                    'e2e_latency': meta_info.get('e2e_latency', None),
                    'ttft': meta_info.get('ttft', None),
                    # Additional useful meta_info fields (if available)
                    'finish_reason': meta_info.get('finish_reason', {}).get('type', None),
                    'total_tokens': (meta_info.get('prompt_tokens', 0) + meta_info.get('completion_tokens', 0)) if meta_info.get('prompt_tokens') is not None and meta_info.get('completion_tokens') is not None else None,

                    # Add any other meta_info fields that might be useful
                    'meta_info_keys': list(meta_info.keys()),  # For debugging/inspection
                }

                self.stats_buffer.append(stats_record)

            except Exception as e:
                # Log error but continue processing
                error_record = {
                    'timestamp': current_time,
                    'batch_id': batch_id,
                    'request_id': f"batch_{batch_id}_req_{req_idx}",
                    'error': str(e),
                    'output_keys': list(output.keys()) if isinstance(output, dict) else 'not_dict'
                }
                self.stats_buffer.append(error_record)

        # Write to file if buffer is full
        if len(self.stats_buffer) >= self.buffer_size:
            self._flush_to_file()

    def _flush_to_file(self) -> None:
        """Write buffered statistics to file"""
        if not self.stats_buffer:
            return

        with open(self.output_file, 'a') as f:
            for record in self.stats_buffer:
                f.write(json.dumps(record) + '\n')

        print(f"Written {len(self.stats_buffer)} records to {self.output_file}")
        self.stats_buffer = []

    def finalize(self) -> None:
        """Flush any remaining data and close"""
        self._flush_to_file()
        print(f"Finalized stats collection. Data saved to {self.output_file}")


class SGLangWorker(Worker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)

        self._cfg = config
        self._placement = placement

        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.rollout.model_dir)
        self._eos = self._cfg.rollout.eos or self._tokenizer.eos_token_id
        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = self._get_sampling_param_from_config()
        if self._cfg.algorithm.rollout_batch_size_per_gpu is None:
            self._rollout_batch_size = None
        else:
            self._rollout_batch_size = (
                self._cfg.algorithm.rollout_batch_size_per_gpu
                * self._cfg.rollout.tensor_parallel_size
                * self._cfg.rollout.pipeline_parallel_size
            )

        self._validate_sampling_params = {"temperature": 0, "max_new_tokens": 32}
        self._validate_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

    def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        if self._cfg.rollout.detokenize:
            outputs = self._engine.generate(
                self._validate_prompts, self._validate_sampling_params
            )
            for prompt, output in zip(self._validate_prompts, outputs):
                print("===============================")
                print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            outputs = self._engine.generate(
                input_ids=prompt_ids, sampling_params=self._validate_sampling_params
            )
            print_sglang_outputs(self._validate_prompts, outputs, self._tokenizer)
        print("===============================", flush=True)

    def _init_engine(self):
        use_cudagraph = not self._cfg.rollout.enforce_eager

        server_args = ServerArgs(
            model_path=self._cfg.rollout.model_dir,
            disable_cuda_graph=not use_cudagraph,
            cuda_graph_max_bs=min(
                self._cfg.rollout.cuda_graph_max_bs,
                self._cfg.rollout.max_running_requests,
            ),
            tp_size=self._cfg.rollout.tensor_parallel_size,
            mem_fraction_static=self._cfg.rollout.gpu_memory_utilization,
            enable_memory_saver=use_cudagraph,
            enable_torch_compile=self._cfg.rollout.sglang.use_torch_compile,
            torch_compile_max_bs=min(
                self._cfg.rollout.sglang.torch_compile_max_bs,
                self._cfg.rollout.max_running_requests,
            ),
            load_format="dummy" if not self._cfg.rollout.validate_weight else "auto",
            # disable_overlap_schedule=True,
            dtype=torch_dtype_from_precision(self._cfg.actor.model.precision),
            # sglang will only return text/output_ids when skip_tokenizer_init=False/True
            # text is not needed in RL training, so set to True can save time.
            skip_tokenizer_init=not self._cfg.rollout.detokenize,
            # sglang will print statistics every decode_log_interval decode steps.
            decode_log_interval=self._cfg.rollout.sglang.decode_log_interval,
            attention_backend=self._cfg.rollout.sglang.attention_backend,
            log_level="info",
            max_running_requests=self._cfg.rollout.max_running_requests,
            dist_init_addr=f"127.0.0.1:{str(Cluster.find_free_port())}",
        )

        self.log_on_first_rank(f"{server_args=}")

        self._engine = Engine(
            parent_address=self.worker_address,
            placement=self._placement,
            config=self._cfg,
            dp_rank=self._rank,
            **dataclasses.asdict(server_args),
        )

    def _get_sampling_param_from_config(self) -> dict:
        """
        Get sampling parameters from the configuration.
        """
        cfg_sampling_params = self._cfg.algorithm.sampling_params
        if cfg_sampling_params.use_greedy:
            sampling_params = {
                "temperature": 0,
                "max_new_tokens": cfg_sampling_params.max_new_tokens,
            }
        else:
            sampling_params = {
                "temperature": cfg_sampling_params.temperature,
                "top_k": cfg_sampling_params.top_k,
                "top_p": cfg_sampling_params.top_p,
                "repetition_penalty": cfg_sampling_params.repetition_penalty,
                "max_new_tokens": cfg_sampling_params.max_new_tokens,
            }
        return sampling_params

    def _stop(self):
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        self._engine.offload_model_weights()

    def init_worker(self):
        # init rollout engine.
        self._init_engine()
        if self._cfg.rollout.validate_weight:
            self._validate_weight_at_first()

        # Rollout Engine should use parameters from actor, so it offloads its parameter first.
        self._engine.offload_model_weights()

    def sync_model_from_actor(self):
        self._engine.sync_hf_weight()

    def rollout(self, input_channel: Channel, output_channel: Channel):
        request: RolloutRequest = input_channel.get()

        # Repeat prompts based on the group_size config
        requests = request.repeat_and_split(self._rollout_batch_size)

        # Acquire the GPUs to ensure no one is using them during rollout
        output_channel.gpu_lock.acquire()
        rollout_results = []
        for request in requests:
            # Generate outputs using the SGLang engine.
            with self.worker_timer():
                results = self._engine.generate(
                    input_ids=request.input_ids,
                    sampling_params=self._sampling_params,
                    return_logprob=self._return_logprobs,
                )

            # Create RolloutResult from the outputs.
            rollout_result = RolloutResult.from_sglang_results(
                results,
                request.n,
                request.input_ids,
                request.answers,
                self._return_logprobs,
            )
            rollout_results.append(rollout_result)

            # Put and print results
            if self._cfg.rollout.print_outputs:
                prompts = self._tokenizer.batch_decode(request.input_ids)
                print_sglang_outputs(prompts, results, self._tokenizer)

        # Stop and offload SGLang first before putting into channel
        # This avoids running SGLang and Megatron simultaneously
        self._stop()
        # Release the GPUs once the engine has offloaded
        output_channel.gpu_lock.release()
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        output_channel.put(rollout_result)


def all_floats_equal(float_list: list[float], epsilon: float = 1e-9) -> bool:
    if len(float_list) <= 1:
        return True
    return np.std(float_list) < epsilon


class AsyncSGLangWorker(SGLangWorker):
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        super().__init__(config, placement)
        self._current_request: RolloutRequest = None
        self._input_queue = asyncio.Queue[RolloutRequest]()
        # (req_input_token_ids, sglang_result)
        self._output_queue = asyncio.Queue[Tuple[int, List[int], Dict]]()

        # Queue for completed rollouts
        self._completed_queue = asyncio.Queue[RolloutResult]()
        self._completion_info = CompletionInfo()
        self._rollout_end_event = asyncio.Event()
        self._sync_weight_end_event = asyncio.Event()

        self._reward_model = MathRewardModel(scale=self._cfg.reward.reward_scale)
        assert self._rollout_batch_size is None, (
            "rollout_batch_size_per_gpu is not supported in AsyncSGLangWorker"
        )

        # Initialize meta_stats_collector for async operations
        self.collect_meta_stats = getattr(self._cfg.rollout, 'collect_meta_stats', False)
        if self.collect_meta_stats:
            async_stats_file = getattr(self._cfg.rollout, 'async_meta_stats_file', f"sglang_meta_stats_async_rank_{self._rank}.jsonl")
            self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
            self.async_batch_counter = 0

        self.use_auto_scheduler = (self._placement._placement_mode == PlacementMode.AUTO)
        if self.use_auto_scheduler:
            self.schedule_channel = self.connect_channel(get_scheduler_channel('rollout'))
            # warmup
            self.schedule_channel.put(None, async_op=False)
            self.schedule_channel.get(async_op=False)

            self.scheduler_request_queue = get_scheduler_request_queue(self._rank)
            self.scheduler_response_queue = get_scheduler_response_queue(self._rank)

            if self.collect_meta_stats:
                schedule_stats_file = getattr(self._cfg.rollout, 'schedule_meta_stats_file', f"sglang_meta_stats_rank_{self._rank}.jsonl")
                self.schedule_meta_stats_collector = MetaInfoStatsCollector(schedule_stats_file)
                self.schedule_batch_counter = 0

            # Initialize report progress
            self.progress_report_interval = 1.0  # seconds
            self.last_progress_report = time.time()

    async def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        if self._cfg.rollout.detokenize:
            outputs = await self._engine.async_generate(
                self._validate_prompts, self._validate_sampling_params
            )
            for prompt, output in zip(self._validate_prompts, outputs):
                print("===============================")
                print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            outputs = await self._engine.async_generate(
                input_ids=prompt_ids, sampling_params=self._validate_sampling_params
            )
            print_sglang_outputs(self._validate_prompts, outputs, self._tokenizer)
        print("===============================", flush=True)

    async def init_worker(self):
        self._init_engine()
        if self._cfg.rollout.validate_weight:
            await self._validate_weight_at_first()

    def _compute_reward_and_advantage(
        self, engine_results: List[Dict], answer: str
    ):
        answers = [answer] * len(engine_results)
        texts: List[str] = []
        for res in engine_results:
            if hasattr(res, "text"):
                texts.append(res["text"])
            else:
                texts.append(
                    self._tokenizer.decode(res["output_ids"], skip_special_tokens=True)
                )

        results = math_verify_call(texts, answers)
        rewards = [(1 if r else -1) * self._reward_model.scale for r in results]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)

        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        advantages = (rewards_tensor - mean) / (std + 1e-6)

        return rewards, advantages.tolist()

    async def _async_generate(
        self, raw_id: int, input_ids: List[int], sampling_params: dict
    ):
        result = await self._engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=self._return_logprobs,
        )

        if self._cfg.rollout.print_outputs:
            prompts = self._tokenizer.batch_decode(input_ids)
            print_sglang_outputs(prompts, [result], self._tokenizer)

        # SGLang does not return input_ids, so we need to pass them for further usage.
        return raw_id, input_ids, result

    async def _put_result(self, result: RolloutResult, output_channel: Channel):
        await output_channel.put(item=result, async_op=True).async_wait()

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        self.log_info("Starting async generation...")
        completion_info = CompletionInfo(self._logger)
        task_queue = AsyncTaskQueue()

        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        # self._current_request = rollout_request
        # self._completion_info.clear_and_set(rollout_request)

        with self.worker_timer():
            # rollout_tasks = [
            #     asyncio.create_task(
            #         self._async_generate(raw_id, input_ids, self._sampling_params)
            #     )
            #     for raw_id, input_ids in enumerate(rollout_request.input_ids)
            #     for _ in range(rollout_request.n)
            # ]
            completion_info.n_result_each_request = rollout_request.n

            for input_ids, answer in zip(rollout_request.input_ids, rollout_request.answers):
                unique_id = completion_info.add_unique_id(input_ids, answer)
                for _ in range(rollout_request.n):
                    task_queue.put_one(self._async_generate(completion_info, unique_id, input_ids, self._sampling_params))

            return_tasks = []
            all_results = []

            def post_process_result(result, rollout_result: RolloutResult):
                all_results.append(result)  # Collect for meta_info stats
                if rollout_result is not None:
                    return_tasks.append(
                        asyncio.create_task(
                            self._put_result(rollout_result, output_channel)
                        )
                    )
            loop_handle_rollout_tasks = asyncio.create_task(task_queue.loop_tasks(post_process_result))

            if self.use_auto_scheduler:
                scheduler_tasks = asyncio.create_task(self.run_with_scheduler(task_queue, completion_info))
                await asyncio.gather(loop_handle_rollout_tasks, scheduler_tasks)
            else:
                # loop will be finished if unfinished_len is 0
                task_queue.set_exit_flag()
                await asyncio.gather(loop_handle_rollout_tasks)

            # wait for send result
            await asyncio.gather(*return_tasks)

            # Collect meta_info statistics for all results only if enabled in config
            if self.collect_meta_stats and hasattr(self, 'async_meta_stats_collector'):
                self.async_meta_stats_collector.collect_batch_stats(all_results, self.async_batch_counter)
                self.async_batch_counter += 1

            self.log_info(f"Async generation and send completed. send results len={len(return_tasks)}")
            await self.offload_engine()
            await self.schedule_channel.put(RolloutScheduleInfo(instance_id=self._rank, action=RolloutAction.Offloaded), queue_name=self.scheduler_response_queue, async_op=True).async_wait()

    async def offload_engine(self):
        """
        Offload the model weights from the SGLang engine.
        """
        await self._engine.tokenizer_manager.offload_model_weights(
            io_struct.OffloadReqInput()
        )

    async def sync_model_from_actor(self):
        """Update the weights of the SGLang engine."""
        await self._engine.tokenizer_manager.sync_hf_weight(
            obj=io_struct.SyncHFWeightInput()
        )

    def shutdown(self):
        """
        Shutdown the SGLang task.
        """
        # Finalize meta_info statistics collectors if they exist
        if hasattr(self, 'async_meta_stats_collector'):
            self.async_meta_stats_collector.finalize()

        if hasattr(self, 'schedule_meta_stats_collector'):
            self.schedule_meta_stats_collector.finalize()

        self.log_info(f"Shutting down SGLang worker {self._rank} ...")
        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")
