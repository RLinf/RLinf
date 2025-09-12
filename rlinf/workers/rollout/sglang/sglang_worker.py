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
from typing import Dict, List, Tuple

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
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    print_sglang_outputs,
)
from toolkits.math_verifier.verify import MathRewardModel, math_verify_call


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

    async def _compute_reward_and_advantage(
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
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        self._current_request = rollout_request
        self._completion_info.clear_and_set(rollout_request)

        with self.worker_timer():
            rollout_tasks = [
                asyncio.create_task(
                    self._async_generate(raw_id, input_ids, self._sampling_params)
                )
                for raw_id, input_ids in enumerate(rollout_request.input_ids)
                for _ in range(rollout_request.n)
            ]
            return_tasks = []
            total_reqs = len(rollout_tasks)
            required_reqs = total_reqs // self._cfg.algorithm.max_num_gen_batches

            droped_reqs = 0
            finished_reqs = 0
            abort_flag = False

            for future in asyncio.as_completed(rollout_tasks):
                raw_id, input_ids, result = await future
                hash_id = self._completion_info.hash(input_ids)
                self._completion_info.record_result(input_ids, result)

                if self._completion_info.is_completed(hash_id):
                    results = self._completion_info.get_results(hash_id)
                    (
                        rewards,
                        advantages,
                    ) = await self._compute_reward_and_advantage(
                        results,
                        self._current_request.answers[raw_id],
                    )
                    if (
                        all_floats_equal(rewards)
                        and self._cfg.algorithm.get("max_num_gen_batches", 1) > 1
                    ):
                        if (total_reqs - droped_reqs) > required_reqs:
                            droped_reqs += rollout_request.n
                            continue

                    input_ids = [input_ids] * len(results)
                    rollout_result = RolloutResult.from_sglang_results(
                        results,
                        rollout_request.n,
                        input_ids,
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
                    if finished_reqs == required_reqs:
                        abort_flag = True
                        break

            if abort_flag:
                # abort all req (running and waiting)
                await self._engine.tokenizer_manager.pause_generation()

            await asyncio.gather(*return_tasks)

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
        self.log_info(f"Shutting down SGLang worker {self._rank} ...")
        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")
