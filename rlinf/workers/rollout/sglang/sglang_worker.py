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
from dataclasses import dataclass
from typing import Dict, List

import torch
from omegaconf import DictConfig
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    print_sglang_outputs,
)
from toolkits.math_verifier.verify import MathRewardModel, math_verify_call


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    max_running_reqs: int = 0
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0


class AsyncSGLangWorker(Worker):
    def __init__(self, config: DictConfig, placement: ModelParallelComponentPlacement):
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
        self._reward_model = MathRewardModel(scale=self._cfg.reward.reward_scale)

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

    def _pre_process_rollout_request(
        self, request: RolloutRequest
    ) -> List[List[RolloutRequest]]:
        group_size = request.n
        repeated_request = request.repeat()
        if self._rollout_batch_size is not None:
            assert len(repeated_request.input_ids) % self._rollout_batch_size == 0, (
                f"rollout_batch_size {self._rollout_batch_size} must divide the total number of requests {len(repeated_request)}"
            )
            num_batch = len(repeated_request.input_ids) // self._rollout_batch_size
        else:
            num_batch = 1

        # Split the repeated request into smaller requests based on the rollout batch size
        # avoid too large request that may cause KV cache OOM
        split_requests = repeated_request.split(num_batch)
        if self._placement.is_disaggregated:
            num_prompts_per_request = len(split_requests[0].input_ids) // group_size
            # for disaggregated mode, split to ensure each small request has full group_size prompts
            return [r.split(num_prompts_per_request) for r in split_requests]
        else:
            return [r.split(1) for r in split_requests]

    def shutdown(self):
        """
        Shutdown the SGLang task.
        """
        self.log_info(f"Shutting down SGLang worker {self._rank} ...")
        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")

    async def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        if self._cfg.rollout.detokenize:
            self.log_warning(
                "validate_weight with detokenize=True is not supported yet."
            )
        else:
            prompt_ids = self._tokenizer(self._validate_prompts).input_ids
            _, _, engine_results = await self._async_generate(
                prompt_ids, None, self._validate_sampling_params, False
            )
            print_sglang_outputs(
                self._validate_prompts, engine_results, self._tokenizer
            )
            print("===============================", flush=True)

    async def _stop(self):
        self.log_debug(
            f"[LLM dp {self._rank}] Received None input tokens, rollout end."
        )
        if not self._placement.is_disaggregated:
            await self.offload_engine()

    def _compute_reward_and_advantage(
        self, engine_results: List[Dict], answers: List[List[str]]
    ):
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
        self,
        input_ids: List[List[int]],
        answers: List[List[str]],
        sampling_params: dict,
        return_logprobs: bool,
    ):
        result = await self._engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprobs,
        )
        # SGLang does not return input_ids, so we need to pass them for further usage.
        return input_ids, answers, result

    async def init_worker(self):
        self._init_engine()
        self.log_info(f"SGLang worker {self._rank} initialized.")
        if self._cfg.rollout.validate_weight:
            await self._validate_weight_at_first()
        if not self._placement.is_disaggregated:
            await self.offload_engine()

    async def offload_engine(self):
        """
        Offload the model weights from the SGLang engine.
        """
        await self._engine.tokenizer_manager.release_memory_occupation(
            obj=ReleaseMemoryOccupationReqInput()
        )

    async def sync_model_from_actor(self):
        """Update the weights of the SGLang engine."""
        await self._engine.tokenizer_manager.sync_hf_weight(
            obj=io_struct.SyncHFWeightInput()
        )

    async def check_running_state(self):
        state = await self._engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(method_name="get_scheduler_running_state")
        )
        state = SchedulerStats(**state)

        self.log_info(f"SGLang scheduler running state: {state}")

        if state.num_queue_reqs == 0 and state.num_running_reqs < state.max_running_reqs:
            # no pending requests in the queue and the running requests is not full
            return True
        return False

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        request: RolloutRequest = input_channel.get()
        output_channel.gpu_lock.acquire()
        # Repeat prompts based on the group_size config
        requests = self._pre_process_rollout_request(request)

        self.log_info(
            f"Received {len(request.input_ids)} prompts, group_size = {request.n}, "
            f"total num_req = {len(request.input_ids) * request.n}. "
            f"Split to {len(requests)} batches, each has {len(requests[0])} group with {len(requests[0][0].input_ids)} sequences."
        )

        with self.worker_timer():
            # for collocated mode, len(requests) == 1. for disaggregated mode, len(requests) == num_group in smaller requests
            for request_groups in requests:
                tasks = [
                    asyncio.create_task(
                        self._async_generate(
                            group.input_ids,
                            group.answers,
                            self._sampling_params,
                            self._return_logprobs,
                        )
                    )
                    for group in request_groups
                ]

                # Enhanced as_completed: support dynamically adding new tasks
                pending = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for future in done:
                        input_ids, answers, engine_results = await future
                        rollout_result = RolloutResult.from_sglang_results(
                            engine_results,
                            request.n,
                            input_ids,
                            answers,
                            self._return_logprobs,
                        )
                        if self._placement.is_disaggregated:
                            (
                                rewards,
                                advantages,
                            ) = await asyncio.to_thread(
                                self._compute_reward_and_advantage,
                                engine_results,
                                answers,
                            )

                            rollout_result.rewards = torch.tensor(
                                rewards, dtype=torch.float32
                            ).reshape(-1, 1)
                            rollout_result.advantages = advantages

                        await output_channel.put(
                            item=rollout_result, async_op=True
                        ).async_wait()

                        # If you want to add new tasks dynamically, do it here:
                        # new_task = asyncio.create_task(...)
                        # pending.add(new_task)
                        if await self.check_running_state():
                            pass

        await self._stop()
        output_channel.gpu_lock.release()
