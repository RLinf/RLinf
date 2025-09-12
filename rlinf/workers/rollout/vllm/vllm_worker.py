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

import os
from typing import List

from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.utils import print_vllm_outputs

from . import VLLMEngine


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
            )
        else:
            sampling_params = SamplingParams(
                temperature=cfg_sampling_params.temperature,
                top_k=cfg_sampling_params.top_k,
                top_p=cfg_sampling_params.top_p,
                repetition_penalty=cfg_sampling_params.repetition_penalty,
                max_tokens=cfg_sampling_params.max_new_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
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
        output_channel.gpu_lock.acquire()

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
        output_channel.gpu_lock.release()
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        output_channel.put(rollout_result)
