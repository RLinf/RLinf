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
import dataclasses
from typing import Any, Optional

from omegaconf import DictConfig
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
    SeqGroupInfo,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler.dynamic_scheduler.manager import RolloutScalingScheduler
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_scheduler_channel,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    MetaInfoStatsCollector,
    RolloutEngineStats,
    RunningStatusManager,
    print_sglang_outputs,
)


class SGLangWorker(Worker):
    def __init__(self, config: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)

        self._cfg = config
        self._placement = placement

        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.rollout.model_dir)
        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = SGLangWorker.get_sampling_param_from_config(self._cfg)

        self._validate_sampling_params = {"temperature": 0, "max_new_tokens": 32}
        self._validate_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        self.status_manager = RunningStatusManager()

        # Initialize meta_stats_collector for async operations
        self._collect_meta_stats = getattr(
            self._cfg.rollout, "collect_meta_stats", False
        )
        self._use_auto_scheduler = self._placement.is_auto

        if self._collect_meta_stats:
            self._init_meta_stats_collector()
        if self._use_auto_scheduler:
            self._init_scheduler()

    def _init_scheduler(self):
        self.schedule_channel = self.connect_channel(
            get_scheduler_channel("rollout", self._rank)
        )

        self._scheduler = RolloutScalingScheduler(
            self._rank, self.schedule_channel, self
        )

    def _init_meta_stats_collector(self):
        async_stats_file = getattr(
            self._cfg.rollout,
            "async_meta_stats_file",
            f"sglang_meta_stats_async_rank_{self._rank}.jsonl",
        )
        self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
        self.async_batch_counter = 0

    def _collect_stats(self, engine_results: list[dict]):
        self.async_meta_stats_collector.collect_batch_stats(
            engine_results, self.async_batch_counter
        )
        self.async_batch_counter += 1

    @staticmethod
    def get_sampling_param_from_config(cfg: DictConfig) -> dict:
        """
        Get sampling parameters from the configuration.
        """
        cfg_sampling_params = cfg.algorithm.sampling_params
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
            load_format="dummy"
            if not self._cfg.rollout.validate_weight
            and not getattr(self._cfg.rollout, "validate_weight_first_sync", False)
            else "auto",
            # disable_overlap_schedule=True,
            dtype=torch_dtype_from_precision(self._cfg.rollout.precision),
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
            **dataclasses.asdict(server_args),
        )

    def shutdown(self):
        """
        Shutdown the SGLang task.
        """
        # Finalize meta_info statistics collectors if they exist
        if self._collect_meta_stats:
            self.async_meta_stats_collector.finalize()

        self.log_info(f"Shutting down SGLang worker {self._rank} ...")
        self._engine.shutdown()
        self.log_info(f"SGLang worker {self._rank} shutdown complete.")

    async def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.
        """
        input_ids = self._tokenizer(self._validate_prompts).input_ids
        engine_results, _ = await self.async_generate(
            input_ids=input_ids,
            sampling_params=self._validate_sampling_params,
            return_logprob=False,
        )
        print_sglang_outputs(self._validate_prompts, engine_results, self._tokenizer)
        print("===============================", flush=True)

    async def async_generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        input_ids: list[list[int]] | list[int] | None = None,
        image_data: list | None = None,
        return_logprob: list[bool] | bool | None = False,
        request_info: Any | None = None,
    ):
        """
        Asynchronously generate text using the underlying SGLang engine and return
        the engine result together with the original input_ids, answers, and idx.

        This wrapper calls self._engine.async_generate(...) and forwards the provided
        arguments. Because the SGLang engine does not include the original input_ids
        in its response, this method returns the input_ids alongside the engine
        result for downstream use.

        Args:
            prompt (List[str] | str | None): Same as SGLang engine's prompt argument.
            sampling_params (List[Dict] | Dict | None): Same as SGLang engine's sampling_params argument.
            input_ids (List[List[int]] | List[int] | None): Same as SGLang engine's input_ids argument.
            return_logprob (List[bool] | bool | None): Same as SGLang engine's return_logprob argument.
            request_info (Any | None): Any additional request info you wish to be associated with this
                generation request. This argument will not be passed to the SGLang engine and returned directly.

        Returns:
            Tuple[Dict, Any | None]: A tuple containing the engine result and the original request_info.
        """
        result = await self._engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data
            if image_data is not None and any(image_data)
            else None,
            return_logprob=return_logprob,
        )
        return result, request_info

    async def init_worker(self):
        self._init_engine()
        await self._engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(
                method_name="init_rlinf_worker",
                args=(
                    self.worker_address,
                    self._placement,
                    self._cfg,
                ),
            )
        )
        self.log_info(f"SGLang worker {self._rank} initialized.")
        if self._cfg.rollout.validate_weight:
            await self._validate_weight_at_first()
        if self._placement.is_collocated:
            await self.offload_engine()
        if self._use_auto_scheduler:
            asyncio.create_task(self._scheduler.main_loop())

    async def offload_engine(self):
        """
        Offload the model weights from the SGLang engine.
        """
        await self._engine.tokenizer_manager.release_memory_occupation(
            obj=ReleaseMemoryOccupationReqInput()
        )

    async def abort_generation(self):
        """Abort the generation."""
        await self._engine.tokenizer_manager.abort_generation(
            obj=io_struct.AbortGenerationInput()
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
        state = RolloutEngineStats(**state)

        return state

    async def _async_generate_group(self, seq_group_info: SeqGroupInfo):
        """Generate a group of responses for a request (for GRPO-like behavior)."""
        if seq_group_info.num_aborted == 0:
            # No aborted sequences, repeat the input for group_size times
            assert seq_group_info.num_returned == 0
            seq_idx_list = list(range(seq_group_info.group_size))
            input_batch = [seq_group_info.input_ids] * seq_group_info.group_size
            sampling_params_list = [self._sampling_params] * seq_group_info.group_size
            image_data_list = [seq_group_info.image_data] * seq_group_info.group_size
        else:
            # Have aborted sequences (e.g., migrated from other engines)
            # Continue generation for the aborted group
            idx_aborted = seq_group_info.idx_aborted.copy()
            seq_idx_list: list[int] = []
            seq_group_info.idx_aborted.clear()
            input_batch: list[list[int]] = []
            sampling_params_list: list[dict] = []
            image_data_list: list = []
            for idx in idx_aborted:
                generated_ids: list[int] = seq_group_info.results[idx]["output_ids"]
                if len(generated_ids) >= self._sampling_params["max_new_tokens"]:
                    # avoid genererating for sequences that have already meet their max_new_tokens
                    self.log_warning(
                        f"SeqGroup {seq_group_info.id} idx {idx} "
                        f"has generated {len(generated_ids)} tokens, "
                        f"exceeding max_new_tokens={self._sampling_params['max_new_tokens']}, "
                        f"it will be truncatured."
                    )
                    result = seq_group_info.results[idx]
                    seq_group_info.results[idx] = None
                    result["meta_info"]["finish_reason"]["type"] = "length"
                    seq_group_info.record_sglang_result(idx, result)
                    continue
                seq_idx_list.append(idx)
                input_batch.append(seq_group_info.input_ids + generated_ids)
                params = self._sampling_params.copy()
                params["max_new_tokens"] -= len(generated_ids)
                sampling_params_list.append(params)
                image_data_list.append(seq_group_info.image_data)

        tasks = [
            asyncio.create_task(
                self.async_generate(
                    input_ids=input_ids,
                    image_data=image_data,
                    sampling_params=sampling_params,
                    return_logprob=self._return_logprobs,
                    request_info={
                        "seq_idx": seq_idx,
                    },
                )
            )
            for seq_idx, input_ids, sampling_params, image_data in zip(
                seq_idx_list,
                input_batch,
                sampling_params_list,
                image_data_list,
                strict=True,
            )
        ]
        for future in asyncio.as_completed(tasks):
            result, request_info = await future
            seq_group_info.record_sglang_result(
                request_info["seq_idx"], result, self._logger
            )

        return seq_group_info

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        self.log_on_first_rank("Start generation...")
        request: RolloutRequest = input_channel.get()
        groups = request.to_seq_group_infos()
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

            all_rollout_results = []
            while pending := self.status_manager.get_running_tasks():
                done, pending = await asyncio.wait(pending, return_when=async_wait_type)
                returned_seq_groups: list[SeqGroupInfo] = [
                    task.result() for task in done
                ]
                for group in returned_seq_groups:
                    if group.all_completed:
                        rollout_result = RolloutResult.from_sglang_seq_group(
                            group,
                            self._return_logprobs,
                        )
                        all_rollout_results.append(rollout_result)
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

            if self._collect_meta_stats:
                self._collect_stats(all_rollout_results)

            if self._placement.is_collocated or self._placement.is_auto:
                await self.offload_engine()
                if self._use_auto_scheduler:
                    await self._scheduler.report_offloaded()

    async def generate_and_send(
        self,
        output_channel: Channel,
        channel_key: str,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
    ):
        final_sampling_params = self._sampling_params
        if sampling_params is not None and len(sampling_params) > 0:
            final_sampling_params = copy.deepcopy(self._sampling_params)
            for key, value in sampling_params.items():
                final_sampling_params[key] = value

        result = await self._engine.async_generate(
            input_ids=prompt_ids,
            sampling_params=final_sampling_params,
            return_logprob=self._return_logprobs,
        )
        # sglang will trim matched stop in result text, so we should only return output_ids
        result_dict = {
            "output_ids": result["output_ids"],
            "finish_reason": result["meta_info"]["finish_reason"]["type"],
        }
        if self._return_logprobs:
            result_dict["logprobs"] = [
                item[0] for item in result["meta_info"]["output_token_logprobs"]
            ]
        await output_channel.put(
            result_dict, key=channel_key, async_op=True
        ).async_wait()

    async def rollout_serverless(self, input_channel: Channel, output_channel: Channel):
        while True:
            rollout_request = await input_channel.get(async_op=True).async_wait()
            asyncio.create_task(
                self.generate_and_send(
                    output_channel=output_channel,
                    channel_key=rollout_request["channel_key"],
                    prompt_ids=rollout_request["prompt_ids"],
                    sampling_params=rollout_request.get("sampling_params", None),
                )
            )
