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
from typing import Any, Literal, Optional

from omegaconf import DictConfig
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.schema.reasoning_requests import RolloutRequest, SeqGroupInfo
from rlinf.data.schema.reasoning_results import RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.dynamic_scheduler.manager import RolloutScalingScheduler
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_scheduler_channel,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.backend import RlinfContext
from rlinf.workers.rollout.sglang.backends import SGLangEngineBackend, make_backend
from rlinf.workers.rollout.sglang_server.rendezvous import (
    broadcast_instance_readiness,
    negotiate_model_instance_dist_addr,
    wait_for_instance_readiness,
)
from rlinf.workers.rollout.utils import (
    MetaInfoStatsCollector,
    RunningStatusManager,
    print_sglang_outputs,
)

_SERVER_ARGS_FIELDS = {f.name for f in dataclasses.fields(ServerArgs)}


class SGLangWorker(Worker):
    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload: Literal["sync", "cpu", None] = "sync",
        config_rollout: DictConfig = None,
    ):
        Worker.__init__(self)

        self._cfg = config
        # 'sync': sync weight from actor;
        # 'cpu': save weight from cpu and reload weight from cpu
        # None: no need to reload, only used in eval
        self.weight_reload = weight_reload
        if config_rollout is None:
            config_rollout = self._cfg.rollout
        self._cfg_rollout = config_rollout
        self._placement = placement
        # Whether this process is the entry (node rank 0) of its model
        # instance. Only the entry consumes channels and drives the HTTP
        # control plane; the other ranks of a cross-node instance just
        # keep their sglang server subprocess alive.
        # Degenerates to True on single-node layouts, where every process
        # is its instance's entry.
        self._is_model_instance_entry = (
            placement.rollout_node_rank_in_model_instance(self._rank) == 0
        )
        # Set in init_worker(); declared here so RPC methods can check it.
        self._backend = None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg_rollout.model.model_path,
            trust_remote_code=self._cfg_rollout.model.trust_remote_code,
        )
        self._return_logprobs = self._cfg_rollout.return_logprobs
        sampling_params = None
        if config_rollout is not None:
            sampling_params = config_rollout.get("sampling_params", None)
        if sampling_params is None:
            sampling_params = self._cfg.algorithm.sampling_params
        self._sampling_params = SGLangWorker.get_sampling_param_from_config(
            sampling_params
        )

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
            self._cfg_rollout, "collect_meta_stats", False
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
            self._cfg_rollout,
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
    def get_sampling_param_from_config(cfg_sampling_params: DictConfig) -> dict:
        """
        Get sampling parameters from the configuration.
        """
        if not cfg_sampling_params.do_sample:
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

    def _build_server_args(self) -> dict:
        """Build the shared sglang ServerArgs kwargs for the rollout backend.

        Engine and Server backends consume the same product, so
        version-compatibility branches live here only once.
        """
        use_cudagraph = not self._cfg_rollout.enforce_eager

        load_format = "dummy"  # dummy means randomize init weight
        if self.weight_reload == "sync":
            validate_weight_first_sync = self._cfg_rollout.get(
                "validate_weight_first_sync", False
            )
            if self._cfg.runner.resume_dir is not None:
                # validate_weight_first_sync compare hf weights with megatron weights,
                # and if resume_dir is enabled, hf weights can't equal to megatron's.
                validate_weight_first_sync = False
            if self._cfg_rollout.validate_weight or validate_weight_first_sync:
                load_format = "auto"
        else:
            load_format = "auto"

        # moe_dp_size / moe_a2a_backend / enable_deterministic_inference only
        # exist on newer sglang; passing a keyword the installed version does
        # not define is a TypeError even when the value is the default, so
        # only forward the ones it has. enable_deterministic_inference
        # gives token-level determinism for the Engine/Server EXACT
        # comparison pair; default False matches sglang's own default, so
        # configs that never set it are unaffected.
        version_dependent_args = {
            "moe_dp_size": self._cfg_rollout.sglang.get("moe_dp_size", 1),
            "moe_a2a_backend": self._cfg_rollout.sglang.get("moe_a2a_backend", None),
            "enable_deterministic_inference": self._cfg_rollout.sglang.get(
                "enable_deterministic_inference", False
            ),
        }
        for name in list(version_dependent_args):
            if name in _SERVER_ARGS_FIELDS:
                continue
            value = version_dependent_args.pop(name)
            if name in self._cfg_rollout.sglang:
                self.log_warning(
                    f"sglang ServerArgs has no field {name!r}; "
                    f"the configured value {value!r} is ignored"
                )

        server_args = ServerArgs(
            model_path=self._cfg_rollout.model.model_path,
            disable_cuda_graph=not use_cudagraph,
            cuda_graph_max_bs=min(
                self._cfg_rollout.cuda_graph_max_bs,
                self._cfg_rollout.max_running_requests,
            ),
            tp_size=self._cfg_rollout.tensor_parallel_size,
            # Expert parallel: pass ep_size explicitly. sglang derives
            # moe_tp = tp // ep // moe_dp itself. ep_size defaults to
            # tp when only enable_ep_moe is set, which is pure EP.
            ep_size=self._cfg_rollout.sglang.get(
                "ep_size",
                self._cfg_rollout.tensor_parallel_size
                if self._cfg_rollout.sglang.get("enable_ep_moe", False)
                else 1,
            ),
            # DP-attention decouples attention TP from MoE TP: dp_size shards
            # attention (attn_tp = tp // dp // attn_cp), while ep_size and
            # moe_tp_size act on the experts, and moe_dense_tp_size on the
            # dense MLP layers. All fall back to sglang's own defaults, so
            # non-DPA configs are unaffected.
            dp_size=self._cfg_rollout.sglang.get("dp_size", 1),
            enable_dp_attention=self._cfg_rollout.sglang.get(
                "enable_dp_attention", False
            ),
            enable_dp_lm_head=self._cfg_rollout.sglang.get("enable_dp_lm_head", False),
            moe_dense_tp_size=self._cfg_rollout.sglang.get("moe_dense_tp_size", None),
            mem_fraction_static=self._cfg_rollout.gpu_memory_utilization,
            enable_memory_saver=use_cudagraph,
            enable_torch_compile=self._cfg_rollout.sglang.use_torch_compile,
            torch_compile_max_bs=min(
                self._cfg_rollout.sglang.torch_compile_max_bs,
                self._cfg_rollout.max_running_requests,
            ),
            load_format=load_format,
            # disable_overlap_schedule=True,
            dtype=torch_dtype_from_precision(self._cfg_rollout.model.precision),
            # sglang will only return text/output_ids when skip_tokenizer_init=False/True
            # text is not needed in RL training, so set to True can save time.
            skip_tokenizer_init=not self._cfg_rollout.detokenize,
            # sglang will print statistics every decode_log_interval decode steps.
            decode_log_interval=self._cfg_rollout.sglang.decode_log_interval,
            attention_backend=self._cfg_rollout.sglang.attention_backend,
            log_level="info",
            max_running_requests=self._cfg_rollout.max_running_requests,
            dist_init_addr=f"127.0.0.1:{str(self.acquire_free_port())}",
            tool_call_parser=self._cfg_rollout.sglang.get("tool_call_parser", None),
            trust_remote_code=self._cfg_rollout.model.trust_remote_code,
            **version_dependent_args,
        )

        if self._cfg_rollout.sglang.get("backend_type", "engine") == "server":
            # Server-only: skip sglang's HTTP warmup. launch_server fires a
            # deterministic batched greedy request before serving, which the
            # Engine backend never runs; it leaves radix-cache/allocator state
            # that makes the two backends non-comparable. RLinf does its own
            # readiness check (_wait_for_http_health), so the warmup is
            # redundant.
            server_args.skip_server_warmup = True

        self.log_on_first_rank(f"{server_args=}")
        return dataclasses.asdict(server_args)

    @property
    def _engine(self):
        """Compat property: only available with the Engine backend.

        Kept for `SGLangAgentWorkerWithHTTPServer` (the serving_mode=
        worker_http path). TODO: remove this compat property once the
        worker_http path has test coverage.
        """
        if not isinstance(self._backend, SGLangEngineBackend):
            raise RuntimeError(
                "self._engine is only available with "
                "rollout.sglang.backend_type=engine; serving_mode=worker_http "
                "does not support backend_type=server"
            )
        return self._backend.engine

    def stop(self):
        """Stop the SGLang engine and finalize stats collectors."""
        # Finalize meta_info statistics collectors if they exist
        if self._collect_meta_stats:
            self.async_meta_stats_collector.finalize()

        self.log_info(f"Stopping SGLang worker {self._rank} ...")
        if self._backend is not None:
            self._backend.shutdown()
        self.log_info(f"SGLang worker {self._rank} stopped.")

    async def _validate_weight_at_first(self):
        """
        Run a test prompt batch and print its output.

        Non-entry processes of a cross-node instance skip this: the
        entry's generate already runs on every node's TP ranks, so a
        second validation from here would be redundant (and has no HTTP
        client to run through).
        """
        if not self._is_model_instance_entry:
            return
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

        This wrapper calls the rollout backend's async_generate(...) and forwards
        the provided arguments. Because the backend does not include the original
        input_ids in its response, this method returns the request_info alongside
        the engine result for downstream use.

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
        result = await self._backend.async_generate(
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
        assert self.weight_reload in ("sync", "cpu", None), (
            f"weight_reload should be in ['sync', 'cpu', None], but now it's {self.weight_reload}"
        )
        backend_type = self._cfg_rollout.sglang.get("backend_type", "engine")
        model_instance_id = self._placement.rollout_model_instance_id(self._rank)
        # Multi-node server mode: every rank of a model instance must launch
        # its sglang subprocess with an identical dist_init_addr pointing at
        # the entry node, so the ranks rendezvous (one broadcast) before the
        # backend spawns anything. Single-node layouts inject nothing:
        # no rendezvous, no multi-node kwargs.
        server_multi_node_kwargs = {}
        multi_node_server = False
        if backend_type == "server":
            nnodes = self._placement.rollout_nnodes_per_model_instance
            if nnodes > 1:
                multi_node_server = True
                node_rank = self._placement.rollout_node_rank_in_model_instance(
                    self._rank
                )
                # Rendezvous derives instance id / entry rank itself from
                # the placement (single source of truth for the layout
                # formulas); node_rank and nnodes are re-derived here only
                # because the server subprocess also needs them as flags.
                host, port = await negotiate_model_instance_dist_addr(
                    self, self._placement
                )
                server_multi_node_kwargs = {
                    "dist_init_addr": f"{host}:{port}",
                    "nnodes": nnodes,
                    "node_rank": node_rank,
                }
        self._backend = make_backend(
            backend_type,
            server_args=self._build_server_args(),
            rlinf_ctx=RlinfContext(
                parent_address=self.worker_address,
                weight_reload=self.weight_reload,
                placement=self._placement,
                cfg=self._cfg,
                # One model instance spans nnodes rollout workers; every
                # rank of the instance maps to the same id (degenerates to
                # the worker rank on single-node).
                model_instance_id=model_instance_id,
            ),
            acquire_free_port=self.acquire_free_port,
            log_info=self.log_info,
            log_error=self.log_error,
            **server_multi_node_kwargs,
        )
        if multi_node_server:
            # The entry broadcasts readiness over the
            # instance-internal group after its /health + registration
            # pass; non-entry ranks keep watching their own subprocess
            # instead of returning after the 2s spawn check.
            await self._initialize_multi_node_server_backend()
        else:
            await self._backend.initialize()
        self.log_info(f"SGLang worker {self._rank} initialized.")
        if self.weight_reload == "sync":
            if self._cfg_rollout.validate_weight:
                await self._validate_weight_at_first()
            if self._placement.is_collocated:
                await self.offload_engine()
            if self._use_auto_scheduler:
                asyncio.create_task(self._scheduler.main_loop())
        elif self.weight_reload == "cpu":
            if self._placement.is_collocated:
                await self.offload_engine()

    async def _initialize_multi_node_server_backend(self) -> None:
        """Server-backend init with instance-wide readiness signaling.

        Entry: on failure, broadcast ``("failed", reason)``
        best-effort (a non-entry rank may already be gone - never let the
        signal mask the original error) and re-raise; on success, broadcast
        ``("ready",)``. Non-entry: spawn + 2s check (inside
        ``initialize()``), then wait for the entry's signal while polling
        own subprocess liveness.
        """
        if self._is_model_instance_entry:
            try:
                await self._backend.initialize()
            except Exception as e:
                try:
                    await broadcast_instance_readiness(
                        self,
                        self._placement,
                        ready=False,
                        reason=repr(e),
                        timeout_s=30.0,
                    )
                except Exception:
                    self.log_warning(
                        "Failed to signal 'failed' to the non-entry ranks of "
                        "this model instance; they will hit their readiness "
                        "timeout instead."
                    )
                raise
            await broadcast_instance_readiness(
                self, self._placement, ready=True, timeout_s=300.0
            )
            return
        await self._backend.initialize()
        await wait_for_instance_readiness(
            self,
            self._placement,
            child_alive=self._backend.is_server_process_alive,
            poll_child_failure=self._backend.poll_server_process_failure,
        )

    async def offload_engine(self):
        """
        Release the model weights from the SGLang engine.

        Non-entry process of a cross-node instance: no-op - the entry's
        offload request reaches this node's schedulers through sglang's
        request broadcast.
        """
        if not self._is_model_instance_entry:
            return
        assert self.weight_reload is not None
        await self._backend.offload()

    async def onload_engine(self):
        """
        Onload the model weights from cpu to the SGLang engine.
        """
        if not self._is_model_instance_entry:
            return
        assert self.weight_reload == "cpu"
        await self._backend.onload()

    async def onload_kv_cudagraph(self):
        """
        Onload only the KV cache and CUDA graph back to GPU, leaving model
        weights on GPU (already resumed in sync_hf_weight). Used in collocated
        'sync' mode where resume_memory_occupation is split: weights resumed
        before load_weights (needs ~10GB), KV+cuda graph deferred until after
        the actor has offloaded its model (avoids both models on GPU
        simultaneously).
        """
        if not self._is_model_instance_entry:
            return
        await self._backend.onload(tags=["kv_cache", "cuda_graph"])

    async def abort_generation(self):
        """Abort the generation."""
        if not self._is_model_instance_entry:
            return
        await self._backend.abort_generation()

    async def sync_model_from_actor(self):
        """Update the weights of the SGLang engine."""
        if not self._is_model_instance_entry:
            return
        await self._backend.sync_weights()

    async def check_running_state(self):
        if not self._is_model_instance_entry:
            return None
        state = await self._backend.get_running_state()

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
        # Non-entry process of a cross-node instance: neither get nor put -
        # the entry process of the instance services the whole channel.
        # The timer must still be recorded, explicitly under the same key
        # as the entry's: the runner's consume_duration() pops "rollout"
        # on every process of the group and a missing key raises
        # ValueError. The near-zero duration is invisible under
        # the default max reduction.
        if not self._is_model_instance_entry:
            with self.worker_timer("rollout"):
                return
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

            if self.weight_reload is not None and (
                self._placement.is_collocated or self._placement.is_auto
            ):
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
        # Second line of defense behind rollout_serverless' entry gate:
        # this is a fire-and-forget task, so an
        # exception here would be swallowed - return instead.
        if not self._is_model_instance_entry:
            return
        final_sampling_params = self._sampling_params
        if sampling_params is not None and len(sampling_params) > 0:
            final_sampling_params = copy.deepcopy(self._sampling_params)
            for key, value in sampling_params.items():
                final_sampling_params[key] = value

        result = await self._backend.async_generate(
            input_ids=prompt_ids,
            sampling_params=final_sampling_params,
            image_data=None,
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
        # The gate must precede the first input_channel.get(): the runner
        # calls this on the whole worker group and every worker races on
        # the same queue, so a non-entry consumer would steal requests and
        # drop them (generate_and_send early-returns without writing the
        # output channel).
        if not self._is_model_instance_entry:
            return
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
