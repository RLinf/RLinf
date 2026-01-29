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

import logging
from importlib.metadata import version
from typing import Any, Literal

import torch
from omegaconf import DictConfig
from packaging.version import parse
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler as _Scheduler
from sglang.srt.managers.scheduler import logger
from sglang.srt.managers.scheduler import (
    run_scheduler_process as _run_scheduler_process,
)

from rlinf.scheduler import Worker, WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import (
    RankMapper,
)

from .io_struct import (
    AbortGenerationInput,
    AbortGenerationOutput,
    SyncHFWeightInput,
    SyncHFWeightOutput,
    TaskMethodInput,
    TaskMethodOutput,
)

logger.setLevel(logging.WARNING)


class Scheduler(_Scheduler):
    """
    Overridden class of SGLang's TP worker class _Scheduler.
    A Scheduler is a Task that manages the TP worker, and performs necessary weight synchronization with actor and weight offloading.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `TpModelWorkerClient` is used when ServerArgs.enable_overlap=True, and it has 'worker' attribute.
        # But in early SGLang version, `TpModelWorker` doesn't have 'worker' attribute.
        if not hasattr(self.tp_worker, "worker"):
            self.tp_worker.worker = self.tp_worker

        self._request_dispatcher._mapping.extend(
            [
                (TaskMethodInput, self.run_task_method),
                (SyncHFWeightInput, self.sync_hf_weight),
                (AbortGenerationInput, self.abort_generation),
            ]
        )

        self.is_weight_offloaded = False
        self.weight_norm_dict = None

        try:
            sglang_version = parse(version("sglang"))
        except Exception as e:
            raise ValueError(f"sglang version not supported: {e}")
        self.patch_return_output_ids = sglang_version < parse("0.5.0")

    def cuda_info(self, text: str = ""):
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        free_gpu_memory /= 2**30
        total_gpu_memory /= 2**30

        memory_allocated = torch.cuda.memory_allocated() / 2**30
        memory_reserved = torch.cuda.memory_reserved() / 2**30

        self._rlinf_worker.log_info(
            f"[dp {self._rlinf_worker.get_parent_rank()}-tp {self.tp_rank}] {text} "
            f"{memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB, "
            f"{free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB"
        )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_weight_offloaded is False, "Weight has been offloaded!"
        self.is_weight_offloaded = True
        return super().release_memory_occupation(recv_req)

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        assert self.is_weight_offloaded is True, "Weight has been onloaded!"
        self.is_weight_offloaded = False
        result = super().resume_memory_occupation(recv_req)
        if self.weight_reload == "cpu":
            model = self.tp_worker.worker.model_runner.model
            model.load_state_dict(self.cpu_state_dict)

        return result

    def batch_load_hf_weight(self, state_dict: dict[str, Any]) -> Any:
        assert self.weight_reload == "sync", (
            "only sglang with 'sync' can run 'batch_load_hf_weight'"
        )
        model = self.tp_worker.worker.model_runner.model
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        batch_weight = []
        if colocate:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)
                batch_weight.append((name, new_weight))
        else:
            # disaggregate mode, recv tensor directly
            for name, tensor in state_dict.items():
                batch_weight.append((name, tensor))

        model.load_weights(batch_weight)

        for name, weight in batch_weight:
            del weight
        batch_weight.clear()

    def sync_hf_weight(self, recv_req: SyncHFWeightInput):
        assert self.weight_reload == "sync", (
            "only sglang with 'sync' can run 'sync_hf_weight'"
        )
        use_cudagraph = not self.cfg.rollout.enforce_eager
        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )

        bucket_length = state_dict.get("bucket_length", None)
        if bucket_length is None:
            # recv from the Sglang backend
            # fsdp just send a bucket and don't have the key bucket_length
            bucket_length = 1
        else:
            # recv from the Megatron backend
            # Megatron use weight bucket to sync weight, the bucket length in dict of bucket 0, bucket_length
            state_dict.pop("bucket_length")
            assert bucket_length > 0, f"bucket_length {bucket_length} is invalid"

        if self.is_weight_offloaded:
            self.resume_memory_occupation(ResumeMemoryOccupationReqInput())

        self.batch_load_hf_weight(state_dict)
        if bucket_length > 1:
            recv_handle = self._rlinf_worker.recv(
                src_group_name=self._actor_group_name,
                src_rank=self.actor_weight_rank,
                async_op=True,
            )
            for _ in range(bucket_length - 2):
                next_recv_handle = self._rlinf_worker.recv(
                    src_group_name=self._actor_group_name,
                    src_rank=self.actor_weight_rank,
                    async_op=True,
                )
                state_dict = recv_handle.wait()
                self.batch_load_hf_weight(state_dict)
                recv_handle = next_recv_handle

            state_dict = recv_handle.wait()
            self.batch_load_hf_weight(state_dict)
        state_dict = None

        if self.weight_norm_dict is not None:
            # validate the weight norm dict between load model and first sync.
            model = self.tp_worker.worker.model_runner.model
            diff_keys = validate_weight_diff(model, self.weight_norm_dict)
            if len(diff_keys) != 0:
                raise RuntimeError(
                    f"sglang: validate_weight failed in first sync. diff_keys = {diff_keys}"
                )
            else:
                self._rlinf_worker.log_info(
                    f"sglang: validate_weight success at rank {self._rlinf_worker.get_parent_rank()}"
                )
            self.weight_norm_dict = None

        self.flush_cache()
        return SyncHFWeightOutput()

    def run_task_method(self, obj: TaskMethodInput):
        """
        Run a CommTask method with the given name and arguments.
        NOTE: will call wait() if async_op is True.
        """
        result = getattr(self, obj.method_name)(*obj.args, **obj.kwargs)
        if "async_op" in obj.kwargs and obj.kwargs["async_op"]:
            result = result.wait()
        return TaskMethodOutput(method_name=obj.method_name, result=result)

    def abort_generation(self, recv_req: AbortGenerationInput):
        # clear waiting reqs
        waiting_reqs = []
        # waiting_reqs.append(self.waiting_queue)
        for req in self.waiting_queue:
            req.to_abort = True

        # abort every running req with no kvcache
        running_reqs = []
        running_reqs.append(self.running_batch.reqs)
        for req in self.running_batch.reqs:
            req.to_abort = True
        res = AbortGenerationOutput(
            waiting_reqs=waiting_reqs, running_reqs=running_reqs
        )
        return res

    def init_rlinf_worker(
        self,
        parent_address: WorkerAddress,
        weight_reload: Literal["sync", "cpu", None] = "sync",
        placement: ModelParallelComponentPlacement = None,
        config: DictConfig = None,
    ):
        # WARNNING(wyq): Is world_size == self.tp_size when we enable EP in MoE?
        self._rlinf_worker = Worker(
            parent_address=parent_address, world_size=self.tp_size, rank=self.tp_rank
        )
        self.weight_reload = weight_reload
        if weight_reload == "sync":
            self.cfg = config
            self._actor_group_name = self.cfg.actor.group_name
            self.placement_mode = placement.placement_mode
            self.actor_weight_rank = RankMapper.get_rollout_rank_to_actor_rank_map(
                placement
            )[(self._rlinf_worker.get_parent_rank(), self._rlinf_worker._rank)]

            use_presharded_weights = (
                False if self.cfg.actor.training_backend == "fsdp" else True
            )
            model = self.tp_worker.worker.model_runner.model
            # it's important to use load_weight to load resharded weight from megatron
            for _, module in model.named_modules():
                if hasattr(module, "use_presharded_weights"):
                    module.use_presharded_weights = use_presharded_weights

            if self.cfg.rollout.get("validate_weight_first_sync", False):
                self.weight_norm_dict = validate_weight_init(model)

            self._rlinf_worker.log_info(
                f"Running Scheduler dp rank {self._rlinf_worker.get_parent_rank()}, tp rank {self.tp_rank}, corresponding actor weight rank = {self.actor_weight_rank}"
            )
        elif weight_reload == "cpu":
            # save state dict to cpu
            model = self.tp_worker.worker.model_runner.model
            cpu_state_dict = {}
            for key, value in model.state_dict().items():
                cpu_state_dict[key] = value.to("cpu", non_blocking=True)
            self.cpu_state_dict = cpu_state_dict
            torch.cuda.synchronize()

            self._rlinf_worker.log_info(
                f"Running Scheduler dp rank {self._rlinf_worker.get_parent_rank()}, tp rank {self.tp_rank}, load weight from cpu"
            )
        elif weight_reload is None:
            # save state dict to cpu
            self._rlinf_worker.log_info(
                f"Running Scheduler dp rank {self._rlinf_worker.get_parent_rank()}, tp rank {self.tp_rank}, no sync weight"
            )

    def get_scheduler_running_state(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )
        num_running_reqs = len(self.running_batch.reqs)
        return {
            "num_running_reqs": num_running_reqs,
            "max_running_reqs": self.max_running_requests,
            "num_used_tokens": num_used,
            "max_total_num_tokens": self.max_total_num_tokens,
            "token_usage": num_used / self.max_total_num_tokens,
            "num_queue_reqs": len(self.waiting_queue),
        }

    # to return output_ids and response_text simaltaneously in sglang 0.4.x.
    # copied from srt/managers/scheduler.py (0.4.6) and only delete the condition outside the assignment of "output_ids"
    def stream_output_generation(
        self,
        reqs,
        return_logprob: bool,
        skip_req=None,
    ):
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            DEFAULT_FORCE_STREAM_INTERVAL,
            BaseFinishReason,
            BatchTokenIDOut,
            DisaggregationMode,
        )

        # for sglang 0.5.0 and later, we use the original _handle_batch_output
        if not self.patch_return_output_ids:
            return super().stream_output_generation(reqs, return_logprob, skip_req)

        rids = []
        finished_reasons: list[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )
                    should_output = len(req.output_ids) % stream_interval == 0
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        and not self.model_config.is_multimodal_gen
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                # ----- patched code start -----
                output_ids.append(req.output_ids[send_token_offset:])
                # -----  patched code end  -----
                req.send_token_offset = len(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)

            if (
                req.finished()
                and self.tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return

            self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids,
                    finished_reasons,
                    decoded_texts,
                    decode_ids_list,
                    read_offsets,
                    output_ids,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                    no_stop_trim,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_ct,
                    input_token_logprobs_val,
                    input_token_logprobs_idx,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx,
                    output_hidden_states,
                )
            )


def posi_norm(tensor: torch.Tensor):
    # use a position-aware norm to do validate. otherwise the misalignment cannot be detect.
    posi_tensor = (
        torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device).view_as(
            tensor
        )
        / tensor.numel()
        - 0.5
    )
    return (tensor - posi_tensor).norm()


def validate_weight_init(model):
    weight_norm_dict = {}

    for key, value in model.state_dict().items():
        weight_norm_dict[key] = posi_norm(value)

    # avoid release memory before norm kernel launch (gpu is async from cpu)
    torch.cuda.synchronize()
    return weight_norm_dict


def validate_weight_diff(model, weight_norm_dict):
    weight_norm_dict_sync = {}
    diff_keys = []

    for name, value in model.state_dict().items():
        weight_norm_dict_sync[name] = posi_norm(value)

    for k in weight_norm_dict_sync.keys():
        if not torch.allclose(
            weight_norm_dict_sync[k],
            weight_norm_dict[k],
            rtol=1e-3,
            atol=1e-4,
        ):
            diff_keys.append(k)

    return diff_keys


def run_scheduler_process(*args, **kwargs):
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "sglang.srt.managers.scheduler.Scheduler",
        "rlinf.hybrid_engines.sglang.common.sgl_scheduler.Scheduler",
    )
    Patcher.apply()
    _run_scheduler_process(*args, **kwargs)
