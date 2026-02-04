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
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = self.cfg.actor.get(
            "enable_dp_load_balance", False
        )
        self.lr_sched_sync_with_optim = self.cfg.actor.get(
            "lr_sched_sync_with_optim", True
        )
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

        self.bucket_capacity = 128 * 1024 * 1024

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def divide_model_to_bucket(self, state_dict, has_visual):
        bucket_capacity = self.bucket_capacity
        model_bucket_list = []
        current_capacity = 0
        model_bucket = {}
        for key, val in state_dict.items():
            name = key
            if "_extra_state" in name:
                continue
            if has_visual:
                if name.startswith("model.language_model."):
                    name = "model." + name[21:]
                # NOTE:
                # if transformers version is 4.56.1 or older(not tested),
                # the following line should be uncommented

                # elif name.startswith("model."):
                #     name = name[6:]

            model_bucket[name] = val
            current_capacity += (
                val.numel() * val.element_size() * torch.distributed.get_world_size()
            )

            if current_capacity >= bucket_capacity:
                model_bucket_list.append(model_bucket)
                current_capacity = 0
                model_bucket = {}

        if len(model_bucket) > 0:
            model_bucket_list.append(model_bucket)
        return model_bucket_list

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())
        if self._weight_dst_rank_in_rollout is not None:
            rollout_dtype = None
            if self._cfg.get("sync_precision", None) is not None:
                rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)
            model_bucket_list = self.divide_model_to_bucket(
                self.rollout_state_dict, has_visual
            )
            self.log_debug(
                f"[sync_model_to_rollout rank-{self._rank}] length of model_bucket_list: {len(model_bucket_list)}"
            )
            for bucket_idx, model_bucket in enumerate(model_bucket_list):
                buffer = {}
                for k, v in model_bucket.items():
                    if isinstance(v, DTensor):
                        v = v.full_tensor()
                    if rollout_dtype is not None:
                        v = v.to(rollout_dtype)
                    if not self.is_pipeline:
                        v = reduce_tensor(v)
                    buffer[k] = v
                if bucket_idx == 0:
                    buffer["bucket_length"] = len(model_bucket_list)
                if not self.is_pipeline:
                    self.send(
                        buffer,
                        self._rollout_group_name,
                        self._weight_dst_rank_in_rollout,
                    )
                else:
                    for weight_dst_rank in self._weight_dst_rank_in_rollout:
                        self.send(
                            buffer,
                            self._rollout_group_name,
                            weight_dst_rank,
                        )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_info(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result, async_op=True)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
                loss_mask=loss_mask,
                task_type=self.task_type,
            )

            entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                            **kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
