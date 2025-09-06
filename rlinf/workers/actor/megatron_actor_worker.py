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

import copy
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from torch.multiprocessing.reductions import reduce_tensor

from rlinf.algorithms.math.algo_functions import (
    actor_loss_fn,
    calculate_adv_and_returns,
    kl_penalty,
)
from rlinf.algorithms.math.verifier.verify import math_verify_call
from rlinf.data.io_struct import (
    BatchResizingIterator,
    RolloutResult,
    get_batch_size,
    get_seq_length,
)
from rlinf.hybrid_engines.megatron.megatron_model_manager import (
    MegatronModelManager,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_last_rank,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    broadcast_tensor_within_mp,
    broadcast_tensor_within_pp,
    compute_rollout_metrics,
    masked_normalization,
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.profiler import PyTorchProfiler, PyTorchProfilerFunc
from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
from rlinf.utils.resharding.reshard_config import ReshardConfig
from rlinf.utils.train_utils import (
    set_eval,
    set_sync_funcs,
    set_train,
)
from rlinf.utils.utils import (
    clear_memory,
    configure_batch_sizes,
    cpu_dict,
    cpu_weight_swap,
    masked_mean,
    retrieve_model_state_dict_in_cpu,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import RankMapper


class MegatronActor(MegatronModelManager, Worker):
    """The class for running the actor training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role="actor"
    ):
        """Initialize the MegatronActor.

        Args:
            cfg (DictConfig): The configuration for the actor.
        """
        Worker.__init__(self)
        role_cfg = getattr(cfg, role, None)
        self.role_cfg = role_cfg
        if role_cfg is None:
            raise ValueError(f"Role {role} is not defined in the configuration.")
        super().__init__(role_cfg)
        self.cfg = cfg
        self.component_placement = placement

        # Data configurations
        self.response_len = (
            role_cfg.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.average_response_len = self.response_len

        # Algo configurations
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.ratio_eps = self.cfg.algorithm.ratio_clip_eps
        self.logprob_forward_micro_batch_size = (
            self.cfg.algorithm.logprob_forward_micro_batch_size
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type
        self.clip_ratio_c = self.cfg.algorithm.clip_ratio_c
        if self.cfg.algorithm.loss_agg_func == "token-mean":
            self.loss_agg_func = masked_mean
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-sum":
            self.loss_agg_func = seq_mean_token_sum
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-mean":
            self.loss_agg_func = seq_mean_token_mean
        else:
            raise NotImplementedError(
                f"algorithm.loss_agg_func={self.cfg.algorithm.loss_agg_func} is not supported!"
            )

        # Actor configurations
        self.enable_dynamic_batch_size = self.cfg.runner.enable_dynamic_batch_size
        self.max_tokens_per_mbs = self.cfg.runner.max_tokens_per_mbs
        self.offload_optimizer = self.cfg.actor.offload_optimizer
        self.offload_weight = self.cfg.actor.offload_weight
        self.offload_grad = self.cfg.actor.offload_grad
        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False
        self.ref_policy_state_dict = None
        self.is_pipeline = self.component_placement.is_disaggregated

        # Reward configurations
        if not self.cfg.reward.use_reward_model:
            assert self.cfg.reward.reward_type == "math", "only support math"
            self.reward_fn = math_verify_call

        # Rollout configurations
        self.rollout_group_name = self.cfg.rollout.group_name

        # Data I/O configurations
        self.num_train_steps = self.cfg.algorithm.n_minibatches
        self.is_data_io_rank = (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_context_parallel_rank() == 0
            and parallel_state.get_pipeline_model_parallel_rank() == 0
        )
        assert (
            self.cfg.data.rollout_batch_size * self.cfg.algorithm.group_size
        ) % parallel_state.get_data_parallel_world_size() == 0, (
            f"rollout_batch_size * group_size must be divisible by {role} data parallel size."
        )
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // parallel_state.get_data_parallel_world_size()
        )

        # Config validation
        if self.is_pipeline:
            assert not self.cfg.algorithm.normalize_advantages, (
                "Advantage normalization is not supported in pipeline mode."
            )
            assert not self.role_cfg.get("enable_dp_load_balance", False), (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.cfg.runner.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )

        # Create GLOO MP group for broadcast
        self._mp_group_ranks = parallel_state._MODEL_PARALLEL_GLOBAL_RANKS
        self._cp_group_ranks = parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS

        self._init_profiler()

    def _init_profiler(self):
        def _validate_schedule_info():
            assert (
                self.cfg.actor.megatron.profiler.schedule_warmup is not None
                and self.cfg.actor.megatron.profiler.schedule_warmup >= 0
            ), "<schedule_warmup> must be set and greater than 0 when using profiler."
            assert (
                self.cfg.actor.megatron.profiler.schedule_active is not None
                and self.cfg.actor.megatron.profiler.schedule_active > 0
            ), "<schedule_active> must be set and greater than 0 when using profiler."

        self.use_profiler = self.cfg.actor.megatron.use_profiler

        # here we should validate profiler's schedule info
        if self.use_profiler:
            _validate_schedule_info()
        self.profiler = (
            PyTorchProfiler.from_config(self.cfg.actor.megatron.profiler)
            if self.use_profiler
            else None
        )
        self._forward_only_record = PyTorchProfilerFunc(
            "forward_only", self.use_profiler
        )
        self._dynamic_batch_processing_record = PyTorchProfilerFunc(
            "dynamic_batch_processing", self.use_profiler
        )
        self._static_batch_processing_record = PyTorchProfilerFunc(
            "static_batch_processing", self.use_profiler
        )
        self._broadcast_outputs_record = PyTorchProfilerFunc(
            "broadcast_outputs", self.use_profiler
        )

        self._megatron_forward_backward_record = PyTorchProfilerFunc(
            "megatron_forward_backward", self.use_profiler
        )

    def _load_weight_and_optimizer(self, channel: Channel):
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with channel.gpu_lock:
            if self.is_weight_offloaded:
                self.onload_model_weights_and_grad(load_grad=self.offload_grad)
                self.is_weight_offloaded = False
            if self.is_optimizer_offloaded:
                self.onload_megatron_optimizer()
                self.is_optimizer_offloaded = False

    def init_worker(self):
        self.setup_model_and_optimizer()

        ref_policy_state_dict = None
        # only need this if we are running with inital kl penalty & full-parameter tuning
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
        self.ref_policy_state_dict = ref_policy_state_dict

        rollout_reshard_config = ReshardConfig(
            model_arch=self.cfg.rollout.model_arch,
            model_config=self.transformer_config,
            reshard_tp_size=self.cfg.rollout.tensor_parallel_size,
            reshard_pp_size=self.cfg.rollout.pipeline_parallel_size,
        )
        self.rollout_weights_reshard = MegatronCoreWeightReshard(rollout_reshard_config)
        self._setup_rollout_weight_dst_ranks()

        if self.component_placement.has_dedicated_inference:
            inference_reshard_config = ReshardConfig(
                model_arch=self.cfg.inference.model_arch,
                model_config=self.transformer_config,
                reshard_weights_format="mcore",
                reshard_tp_size=self.cfg.inference.model.tensor_model_parallel_size,
                reshard_pp_size=self.cfg.inference.model.pipeline_model_parallel_size,
            )
            self.inference_weights_reshard = MegatronCoreWeightReshard(
                inference_reshard_config
            )
            self._setup_inference_weight_dst_ranks()

        torch.distributed.barrier()

    def get_batch(
        self, channel: Channel
    ) -> Tuple[Dict[str, torch.Tensor], RolloutResult]:
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            result: RolloutResult = channel.get()
        else:
            if self.is_data_io_rank:
                result: RolloutResult = channel.get()
            else:
                result = None
            result = self.broadcast(result, ranks=self._mp_group_ranks)
            result = self.broadcast(result, ranks=self._cp_group_ranks)

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )

        return batch, result

    def put_result(self, result: RolloutResult, channel: Channel):
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            channel.put(result)
        else:
            if self.is_data_io_rank:
                channel.put(result)

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            batch = {key: val.cuda() for key, val in batch.items()}

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            response_len = self.response_len
            responses = input_ids[:, -response_len:]
            label = copy.deepcopy(position_ids)
            label[:, -response_len - 1 : -1] = responses
            label_mask = copy.deepcopy(attention_mask)
            label_mask[:, : -response_len - 1] = False
            label_mask[:, -1] = False

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape

                if self.calculate_entropy:
                    entropy, log_probs = vocab_parallel_entropy_and_log_probs(
                        logits,
                        label,
                        calculate_entropy_loss=self.calculate_entropy_loss,
                    )
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs, "entropy": entropy}
                else:
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs}

                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}

            output = self.custom_forward(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                temperature=self.cfg.algorithm.sampling_params.temperature,
            )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output["log_probs"][:, -response_len - 1 : -1].contiguous()

                return output, id_func

            def loss_func(output):
                curr_logprobs = output["log_probs"][
                    :, -response_len - 1 : -1
                ].contiguous()

                advantages = batch["advantages"]
                prev_logprobs = batch["prev_logprobs"]
                ref_logprobs = None
                if "ref_logprobs" in batch:
                    ref_logprobs = batch["ref_logprobs"]

                mask = batch["attention_mask"][:, -response_len:]

                # Calculate clipped PPO surrogate loss function.
                (
                    loss,
                    proportion_clipped,
                    approx_kl,
                    ratios,
                    cliped_ratio,
                    dual_cliped_ratio,
                ) = actor_loss_fn(
                    self.loss_agg_func,
                    curr_logprobs,
                    prev_logprobs,
                    advantages,
                    self.ratio_eps,
                    mask,
                )

                logging_loss = loss.detach()
                entropy_loss = torch.zeros(1, device=loss.device)
                if self.calculate_entropy:
                    entropy = output["entropy"][:, -response_len - 1 : -1].contiguous()
                    entropy_loss = self.loss_agg_func(entropy, mask=mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, curr_logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, mask)
                    loss = loss + kl_loss * self.kl_beta

                # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
                _imp = (ratios.detach().float() * mask).sum()
                torch.distributed.all_reduce(
                    _imp, group=parallel_state.get_data_parallel_group()
                )
                _n_valid_tokens = mask.count_nonzero().clone()
                torch.distributed.all_reduce(
                    _n_valid_tokens, group=parallel_state.get_data_parallel_group()
                )
                _imp /= _n_valid_tokens
                # Early stopping.
                if (
                    self.cfg.algorithm.early_stop_imp_ratio is not None
                    and _imp > self.cfg.algorithm.early_stop_imp_ratio
                ):
                    self.log_warning(
                        f"Current importance ratio {_imp.item():.4f} is larger "
                        f"than early stop threshold {self.cfg.algorithm.early_stop_imp_ratio}. Abandon this microbatch."
                    )
                    loss = loss * 0.0
                if self.cfg.algorithm.use_valid_token_scale:
                    loss_scale = (
                        mask.sum()
                        / self.global_valid_token
                        * parallel_state.get_data_parallel_world_size()
                        * self.num_microbatches
                    )
                    loss *= loss_scale.item()

                with torch.no_grad():
                    ratios = masked_mean(ratios.detach(), mask)
                    cliped_ratio = masked_mean(cliped_ratio.detach(), mask)
                    dual_cliped_ratio = masked_mean(dual_cliped_ratio.detach(), mask)
                    entropy_loss = entropy_loss.detach()
                    kl_loss = kl_loss.detach()
                    approx_kl = approx_kl.detach()
                    proportion_clipped = proportion_clipped.detach()

                (
                    reduced_actor_loss,
                    ratios,
                    cliped_ratio,
                    dual_cliped_ratio,
                    entropy_loss,
                    kl_loss,
                    approx_kl,
                    proportion_clipped,
                ) = average_losses_across_data_parallel_group(
                    [
                        logging_loss,
                        ratios,
                        cliped_ratio,
                        dual_cliped_ratio,
                        entropy_loss,
                        kl_loss,
                        approx_kl,
                        proportion_clipped,
                    ]
                )
                return (
                    loss,
                    {
                        "loss": reduced_actor_loss,
                        "ratio": ratios,
                        "cliped_ratio": cliped_ratio,
                        "dual_cliped_ratio": dual_cliped_ratio,
                        "entropy_loss": entropy_loss,
                        "kl_loss": kl_loss,
                        "approx_kl": approx_kl,
                        "proportion_clipped": proportion_clipped,
                    },
                )

            return output, loss_func

        return forward_output_and_loss_func

    def _get_num_microbatches(self, batch: Dict[str, torch.Tensor], forward_only: bool):
        batch_size = get_batch_size(batch)
        forward_micro_batch_size = (
            self.logprob_forward_micro_batch_size
            if forward_only
            else self.cfg.actor.micro_batch_size
        )
        num_microbatches = (
            max(1, batch_size // forward_micro_batch_size)
            if forward_only
            else get_num_microbatches()
        )
        return num_microbatches

    def run_forward_backward(self, batch: Dict[str, torch.Tensor], forward_only=True):
        """Run the forward and backward pass on the model.

        Args:
            batch_iterator (Iterator): The input batch iterator for the forward pass.
            forward_only (bool): If True, only run the forward pass without backpropagation.
        """
        clear_memory()
        if self.enable_dynamic_batch_size:
            # Enable dynamic batch sizing
            batch_iter, total_seqlen, num_microbatches, self.dbs_indices = (
                split_dynamic_batch_size(
                    batch=batch,
                    cp_world_size=parallel_state.get_context_parallel_world_size(),
                    vpp_world_size=parallel_state.get_virtual_pipeline_model_parallel_world_size(),
                    max_tokens_per_mbs=self.max_tokens_per_mbs,
                    microbatch_group_size_per_vp_stage=self.transformer_config.microbatch_group_size_per_vp_stage,
                )
            )
        else:
            total_seqlen = get_seq_length(batch)
            num_microbatches = self._get_num_microbatches(batch, forward_only)
            batch_iter = get_iterator_k_split(batch, num_splits=num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        self.num_microbatches = num_microbatches
        self.return_loss = not forward_only

        self.log_debug(f"{total_seqlen=}, {num_microbatches=}")

        if forward_only:
            self._forward_only_record.start()
        forward_outputs = fwd_bwd_function(
            forward_step_func=self.get_forward_step_func(),
            data_iterator=self.make_data_iterator_list(batch_iter),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=total_seqlen,
            micro_batch_size=1,
            collect_non_loss_data=forward_only,
        )
        if forward_only:
            self._forward_only_record.stop()

        outputs = self._process_fwd_bwd_outputs(forward_outputs, forward_only)

        return outputs

    def run_forward_backward_iterator(
        self, batch_iterator: BatchResizingIterator, forward_only: bool = False
    ):
        """Run the forward and backward pass on the model using batch resizing iterator.

        This function is solely intended for the training step of pipeline mode, which mixes RL pipeline with training pipeline parallelism.
        So this function enforces forward_only to be false.

        Args:
            batch_iterator (Iterator): The input batch iterator for the forward pass.
        """
        clear_memory()
        assert not self.enable_dynamic_batch_size, (
            "Dynamic batch size is not supported in pipeline mode."
        )

        sample_batch = batch_iterator.prefetch_one_batch()
        total_seqlen = get_seq_length(sample_batch)
        num_microbatches = self._get_num_microbatches(sample_batch, forward_only)

        fwd_bwd_function = get_forward_backward_func()
        self.num_microbatches = num_microbatches
        self.return_loss = not forward_only

        self.log_debug(f"{total_seqlen=}, {num_microbatches=}")

        if forward_only:
            self._forward_only_record.start()
        forward_outputs = fwd_bwd_function(
            forward_step_func=self.get_forward_step_func(),
            data_iterator=self.make_data_iterator_list(batch_iterator),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=total_seqlen,
            micro_batch_size=1,
            collect_non_loss_data=forward_only,
        )
        if forward_only:
            self._forward_only_record.stop()

        outputs = self._process_fwd_bwd_outputs(forward_outputs, forward_only)

        return outputs

    def _process_fwd_bwd_outputs(
        self, forward_outputs: Dict[str, torch.Tensor], forward_only: bool
    ):
        if forward_only:
            outputs = torch.cat(forward_outputs) if len(forward_outputs) > 0 else None
            if self.enable_dynamic_batch_size:
                indices = sum(self.dbs_indices, [])
                assert len(indices) == outputs.size(0), (
                    f"Dynamic batch size indices length {len(indices)} does not equal output length {outputs.size()}"
                )
                revert_indices = torch.tensor(
                    get_reverse_idx(indices), dtype=torch.long
                )
                outputs = outputs[revert_indices]
            outputs = broadcast_tensor_within_pp(outputs)
        else:
            outputs = {}
            if forward_outputs:
                keys = forward_outputs[0].keys()
                for key in keys:
                    metric_mean = torch.stack(
                        [loss_reduced[key] for loss_reduced in forward_outputs]
                    ).mean()
                    outputs[key] = metric_mean.cpu().item()
            output_list = [outputs]
            torch.distributed.broadcast_object_list(output_list, get_last_rank())
            outputs = output_list[0]
        return outputs

    # Training
    def training_step(self, batch: Dict[str, torch.Tensor] | BatchResizingIterator):
        """Run a single training step on the model.

        Args:
            batch (Dict[str, torch.Tensor] | BatchResizingIterator): The input batch containing the data for the forward pass.
        """
        set_sync_funcs(self, forward_only=False)
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        if isinstance(batch, BatchResizingIterator):
            train_metrics = self.run_forward_backward_iterator(batch)
        else:
            train_metrics = self.run_forward_backward(batch, forward_only=False)
        increment = (
            get_num_microbatches()
            * self.cfg.actor.micro_batch_size
            * parallel_state.get_data_parallel_world_size()
        )
        success, grad_norm, num_zeros_in_grad, lr = self.optimizer_step(increment)

        # Training metrics
        train_metrics["grad_norm"] = (
            grad_norm if grad_norm is not None else float("nan")
        )
        train_metrics["num_zeros_in_grad"] = (
            num_zeros_in_grad if num_zeros_in_grad is not None else float("nan")
        )
        train_metrics["lr"] = lr if lr is not None else float("nan")
        train_metrics["update_success"] = int(success)

        return train_metrics

    def _training_setup(self):
        set_train(self)
        configure_batch_sizes(
            rank=torch.distributed.get_rank(),
            mbs=self.cfg.actor.micro_batch_size,
            gbs=self.cfg.actor.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )

    def _setup_valid_token_scale(self, batch: Optional[Dict[str, torch.Tensor]] = None):
        if batch is None:
            self.global_valid_token = (
                self.average_response_len
                * get_num_microbatches()
                * self.cfg.actor.micro_batch_size
            )
        else:
            loss_mask = batch["attention_mask"][:, -self.response_len :]
            global_valid_token = loss_mask.to(dtype=torch.float32).sum().cuda()
            torch.distributed.all_reduce(
                global_valid_token, group=parallel_state.get_data_parallel_group()
            )
            self.global_valid_token = global_valid_token
            return batch

    def _dp_load_balance(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=parallel_state.get_data_parallel_world_size(),
            dp_rank=parallel_state.get_data_parallel_rank(),
            dp_group=parallel_state.get_data_parallel_group(),
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def _gather_weights_among_dp(self):
        """Gather weights across DP when using distributed optimizer with overlap_param_gather.

        When overlap_param_gather is enabled, weights are scattered across DP and gathered in the next forward pass.
        We need to force a gather here to ensure all weights are correct before the next weight sync.
        """
        if (
            self.role_cfg.optim.use_distributed_optimizer
            and self.role_cfg.optim.overlap_param_gather
        ):
            for model_chunk in self.model:
                assert isinstance(model_chunk, DDP)
                model_chunk.start_param_sync(force_sync=True)

    def run_training(self, input_channel: Channel):
        """Run the training loop for the actor."""
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        # Get all batches for this DP
        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        batch = RolloutResult.merge_batches(batches)

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer(input_channel)
        self._training_setup()

        # Advantage normalization
        if self.cfg.algorithm.normalize_advantages:
            mask = batch["attention_mask"][:, -self.response_len :]
            batch["advantages"] = masked_normalization(batch["advantages"], mask)

        # Valid token scale
        if self.cfg.algorithm.use_valid_token_scale:
            self._setup_valid_token_scale(batch)

        # DP batch load balance
        if self.cfg.actor.get("enable_dp_load_balance", False):
            batch = self._dp_load_balance(batch)

        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.num_train_steps,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        # Global batch iterations
        with self.worker_timer():
            training_metrics_list = []
            for global_batch in global_batches:
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        # Rollout metrics
        rollout_metrics = self._compute_rollout_metrics(batch)

        return rollout_metrics, training_metrics_list

    def run_training_pipeline(self, input_channel: Channel):
        """Run the training loop for the actor."""
        self._load_weight_and_optimizer(input_channel)
        self._training_setup()
        # Built iterator for Megatron's pipeline schedule to run
        # NOTE: We cannot iterate over the iterator here, as Megatron's pipeline schedule is responsible for iterating over data
        # As a result, we need to separate the implementation for pipeline and non-pipeline mode for Megatron
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.role_cfg.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.num_train_steps,
            forward_only=False,
        )

        # Advantage normalization
        assert not self.cfg.algorithm.normalize_advantages, (
            "Advantage normalization is not supported in pipeline mode."
        )

        # Valid token scale
        if self.cfg.algorithm.use_valid_token_scale:
            self._setup_valid_token_scale()

        # DP batch load balance
        assert not self.cfg.actor.get("enable_dp_load_balance", False), (
            "DP load balance is not supported in pipeline mode."
        )

        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        # Global batch iterations
        training_metrics_list = []
        for _ in range(self.num_train_steps):
            training_metrics = self.training_step(train_batch_iterator)
            train_batch_iterator.check_finished_global_batch()
            training_metrics_list.append(training_metrics)

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics = self._compute_rollout_metrics(batch)

        return rollout_metrics, training_metrics_list

    # Inference
    def _setup_inference_weight_dst_ranks(self):
        self._weight_dst_rank_in_inference = self.get_inference_weight_dst_ranks(
            self.cfg.inference.model.tensor_model_parallel_size,
            self.cfg.inference.model.pipeline_model_parallel_size,
        )

    def get_inference_weight_dst_ranks(self, inference_tp, inference_pp):
        """
        Calculate the list of ranks corresponding to the first complete inference model parallel group after resharding.

        Returns:
            List of ranks for the first complete inference model parallel group after resharding
        """

        model_parallel_size = inference_tp * inference_pp
        # After resharding, the number of GPUs in a complete model parallel group = new TP × new PP
        # The first complete model parallel group consists of consecutive ranks starting from 0
        return list(range(model_parallel_size))

    def _get_inference_model_state_dict(self):
        """Get the state dictionary of the model for inference."""
        return self.inference_weights_reshard.gather_and_reshard_model(
            unwrap_model(self.model)
        )

    def sync_model_to_inference(self):
        inference_state_dict = self._get_inference_model_state_dict()

        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                self.send(inference_state_dict, self.cfg.inference.group_name, rank)

        self.log_debug(
            f"{self.__class__.__name__}: sync_model_to_inference resharding done."
        )

    @torch.no_grad()
    def inference_step(self, batch):
        set_eval(self)
        return self.run_forward_backward(batch, forward_only=True)

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            # Must be called after batch is retrieved, suggesting that rollout has stopped
            # Otherwise, loading model might cause OOM in the collocated mode
            self._load_weight_and_optimizer(input_channel)

            # Prev logprobs
            with self.worker_timer():
                prev_logprobs = self.inference_step(batch)
                rollout_result.prev_logprobs = prev_logprobs.cpu()

            # Ref logprobs
            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None
                with cpu_weight_swap(self.model[0], self.ref_policy_state_dict):
                    ref_logprobs = self.inference_step(batch)
                    rollout_result.ref_logprobs = ref_logprobs.cpu()

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    # Rewards
    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        if self.is_pipeline:
            # In pipeline mode, rewards are computed in the rollout
            with self.worker_timer():
                return
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            # Compute rule-based reward
            with self.worker_timer():
                if rollout_result.rewards is None:
                    rollout_result.rewards = self._compute_batch_rewards(
                        batch, rollout_result.answers
                    ).cpu()

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_batch_rewards(
        self, batch: Dict[str, torch.Tensor], answers: List[str]
    ):
        """Reward computation using non-model based reward."""
        all_reward_scores = []
        texts = []
        for response, response_len in zip(
            batch["input_ids"],
            batch["response_lengths"],
        ):
            response = response[
                self.cfg.data.max_prompt_length : self.cfg.data.max_prompt_length
                + response_len
            ]
            texts.append(
                self.tokenizer.decode(response.tolist(), skip_special_tokens=True)
            )

        if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
            rewards = self.reward_fn(texts, answers)
            reward_scores = [
                self.cfg.reward.reward_scale
                if reward == 1
                else -self.cfg.reward.reward_scale
                for reward in rewards
            ]
            all_reward_scores.extend(reward_scores)

        if len(all_reward_scores) > 0:
            new_all_rewards = []

            for response in all_reward_scores:
                if response is None:
                    response = 0.0
                new_all_rewards.append(response)

            all_reward_scores = torch.as_tensor(
                new_all_rewards,
                dtype=torch.float,
                device=torch.cuda.current_device(),
            ).view(-1, 1)
        return broadcast_tensor_within_mp(all_reward_scores).flatten().to("cpu")

    # Advantages and returns
    def compute_advantages_and_returns(
        self, input_channel: Channel, output_channel: Channel
    ):
        """Compute the advantages and returns.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        if self.is_pipeline:
            # In pipeline mode, advantages are computed in the rollout
            with self.worker_timer():
                return
        clear_memory()
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            with self.worker_timer():
                if rollout_result.advantages is None:
                    mask = batch["attention_mask"][:, -self.response_len :]
                    advantages, returns = calculate_adv_and_returns(
                        self.cfg.algorithm.adv_type,
                        batch["rewards"].cuda(),
                        mask.cuda(),
                        self.cfg.algorithm.group_size,
                    )
                    rollout_result.advantages = advantages.cpu()

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    # Rollout
    def _get_rollout_model_state_dict(self):
        """Get the state dictionary of the model for rollout."""
        return self.rollout_weights_reshard.gather_and_reshard_model(
            unwrap_model(self.model)
        )

    def _setup_rollout_weight_dst_ranks(self):
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self.component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self):
        if hasattr(self, "reshard_state_dict"):
            del self.reshard_state_dict

    def sync_model_to_rollout(self):
        """Send the model weights to the destination ranks in the rollout task."""
        if self.component_placement._placement_mode == PlacementMode.COLLOCATED:
            if self.offload_optimizer:
                self.offload_megatron_optimizer()
                self.is_optimizer_offloaded = True
            self.reshard_state_dict = self._get_rollout_model_state_dict()
            if self.offload_weight:
                self.offload_model_weights_and_grad(offload_grad=self.offload_grad)
                self.is_weight_offloaded = True

            handle = {k: reduce_tensor(v) for k, v in self.reshard_state_dict.items()}
            self.send(handle, self.rollout_group_name, self._weight_dst_rank_in_rollout)
        else:
            assert (
                self.component_placement._placement_mode == PlacementMode.DISAGGREGATED
            ), "Unsupported placement mode for sending weights."
            assert isinstance(self._weight_dst_rank_in_rollout, list), (
                f"In disaggregated mode, weight_dst_rank_in_rollout should be a list of ranks, got {type(self._weight_dst_rank_in_rollout)}"
            )
            self.reshard_state_dict = self._get_rollout_model_state_dict()
            for weight_dst_rank in self._weight_dst_rank_in_rollout:
                self.send(
                    self.reshard_state_dict,
                    self.rollout_group_name,
                    weight_dst_rank,
                )

    def _compute_rollout_metrics(self, batch):
        rollout_metrics, total_prompt_lengths, total_decode_lengths = (
            compute_rollout_metrics(
                batch, self.cfg.data.max_prompt_length, self.response_len
            )
        )

        rollout_metrics = cpu_dict(rollout_metrics)

        if self.cfg.actor.get("calculate_flops", False):
            rollout_tflops = self.flops_calculator.flops_generate(
                total_prompt_lengths, total_decode_lengths
            )
            rollout_tflops = rollout_tflops.float().sum().item() / 1e12
            inference_tflops = self.flops_calculator.flops_inference(
                total_prompt_lengths + total_decode_lengths
            )
            inference_tflops = inference_tflops.float().sum().item() / 1e12

            rollout_metrics.update(
                {
                    "rollout_tflops": rollout_tflops,
                    "inference_tflops": inference_tflops,
                    "training_tflops": inference_tflops * 3,  # factor
                }
            )
        return rollout_metrics
