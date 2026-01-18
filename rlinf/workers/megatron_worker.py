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

import time
from functools import partial
from typing import Callable, Optional

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.global_vars import get_args
from megatron.training.training import unwrap_model
from omegaconf import DictConfig

from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
)
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
from rlinf.scheduler.dynamic_scheduler.utils import (
    get_scheduler_channel,
    get_scheduler_request_queue,
    get_scheduler_response_queue,
    get_valid_dp_sizes,
)
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_last_rank,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    broadcast_tensor_within_pp,
    compute_rollout_metrics,
    masked_normalization,
)
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
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
    seq_mean_token_mean,
    seq_mean_token_sum,
)

try:
    from params_resharding import resharding_init

    HAVE_RESHARDING = True
except ImportError:
    HAVE_RESHARDING = False

# base model for MegatronActor and MegatronCritic
class MegatronWorker(MegatronModelManager, Worker):
    """The class for running the actor/critic training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role
    ):
        """Initialize the MegatronWorker.

        Args:
            cfg (DictConfig): The configuration for the worker.
        """
        Worker.__init__(self)
        self.role = role
        role_cfg = getattr(cfg, role, None)
        self.role_cfg = role_cfg
        if role_cfg is None:
            raise ValueError(f"Role {role} is not defined in the configuration.")
        super().__init__(role_cfg)
        self.cfg = cfg
        self.inference_cfg = getattr(cfg, '_'.join([role, 'inference']), None)
        if self.inference_cfg is None:
            # compatible with old code
            self.inference_cfg = getattr(cfg, 'inference', None)

        self.component_placement = placement

        # Data configurations
        self.response_len = (
            role_cfg.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.average_response_len = self.response_len

        # TODO move to role code or change name
        self.logprob_forward_micro_batch_size = (
            self.cfg.algorithm.logprob_forward_micro_batch_size
        )

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
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        # Actor configurations
        self.enable_dynamic_batch_size = self.cfg.runner.enable_dynamic_batch_size
        self.max_tokens_per_mbs = self.cfg.runner.max_tokens_per_mbs
        self.offload_optimizer = self.role_cfg.offload_optimizer
        self.offload_weight = self.role_cfg.offload_weight
        self.offload_grad = self.role_cfg.offload_grad

        self.ref_policy_state_dict = None
        self.is_pipeline = self.component_placement.is_pipeline
        self.placement_mode = placement._placement_mode
        self.enable_dp_load_balance = self.role_cfg.get("enable_dp_load_balance", False)

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
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.cfg.runner.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )

        self.use_profiler = self.role_cfg.megatron.use_profiler
        if self.use_profiler:
            self.init_profiler()

        self._init_auto_scheduler(role)

    def _init_auto_scheduler(self, role: str):
        self.use_auto_scheduler = self.placement_mode == PlacementMode.AUTO
        self.use_pre_process_policy = (
            getattr(self.cfg.cluster, "use_pre_process_policy", False)
            and self.use_auto_scheduler
        )
        self.recreate_nccl_groups = (
            getattr(self.cfg.cluster, "recreate_nccl_groups", False)
            and self.use_pre_process_policy
        )

        if self.use_auto_scheduler:
            assert HAVE_RESHARDING, (
                "params_resharding is not installed, resharding is not supported"
            )
        if self.use_auto_scheduler and self._rank == 0:
            self.schedule_channel = self.connect_channel(
                get_scheduler_channel(role, self._rank)
            )
            self.scheduler_request_queue = get_scheduler_request_queue()
            self.scheduler_response_queue = get_scheduler_response_queue()

    def _load_weight(self):
        # only weights need to be loaded for run_inference()
        if not self.is_running:
            return
        with self.device_lock:
            self.onload_model_weights_and_grad(load_grad=self.offload_grad)

    def _load_weight_and_optimizer(self):
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        if not self.is_running:
            return
        with self.device_lock:
            self.onload_model_weights_and_grad(load_grad=self.offload_grad)
            self.onload_megatron_optimizer()

    def _offload_weight_and_optimizer(self):
        if not self.is_running:
            return
        with self.device_lock:
            if self.offload_weight:
                self.offload_model_weights_and_grad(offload_grad=self.offload_grad)
            if self.offload_optimizer:
                self.offload_megatron_optimizer()

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.use_auto_scheduler:
            self.init_trainer_resharding()
            if not self.is_running:
                return

        # initialization of ref_policy_state_dict and rollout_resharding_conf
        # are moved to MegatronActor.init_worker

        if self.component_placement.has_dedicated_inference_for_role(self.role):
            inference_reshard_config = ReshardConfig(
                model_type=self.inference_cfg.model_type,
                model_config=self.transformer_config,
                reshard_weights_format="mcore",
                reshard_tp_size=self.inference_cfg.model.tensor_model_parallel_size,
                reshard_pp_size=self.inference_cfg.model.pipeline_model_parallel_size,
                mg_ep_size=self.role_cfg.model.expert_model_parallel_size,
                mg_tpe_size=self.role_cfg.model.expert_tensor_parallel_size,
                moe_grouped_gemm=self.role_cfg.model.get("moe_grouped_gemm", None),
            )
            self.inference_weights_reshard = MegatronCoreWeightReshard(
                inference_reshard_config
            )
            self._setup_inference_weight_dst_ranks()

        # offload weights and optimizers after initialization if offload is enabled
        self._offload_weight_and_optimizer()

        torch.distributed.barrier()

    def get_batch(
        self, channel: Channel, tag: Optional[str] = None
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        if tag is not None:
            # for pipeline mode to filter channel communicate time.
            start_time = time.perf_counter()
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            result: RolloutResult = channel.get()
        else:
            if self.is_data_io_rank:
                result: RolloutResult = channel.get()
            else:
                result = None
            result = self.broadcast(
                result, ranks=parallel_state._MODEL_PARALLEL_GLOBAL_RANKS
            )
            result = self.broadcast(
                result, ranks=parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS
            )

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.role_cfg.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )

        if tag is not None:
            duration = time.perf_counter() - start_time
            self._timer_metrics[tag] = self._timer_metrics.get(tag, 0.0) - duration
        return batch, result

    def all_reduce_dp_min(
        self,
        obj: int,
    ):
        obj_tensor = torch.tensor(
            [obj], dtype=torch.long, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(
            obj_tensor,
            torch.distributed.ReduceOp.MIN,
            group=parallel_state.get_data_parallel_group(),
        )
        return obj_tensor.item()

    def _get_sample_count_from_rollout_results(self, rollout_results : list[RolloutResult]):
        return sum([r.num_sequence for r in rollout_results])

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_num_samples: int,
        max_num_samples: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        if self.is_data_io_rank:
            # get min_num_samples
            while self._get_sample_count_from_rollout_results(rollout_results) < min_num_samples:
                if unfinished_result is not None:
                    rollout_result: RolloutResult = unfinished_result.wait()
                    unfinished_result = None
                else:
                    rollout_result: RolloutResult = input_channel.get()
                rollout_results.append(rollout_result)
                # print(f'run_inference: rollout_result append, {sum([len(r.prompt_ids) for r in rollout_results])}, {min_num_samples}')

            # try to get result as much
            # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
            # stop at: the min result between dp (result_len) is same as the last min result
            last_result_len = 0
            result_len = len(rollout_results)
            time_until = time.time() + 0.1
            while last_result_len < result_len:
                if self._get_sample_count_from_rollout_results(rollout_results) < max_num_samples:
                    if unfinished_result is None:
                        unfinished_result = input_channel.get(async_op=True)
                    else:
                        time.sleep(0.001)
                    if unfinished_result.done():
                        rollout_results.append(unfinished_result.wait())
                        unfinished_result = None
                    if time.time() >= time_until:
                        last_result_len = result_len
                        result_len = self.all_reduce_dp_min(len(rollout_results))
                        if last_result_len < result_len:
                            time_until = time.time() + 0.1
                else:
                    last_result_len = result_len
                    result_len = self.all_reduce_dp_min(len(rollout_results))

        # broadcast to other ranks
        if not self.is_data_io_rank:
            result_len = None
        result_len = self.broadcast(
            result_len, ranks=parallel_state._MODEL_PARALLEL_GLOBAL_RANKS
        )
        result_len = self.broadcast(
            result_len, ranks=parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS
        )
        if self.is_data_io_rank:
            cliped_results = list(rollout_results[result_len:])
            rollout_results = rollout_results[:result_len]
        else:
            rollout_results = [None for _ in range(result_len)]
        for i in range(result_len):
            rollout_result = rollout_results[i]
            rollout_result = self.broadcast(
                rollout_result, ranks=parallel_state._MODEL_PARALLEL_GLOBAL_RANKS
            )
            rollout_result = self.broadcast(
                rollout_result, ranks=parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS
            )
            rollout_results[i] = rollout_result

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.role_cfg.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    def put_result(self, result: RolloutResult, channel: Channel | list[Channel]):
        if not isinstance(channel, list):
            channel = [channel]
        for ch in channel:
            if ch.is_local:
                # Local channel, every process will put its own data locally
                # No need to broadcast
                ch.put(result, async_op=True)
            else:
                if self.is_data_io_rank:
                    ch.put(result, async_op=True)

    def _get_num_microbatches(self, batch: dict[str, torch.Tensor], forward_only: bool):
        if forward_only:
            batch_size = get_batch_size(batch)
            return max(1, batch_size // self.logprob_forward_micro_batch_size)
        else:
            return get_num_microbatches()

    def run_forward_backward(self, batch: dict[str, torch.Tensor], forward_only=True):
        """Run the forward and backward pass on the model.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch for the forward pass.
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

        if self.use_profiler:
            self.profiler.start(forward_only=forward_only)
            forward_backward_record = (
                self.forward_only_record
                if forward_only
                else self.megatron_forward_backward_record
            )
            forward_backward_record.start()
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
        if self.use_profiler:
            forward_backward_record.stop()
            self.profiler.stop(forward_only=forward_only)

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

        if self.use_profiler:
            self.profiler.start(forward_only=forward_only)
            forward_backward_record = (
                self.forward_only_record
                if forward_only
                else self.megatron_forward_backward_record
            )
            forward_backward_record.start()
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
        if self.use_profiler:
            forward_backward_record.stop()
            self.profiler.stop(forward_only=forward_only)

        outputs = self._process_fwd_bwd_outputs(forward_outputs, forward_only)

        return outputs

    def _process_fwd_bwd_outputs(
        self, forward_outputs: dict[str, torch.Tensor], forward_only: bool
    ):
        if forward_only:
            outputs = torch.cat(forward_outputs) if len(forward_outputs) > 0 else None
            if self.enable_dynamic_batch_size and outputs is not None:
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
    def training_step(self, batch: dict[str, torch.Tensor] | BatchResizingIterator):
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
            * self.role_cfg.micro_batch_size
            * parallel_state.get_data_parallel_world_size()
        )
        success, grad_norm, num_zeros_in_grad, lr = self.optimizer_step(increment)

        # Training metrics
        train_metrics[f"{self.role}/grad_norm"] = (
            grad_norm if grad_norm is not None else float("nan")
        )
        train_metrics[f"{self.role}/num_zeros_in_grad"] = (
            num_zeros_in_grad if num_zeros_in_grad is not None else float("nan")
        )
        train_metrics[f"{self.role}/lr"] = lr if lr is not None else float("nan")
        train_metrics[f"{self.role}/update_success"] = int(success)

        return train_metrics

    def _training_setup(self):
        set_train(self)
        self.calc_num_microbatches()

    def _setup_valid_token_scale(self, batch: Optional[dict[str, torch.Tensor]] = None):
        if batch is None:
            self.global_valid_token = (
                self.average_response_len
                * get_num_microbatches()
                * self.role_cfg.micro_batch_size
            )
        else:
            loss_mask = batch["response_mask"][:, -self.response_len :]
            global_valid_token = loss_mask.to(dtype=torch.float32).sum().cuda()
            torch.distributed.all_reduce(
                global_valid_token, group=parallel_state.get_data_parallel_group()
            )
            self.global_valid_token = global_valid_token
            return batch

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
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
        if not self.is_running:
            return
        if (
            self.role_cfg.optim.use_distributed_optimizer
            and self.role_cfg.optim.overlap_param_gather
        ):
            for model_chunk in self.model:
                assert isinstance(model_chunk, DDP)
                model_chunk.start_param_sync(force_sync=True)

    def run_training(self, input_channel: Channel, output_channel: Channel | None = None,
                     do_offload: bool = True, compute_rollout_metrics: bool = True):
        """Run the training loop."""
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        # Get all batches for this DP
        batches = []
        rollout_results = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            rollout_results.append(rollout_result)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)

        # Compute advantages and returns
        if "advantages" not in batch:
            batch = self.compute_advantages_and_returns(batch)
            if self.cfg.algorithm.normalize_advantages:
                mask = batch["attention_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(
                    batch["advantages"],
                    mask,
                    unbiased=True,
                    group=parallel_state.get_data_parallel_group(),
                )

            if output_channel is not None:
                rollout_result.returns = batch["returns"].cpu()
                rollout_result.advantages = batch["advantages"].cpu()

        # Rollout metrics
        rollout_metrics = None
        if compute_rollout_metrics:
            rollout_metrics = self._compute_rollout_metrics(batch)
            
        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()
        self._training_setup()

        # Valid token scale
        if self.cfg.algorithm.use_valid_token_scale:
            self._setup_valid_token_scale(batch)

        # DP batch load balance
        if (
            self.role_cfg.get("enable_dp_load_balance", False)
            and parallel_state.get_data_parallel_world_size() > 1
        ):
            batch = self._dp_load_balance(batch)

        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.num_train_steps,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.role_cfg.seed,
        )

        # Global batch iterations
        with self.worker_timer():
            training_metrics_list = []
            for global_batch in global_batches:
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        if output_channel is not None:
            self.put_result(rollout_result, output_channel)

        if do_offload:
            # this is training, so we offload both weights and optimizer states
            self._offload_weight_and_optimizer()

        return rollout_metrics, training_metrics_list

    def run_training_pipeline(self, input_channel: Channel):
        """Run the training loop for the actor."""
        self.scheduler_pre_process()
        self._load_weight_and_optimizer()
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

        # Compute advantages and returns
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        # Advantage normalization
        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(
                    batch["advantages"],
                    mask,
                    group=parallel_state.get_data_parallel_group(),
                )
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        # Valid token scale
        if self.cfg.algorithm.use_valid_token_scale:
            self._setup_valid_token_scale()

        # DP batch load balance
        assert not self.role_cfg.get("enable_dp_load_balance", False), (
            "DP load balance is not supported in pipeline mode."
        )

        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        # Global batch iterations
        training_metrics_list = []
        for _ in range(self.num_train_steps):
            if self.is_running:
                train_batch_iterator.reset_total_batch_size(
                    self.total_batch_size_per_dp
                )
                training_metrics = self.training_step(train_batch_iterator)
                train_batch_iterator.check_finished_global_batch()
                training_metrics_list.append(training_metrics)
            self.scheduler_scale_sync()

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics = self._compute_rollout_metrics(batch)

        return rollout_metrics, training_metrics_list

    # Elastic-Training
    def get_scheduler_response(self, send_request_first: bool):
        if self._rank == 0:
            if send_request_first:
                self.schedule_channel.put(None, key=self.scheduler_response_queue)

            response = self.schedule_channel.get(key=self.scheduler_request_queue)
        else:
            response = None
        return self.broadcast_obj(response)

    def scheduler_pre_process(self):
        """Wait for the scheduler to send the pre-process response."""
        if not self.use_pre_process_policy:
            return
        self.get_scheduler_response(False)

    def scheduler_scale_sync(self):
        """Get a resharding response from the scheduler and apply this resharding response if it's not None."""
        if not self.use_auto_scheduler:
            return
        resharding_response = self.get_scheduler_response(True)
        if resharding_response is not None:
            self.apply_parallel_strategy(resharding_response)
            self.calc_num_microbatches()

    def scheduler_offload_sync(self):
        """Send offloaded signal to the scheduler."""
        inference_world_size = self.component_placement.inference_world_size
        if inference_world_size == 0 or not self.use_auto_scheduler:
            return
        assert not self.is_weight_offloaded
        self.offload_model_weights_and_grad(offload_grad=True)
        self.broadcast(None, ranks=list(range(inference_world_size)))
        self.is_weight_offloaded = True
        if self._rank == 0:
            self.schedule_channel.put(None, key=self.scheduler_response_queue)

    def get_rollout_metrics_group(self, batch):
        if not self.use_auto_scheduler:
            return parallel_state.get_data_parallel_group()

        if len(batch) == 0:
            trained_batch_size = 0
        else:
            trained_batch_size = get_batch_size(batch)

        max_data_parallel_group = parallel_state.get_data_parallel_group_elastic_max()
        max_data_parallel_ranks = torch.distributed.get_process_group_ranks(
            max_data_parallel_group
        )

        rollout_metrics_dp_ranks_states = torch.tensor(
            [
                (dp_rank == self._rank and trained_batch_size > 0)
                for dp_rank in max_data_parallel_ranks
            ]
        ).cuda()
        torch.distributed.all_reduce(
            rollout_metrics_dp_ranks_states,
            torch.distributed.ReduceOp.MAX,
            group=max_data_parallel_group,
        )

        rollout_metrics_dp_ranks_states = rollout_metrics_dp_ranks_states.tolist()
        rollout_metrics_valid_dp_ranks = [
            rank
            for rank, state in zip(
                max_data_parallel_ranks, rollout_metrics_dp_ranks_states
            )
            if state
        ]

        if trained_batch_size > 0:
            return parallel_state.create_group(rollout_metrics_valid_dp_ranks)
        return None

    def calc_num_microbatches(self):
        if not self.is_running:
            return
        configure_batch_sizes(
            rank=torch.distributed.get_rank(),
            mbs=self.role_cfg.micro_batch_size,
            gbs=self.role_cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // parallel_state.get_data_parallel_world_size()
        )
        if self._rank == 0:
            self.log_info(
                f"run_training_pipeline: mbs={self.role_cfg.micro_batch_size}, gbs={self.role_cfg.global_batch_size}, dp={parallel_state.get_data_parallel_world_size()}, self.total_batch_size_per_dp={self.total_batch_size_per_dp}"
            )

    def init_trainer_resharding(self, first_world_size: int = -1):
        """Init resharding func."""
        from megatron.core import __version__ as megatron_version
        from packaging import version

        assert version.parse(megatron_version).minor == 11, (
            "only megatron 0.11 is supported for online-resharding now"
        )

        args = get_args()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        args.data_parallel_size = args.world_size // (
            args.tensor_model_parallel_size
            * args.pipeline_model_parallel_size
            * args.context_parallel_size
        )
        args.load = None
        self.default_parallel_strategy = {
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "context_parallel_size": args.context_parallel_size,
        }
        default_model_parallel_size_with_cp = (
            args.tensor_model_parallel_size
            * args.pipeline_model_parallel_size
            * args.context_parallel_size
        )

        assert args.world_size == self.component_placement._cluster_num_gpus, (
            "In Auto-Scheduler mode, actor should be initialized on all GPUs."
        )
        assert self.component_placement.actor_world_size < args.world_size

        valid_dp_sizes = get_valid_dp_sizes(
            self.cfg,
            self.component_placement._cluster_num_gpus,
            default_model_parallel_size_with_cp,
        )
        assert len(valid_dp_sizes) > 0
        resharding_strategies = []

        for valid_dp_size in reversed(valid_dp_sizes):
            world_size = default_model_parallel_size_with_cp * valid_dp_size
            assert world_size <= self.component_placement._cluster_num_gpus

            resharding_strategies.append(
                {
                    "world_size": world_size,
                    "tensor_model_parallel_size": args.tensor_model_parallel_size,
                    "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
                    "context_parallel_size": args.context_parallel_size,
                }
            )

        assert resharding_strategies[0]["world_size"] == args.world_size

        self.trainer_resharding_func: Callable = resharding_init(
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.lr_scheduler,
            trainer_parallel_strategies=resharding_strategies,
            offload_frist_strategy=False,
            model_provider=self.model_provider_func,
            _logger=self._logger,
        )

        if first_world_size == -1:
            first_world_size = self.component_placement.actor_world_size

        self.is_running = True
        self.apply_parallel_strategy({"world_size": first_world_size})

    def apply_parallel_strategy(self, parallel_strategy):
        """Apply specified training parallel strategy"""

        args = get_args()
        args.load = None

        parallel_keys = [
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ]
        assert parallel_strategy.get("world_size") is not None, (
            "Error : can't find world_size in parallel_strategy"
        )
        if parallel_strategy.get("context_parallel_size") is not None:
            assert (
                parallel_strategy["context_parallel_size"]
                == self.default_parallel_strategy["context_parallel_size"]
            ), "change context_parallel_size is not supported"

        new_parallel_strategy = {}
        for parallel_key in parallel_keys:
            if parallel_strategy.get(parallel_key) is not None:
                new_parallel_strategy[parallel_key] = parallel_strategy[parallel_key]
            else:
                new_parallel_strategy[parallel_key] = self.default_parallel_strategy[
                    parallel_key
                ]

        if self._rank == 0:
            self.log_info(
                f"[ElasticMegatron-Info] start resharing with new_parallel_strategy = {new_parallel_strategy}"
            )
        training_states, _ = self.trainer_resharding_func(
            new_parallel_strategy=new_parallel_strategy
        )

        if training_states is not None:
            self.model = training_states.model
            self.optimizer = training_states.optimizer
            self.lr_scheduler = training_states.opt_param_scheduler
            self.is_running = True
        else:
            self.is_running = False

    def broadcast_obj(self, obj):
        parallel_state.global_barrier_by_gloo()
        device = torch.cuda.current_device()

        if self._rank == 0:
            torch.distributed.broadcast_object_list(
                [obj], src=0, device=device, group=torch.distributed.GroupMember.WORLD
            )
        else:
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list,
                src=0,
                device=device,
                group=torch.distributed.GroupMember.WORLD,
            )
            obj = obj_list[0]
        return obj

    # Inference
    def _setup_inference_weight_dst_ranks(self):
        self._weight_dst_rank_in_inference = self.get_inference_weight_dst_ranks(
            self.inference_cfg.model.tensor_model_parallel_size,
            self.inference_cfg.model.pipeline_model_parallel_size,
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
        model = unwrap_model(self.model)
        model_state_dict = {}
        for key, val in model[0].state_dict().items():
            if "_extra_state" in key:
                continue
            model_state_dict[key] = val
        return self.inference_weights_reshard.gather_and_reshard_model(
            model_state_dict, self.dst_tp_rank
        )

    def sync_model_to_inference(self):
        if not self.is_running:
            return

        assert not self.is_weight_offloaded, \
            "Weight has been offloaded, which prevents from syncing model to inference"

        inference_state_dict = self._get_inference_model_state_dict()

        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                self.send(inference_state_dict, self.inference_cfg.group_name, rank)

        self.log_debug(
            f"{self.__class__.__name__}: sync_model_to_inference resharding done."
        )

    @torch.no_grad()
    def inference_step(self, batch):
        # set the megatron worker in inference step
        set_sync_funcs(self, forward_only=True)
        set_eval(self)
        return self.run_forward_backward(batch, forward_only=True)

    def run_inference(
            self,
            input_channel: Channel,
            output_channel: Channel,
            compute_ref_logprobs: bool = False,
            do_offload = True
    ):
        """
        For actor model, compute prev/ref logprobs using the actor model's forward.
        For critic model, compute values

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference done
        """
        inference_split = self.role_cfg.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.total_batch_size_per_dp // self.micro_batch_size
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"MegatronWorker: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_num_samples = 1
        max_num_samples = self.total_batch_size_per_dp // inference_split
        if not self.is_pipeline:
            min_num_samples = max_num_samples
            coll_rollout_results = []
        total_num_samples = 0
        total_result_len = 0

        cliped_results, unfinished_result = [], None
        while total_num_samples < self.total_batch_size_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_num_samples, self.total_batch_size_per_dp - total_num_samples),
                    min(max_num_samples, self.total_batch_size_per_dp - total_num_samples),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            total_num_samples += rollout_result.num_sequence
            self.log_info(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_num_samples={total_num_samples}/{self.total_batch_size_per_dp}"
            )
            self._load_weight_and_optimizer()

            with self.worker_timer():
                # prev logprobs for actor, values for critic
                infer_out = self.inference_step(batch).cpu()
                if self.role == 'actor':
                    prev_logprobs = infer_out
                    if rollout_result.rollout_logprobs is not None:
                        # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                        rollout_result.recompute_prev_logprobs = prev_logprobs
                    else:
                        # Otherwise, store the logprobs in prev_logprobs (the final logprobs used for training)
                        rollout_result.prev_logprobs = prev_logprobs
                elif self.role == 'critic':
                    values = infer_out
                    rollout_result.values = values
                else:
                    raise ValueError(f"Unknown role '{self.role}' for MegatronWorker")

                # Ref logprobs
                if compute_ref_logprobs:
                    assert self.ref_policy_state_dict is not None
                    with cpu_weight_swap(
                        self.model[0],
                        self.ref_policy_state_dict,
                        self.offload_model_buffer,
                    ):
                        ref_logprobs = self.inference_step(batch).cpu()
                        rollout_result.ref_logprobs = ref_logprobs
            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    self.put_result(split_result, output_channel)
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
                # print(f'run_inference: put_result, {len(split_result.prompt_ids)}')
                self.put_result(split_result, output_channel)
        assert total_num_samples == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {total_result_len}"
        )
        self.scheduler_offload_sync()
        if do_offload:
            # this is training, so we offload both weights and optimizer states
            self._offload_weight_and_optimizer()

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            prev_values = batch["values"].cuda() if "values" in batch else None
            prev_logprobs = batch["prev_logprobs"].cuda() if "prev_logprobs" in batch else None
            ref_logprobs = batch["ref_logprobs"].cuda() if "ref_logprobs" in batch else None

            if batch.get("advantages", None) is None:
                assert batch.get("returns", None) is None
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, returns = calculate_adv_and_returns(
                    task_type=self.cfg.runner.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    values=prev_values,
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0), # TODO use kl_beta or reinpp_kl_beta??
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=prev_logprobs,
                    ref_logprob=ref_logprobs,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                    gamma=self.cfg.algorithm.get("gamma", 1),
                    gae_lambda=self.cfg.algorithm.get("gae_lambda", 1),
                    # Normalization of advantages is done in run_training,
                    # so we set this argument to False here
                    normalize_advantages=False
                )
                batch["advantages"] = advantages
                batch["returns"] = returns

        return batch

    def del_reshard_state_dict(self):
        if hasattr(self, "reshard_state_dict"):
            del self.reshard_state_dict
            clear_memory()

    def _compute_rollout_metrics(self, batch):
        rollout_metrics_compute_data_group = self.get_rollout_metrics_group(batch)
        if rollout_metrics_compute_data_group is None:
            return None
        rollout_metrics, total_prompt_lengths, total_decode_lengths = (
            compute_rollout_metrics(
                batch,
                self.cfg.data.max_prompt_length,
                self.response_len,
                rollout_metrics_compute_data_group,
            )
        )

        rollout_metrics = cpu_dict(rollout_metrics)

        if self.role_cfg.get("calculate_flops", False):
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
