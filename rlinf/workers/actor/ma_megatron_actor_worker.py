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
from typing import Optional

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    policy_loss,
)
from rlinf.algorithms.utils import kl_penalty
from rlinf.data.io_struct import (
    DynamicRolloutResult,
    get_seq_length,
)
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    compute_rollout_metrics_dynamic,
    masked_normalization,
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import (
    clear_memory,
    configure_batch_sizes,
    cpu_dict,
    cpu_weight_swap,
)
from rlinf.workers.actor.megatron_actor_worker import (
    MegatronActor,
)
from rlinf.workers.rollout.utils import CollocateRankMapper, DisaggRankMapper


class MAMegatronActor(MegatronActor):
    """The class for running the actor training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role="actor"
    ):
        super().__init__(cfg, placement)
        self._init_auto_scheduler(role)
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch
        assert self.enable_dp_load_balance, (
            "enable_dp_load_balance must be True when is_dynamic_rollout_batch is True"
        )

        self.use_sub_worker = self.cfg.rollout.get(
            "use_sub_worker", False
        )  # FIXME: check
        assert self.placement_mode == PlacementMode.COLLOCATED
        self.train_roles = self.cfg.agentloop.get("train_roles", None)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], DynamicRolloutResult]:
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            result: DynamicRolloutResult = channel.get()
        else:
            if self.is_data_io_rank:
                result: DynamicRolloutResult = channel.get()
            else:
                result = None
            result = self.broadcast(
                result,
                groups=[
                    (self._group_name, parallel_state._MODEL_PARALLEL_GLOBAL_RANKS)
                ],
            )
            result = self.broadcast(
                result,
                groups=[
                    (self._group_name, parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS)
                ],
            )

        batch = result.to_actor_batch(
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )

        return batch, result

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            batch = {
                key: val.cuda() if torch.is_tensor(val) else val
                for key, val in batch.items()
            }

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            # In DynamicRolloutResult, we need to shift the labels to align with model outputs
            # Create labels by shifting input_ids to the left (next token prediction)
            label = copy.deepcopy(position_ids)
            label[:, :-1] = input_ids[:, 1:]

            assert "response_mask" in batch, (
                "response_mask must be provided in dynamic rollout batch"
            )
            # response_mask needs to be shifted left by 1 to align with the shifted labels
            # because response_mask[i] indicates whether position i is a response token in input_ids
            # but label[i] = input_ids[i+1], so we need to shift response_mask left to align
            label_mask = copy.deepcopy(batch["attention_mask"])
            label_mask[:, :-1] = label_mask[:, 1:].clone()  # Shift left by 1
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
                    # breakpoint()

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
                    return output

                # in last stage need to get the log_probs from the output
                if unwrap_model(model).post_process:
                    output = output["log_probs"]

                return output, id_func

            def loss_func(output):
                curr_logprobs_ori = output["log_probs"]
                curr_logprobs = curr_logprobs_ori.clone()
                curr_logprobs[:, 1:] = curr_logprobs_ori[:, :-1].clone()
                curr_logprobs[:, 0] = 0.0  # first token has no previous token

                # curr_logprobs = output["log_probs"]

                advantages = batch["advantages"]
                prev_logprobs = batch["prev_logprobs"]
                ref_logprobs = None
                if "ref_logprobs" in batch:
                    ref_logprobs = batch["ref_logprobs"]

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    assert False, (
                        "importance_sampling_fix is not supported for dynamic rollout batch"
                    )
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                mask = batch["response_mask"]

                loss, metrics_data = policy_loss(
                    task_type=self.cfg.runner.task_type,
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=curr_logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=self.clip_ratio_low,
                    clip_ratio_high=self.clip_ratio_high,
                    loss_mask=mask,
                )

                entropy_loss = torch.zeros(1, device=loss.device)
                if self.calculate_entropy:
                    entropy = output["entropy"]
                    entropy_loss = self.loss_agg_func(entropy, mask=mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(curr_logprobs, ref_logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, mask)
                    loss = loss + kl_loss * self.kl_beta

                # all gather
                _imp = copy.deepcopy(metrics_data["actor/ratio"])
                torch.distributed.all_reduce(
                    _imp, group=parallel_state.get_data_parallel_group()
                )

                _n_valid_tokens = mask.count_nonzero().clone()
                torch.distributed.all_reduce(
                    _n_valid_tokens, group=parallel_state.get_data_parallel_group()
                )
                if _n_valid_tokens.item() != 0:
                    _imp /= _n_valid_tokens

                # Early stopping.
                if (
                    self.cfg.algorithm.early_stop_imp_ratio is not None
                    and _imp > self.cfg.algorithm.early_stop_imp_ratio
                ):
                    # breakpoint()
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

                assert self.cfg.algorithm.get("use_rollout_scale", False)
                # make sure normalize across groupsize
                num_microbatches = (
                    get_num_microbatches()
                    * parallel_state.get_data_parallel_world_size()
                )
                if self.train_roles is None:
                    loss_scale = num_microbatches / (
                        self.cfg.data.rollout_batch_size
                        * self.cfg.algorithm.get("group_size", 1)
                        / self.cfg.algorithm.n_minibatches
                    )
                else:
                    # groupsize is dynamic, and will be set in pack traj fuction
                    loss_scale = num_microbatches / (
                        self.cfg.data.rollout_batch_size
                        / self.cfg.algorithm.n_minibatches
                    )
                loss *= loss_scale

                # add to log
                metrics_data.update(
                    {
                        "final_loss": loss.detach(),
                        "entropy_loss": entropy_loss.detach(),
                        "kl_loss": kl_loss.detach(),
                    }
                )

                for k in sorted(metrics_data.keys()):
                    v = metrics_data[k]
                    if v is not None:
                        metrics_data[k] = average_losses_across_data_parallel_group([v])

                # for k, v in metrics_data.items():
                #     if v is not None:
                #         metrics_data[k] = average_losses_across_data_parallel_group([v])

                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func

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
            batch_iter = get_iterator_k_split(
                batch, num_splits=num_microbatches, enforce_divisible_batch=False
            )

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

    def _setup_valid_token_scale(self, batch: Optional[dict[str, torch.Tensor]] = None):
        if batch is None:
            self.global_valid_token = (
                self.average_response_len
                * get_num_microbatches()
                * self.cfg.actor.micro_batch_size
            )
        else:
            loss_mask = batch["response_mask"]
            global_valid_token = loss_mask.to(dtype=torch.float32).sum().cuda()
            torch.distributed.all_reduce(
                global_valid_token, group=parallel_state.get_data_parallel_group()
            )
            self.global_valid_token = global_valid_token
            return batch

    def _dp_load_balance_dynamic(
        self, batch: dict[str, torch.Tensor], batch_pad, split_fix_chunk
    ):
        batch = RolloutDataBalance.from_rollout_batches_dynamic(
            rollout_batches=batch,
            dp_world_size=parallel_state.get_data_parallel_world_size(),
            dp_rank=parallel_state.get_data_parallel_rank(),
            dp_group=parallel_state.get_data_parallel_group(),
            rollout_batch_pad=batch_pad,
            split_fix_chunk=split_fix_chunk,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(self, input_channel: Channel):
        """Run the training loop for the actor."""
        assert not self.is_pipeline

        # Get all batches for this DP
        batches = []
        total_result_size_per_dp: int = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )
        for _ in range(total_result_size_per_dp):
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
        batch = DynamicRolloutResult.merge_batches(
            batches, self.cfg.algorithm.group_size
        )
        assert "prev_logprobs" in batch
        # Compute advantages and returns
        batch = self.compute_advantages_and_returns(batch)
        # Advantage normalization
        if self.cfg.algorithm.normalize_advantages:
            assert False, "not implemented in multi-agent"
            mask = batch["response_mask"][:, -self.response_len :]
            batch["advantages"] = masked_normalization(
                batch["advantages"],
                mask,
                group=parallel_state.get_data_parallel_group(),
            )

        # metrics
        rollout_metrics = self._compute_rollout_metrics(batch)
        # eval_metrics = batch.pop("eval_metrics")

        # pack traj
        assert self.cfg.actor.pack_traj
        #####
        adv_turn_level_scale = self.cfg.algorithm.get(
            "adv_turn_level_scale", True
        )  # whether use alg1 or alg2
        sub_traj_adv_scale = self.cfg.algorithm.get("sub_traj_adv_scale", False)
        assert sub_traj_adv_scale
        batch = DynamicRolloutResult.pack_traj_batch(
            batch, adv_turn_level_scale, sub_traj_adv_scale, self.train_roles
        )
        #####

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()
        self._training_setup()

        # Valid token scale
        if self.cfg.algorithm.use_valid_token_scale:
            self._setup_valid_token_scale(batch)

        # DP batch load balance
        if self.cfg.actor.get("enable_dp_load_balance", False):
            batch_pad = DynamicRolloutResult.get_batch_pad(
                self.cfg.actor.model.encoder_seq_length
            )
            batch = self._dp_load_balance_dynamic(
                batch, batch_pad, self.cfg.actor.micro_batch_size
            )

        # batch_size = batch["response_mask"].shape[0]
        # if self._rank % 2 == 0:
        #     num_splits = [
        #         batch_size // self.num_train_steps
        #         + int(i < (batch_size % self.num_train_steps))
        #         for i in range(self.num_train_steps)
        #     ]
        # else:
        #     num_splits = [
        #         batch_size // self.num_train_steps
        #         + int(i >= (self.num_train_steps - batch_size % self.num_train_steps))
        #         for i in range(self.num_train_steps)
        #     ]

        # global_batches = get_iterator_k_split(
        #     batch,
        #     num_splits=num_splits,
        #     shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
        #     shuffle_seed=self.cfg.actor.seed,
        # )

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.num_train_steps,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )
        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        # Global batch iterations
        with self.worker_timer("run_training"):
            training_metrics_list = []
            for idx, global_batch in enumerate(global_batches):
                if global_batch["input_ids"].shape == torch.Size([0]):
                    continue
                global_batch_size_per_dp = global_batch["input_ids"].shape[0]
                dp_size = parallel_state.get_data_parallel_world_size()
                configure_batch_sizes(
                    rank=torch.distributed.get_rank(),
                    mbs=self.cfg.actor.micro_batch_size,
                    gbs=global_batch_size_per_dp * dp_size,
                    dp=dp_size,
                )
                num_microbatches = get_num_microbatches()
                assert num_microbatches == global_batch_size_per_dp
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"]  # [num_sequence, seq_len]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.cfg.runner.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    advantage_mode=self.cfg.algorithm.advantage_mode,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    num_sequence=len(batch["input_ids"]),
                    group_size=self.cfg.algorithm.group_size,
                    idx_to_traj=batch["idx_to_traj"],
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
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

    def _setup_rollout_weight_dst_ranks(self):
        """Setup destination ranks for token and weight communication."""
        assert self.placement_mode == PlacementMode.COLLOCATED
        if not self.use_sub_worker:
            rank_mapper = CollocateRankMapper
        else:
            rank_mapper = DisaggRankMapper
        rank_map = rank_mapper.get_actor_rank_to_rollout_rank_map(
            self.component_placement.actor_tp_size,
            self.component_placement.actor_pp_size,
            self.component_placement.actor_world_size,
            self.component_placement.rollout_tp_size,
            self.component_placement.rollout_world_size,
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def sync_model_to_rollout(self):
        """Send the model weights to the destination ranks in the rollout task."""
        if not self.is_running:
            return

        model_bucket_list = self.divide_model_to_bucket()
        if not hasattr(self, "sync_model_bucket_length"):
            self.sync_model_bucket_length = len(model_bucket_list)
        else:
            assert self.sync_model_bucket_length == len(model_bucket_list), (
                f"last sync_model_bucket_length {self.sync_model_bucket_length} don't equal now the len(model_bucket_list) {len(model_bucket_list)}"
            )
            assert self.sync_model_bucket_length != 0, (
                "error the self.sync_model_bucket_length is 0"
            )

        self.model_state_offload_optimizer_and_grad()

        # send bucket size
        if len(self._weight_dst_rank_in_rollout) > 0:
            if (
                self.placement_mode == PlacementMode.COLLOCATED
                and not self.use_sub_worker
            ):
                send_handle = None
                for bucket_weight in model_bucket_list:
                    reshard_state_dict = self._get_rollout_model_state_dict(
                        bucket_weight
                    )
                    buffer = {
                        k: reduce_tensor(v) for k, v in reshard_state_dict.items()
                    }
                    if send_handle is not None:
                        send_handle.wait()
                    else:
                        # add the bucket_length message in bucket 0
                        buffer["bucket_length"] = len(model_bucket_list)
                    send_handle = self.send(
                        buffer,
                        self.rollout_group_name,
                        self._weight_dst_rank_in_rollout,
                        async_op=True,
                    )
                    del reshard_state_dict
                send_handle.wait()
            else:
                send_handle_bucket = []
                for bucket_weight in model_bucket_list:
                    reshard_state_dict = self._get_rollout_model_state_dict(
                        bucket_weight
                    )

                    if len(send_handle_bucket) != 0:
                        for send_handle in send_handle_bucket:
                            send_handle.wait()
                        send_handle_bucket = []
                    else:
                        # add the bucket_length message in bucket 0
                        reshard_state_dict["bucket_length"] = len(model_bucket_list)

                    for weight_dst_rank in self._weight_dst_rank_in_rollout:
                        send_handle = self.send(
                            reshard_state_dict,
                            self.rollout_group_name,
                            weight_dst_rank,
                            async_op=True,
                        )
                        send_handle_bucket.append(send_handle)

                if len(send_handle_bucket) != 0:
                    for send_handle in send_handle_bucket:
                        send_handle.wait()

        if (
            self.placement_mode == PlacementMode.COLLOCATED
            or self.use_pre_process_policy
        ):
            if self.offload_weight:
                self.offload_model_weights_and_grad(
                    offload_grad=False, offload_weight=True
                )
                self.is_weight_offloaded = True

    def _compute_rollout_metrics(self, batch):
        rollout_metrics_compute_data_group = self.get_rollout_metrics_group(batch)
        if rollout_metrics_compute_data_group is None:
            return None

        rollout_metrics, total_prompt_lengths, total_decode_lengths = (
            compute_rollout_metrics_dynamic(
                batch,
                self.cfg.data.max_prompt_length,
                self.response_len,
                rollout_metrics_compute_data_group,
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

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.
        Override to handle DynamicRolloutResult which has different structure.
        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        batches = []
        rollout_results = []
        total_batch_size_per_dp: int = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )

        for _ in range(total_batch_size_per_dp):
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            rollout_results.append(rollout_result)
        merged_batch, num_sequence_per_group = DynamicRolloutResult.merge_batches(
            batches, adjust_traj_indices=False, return_num_sequence_per_group=True
        )

        rollout_result = DynamicRolloutResult.merge_result_list(
            rollout_results,
        )

        self._load_weight_and_optimizer()
        with self.worker_timer():
            # compute prev logprobs
            prev_logprobs = self.inference_step(merged_batch)
            if rollout_result.rollout_logprobs is not None:
                rollout_result.recompute_prev_logprobs = prev_logprobs.cpu()
            else:
                rollout_result.prev_logprobs = prev_logprobs.cpu()
            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "ref_policy_state_dict must be set to compute ref_logprobs"
                )
                with cpu_weight_swap(self.model[0], self.ref_policy_state_dict):
                    ref_logprobs = self.inference_step(merged_batch)
                    rollout_result.ref_logprobs = ref_logprobs.cpu()

        rollout_result_per_group = DynamicRolloutResult.split_results(
            rollout_result, num_sequence_per_group
        )
        for rollout_result in rollout_result_per_group:
            self.put_result(rollout_result, output_channel)

        self.scheduler_offload_sync()
