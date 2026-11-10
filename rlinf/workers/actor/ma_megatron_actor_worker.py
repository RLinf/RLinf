# Copyright 2026 The RLinf Authors.
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

from typing import Optional

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.loss_scales import restore_source_dp_normalization
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_loss_scales,
    policy_loss,
)
from rlinf.algorithms.utils import kl_penalty
from rlinf.data.schema.actor_batch import (
    DynamicActorBatch,
    pad_actor_batch_rows,
)
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.hybrid_engines.megatron.utils import (
    postprocess_ragged_seqs,
    preprocess_ragged_seqs,
)
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    balance_rollout_groups,
    broadcast_tensor_within_pp,
    compute_rollout_metrics_dynamic,
    masked_normalization,
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.metric_utils import CRITIC_EXPLAINED_VARIANCE_STAT_KEYS
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
    PlacementMode,
)
from rlinf.utils.utils import (
    clear_memory,
    configure_batch_sizes,
    cpu_dict,
    cpu_weight_swap,
)
from rlinf.workers.actor.megatron_actor_worker import (
    MegatronActor,
)


class MAMegatronActor(MegatronActor):
    """The class for running the actor training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role="actor"
    ):
        super().__init__(cfg, placement)
        self._init_auto_scheduler(role)
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch
        self.placement = placement

        assert self.placement_mode == PlacementMode.COLLOCATED, (
            "Only collocated placement is supported for multi-agent actor"
        )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)

    def get_batch(
        self, channel: Channel
    ) -> tuple[DynamicActorBatch, list[DynamicRolloutResult]]:
        """Balance raw groups once, then broadcast them to model-parallel peers."""
        count = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )
        results = None
        if channel.is_local or self.is_data_io_rank:
            results = [channel.get() for _ in range(count)]
            if not channel.is_local:
                for result in results:
                    result.extra_fields_train["source_dp"] = [
                        parallel_state.get_data_parallel_rank()
                    ] * result.num_sequence
            if not channel.is_local and self.enable_dp_load_balance:
                with self.worker_timer("ragged_dp_balance"):
                    results = balance_rollout_groups(
                        results, parallel_state.get_data_parallel_group()
                    )
        if not channel.is_local:
            with self.worker_timer("ragged_model_broadcast"):
                results = self.broadcast(
                    results,
                    groups=[
                        (self._group_name, parallel_state._MODEL_PARALLEL_GLOBAL_RANKS)
                    ],
                )
                results = self.broadcast(
                    results,
                    groups=[
                        (
                            self._group_name,
                            parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS,
                        )
                    ],
                )
        batch = DynamicRolloutResult.merge_batches(
            [result.to_actor_batch() for result in results],
            self.cfg.algorithm.group_size,
        )
        return batch, results

    def _restore_source_normalization(
        self, batch: DynamicActorBatch, scale_context: dict
    ) -> None:
        """Retain the original source-DP loss weights after early redistribution."""
        restore_source_dp_normalization(
            batch,
            scale_context,
            parallel_state.get_data_parallel_group(),
            self.torch_platform.current_device(),
        )

    def _pad_ragged_rows(self, batch: DynamicActorBatch) -> DynamicActorBatch:
        """Align every column with the minibatch and microbatch schedule."""
        return pad_actor_batch_rows(
            batch,
            self.num_train_steps * self.cfg.actor.micro_batch_size,
            parallel_state.get_data_parallel_group(),
            self.torch_platform.current_device(),
        )

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            rows = [
                row.to(self.torch_platform.current_device())
                for row in batch["input_ids"]
            ]
            lengths = [len(row) for row in rows]
            input_ids, packed_params = preprocess_ragged_seqs(rows)
            position_ids, _ = preprocess_ragged_seqs(
                [torch.arange(n, device=input_ids.device) for n in lengths]
            )
            labels = [torch.cat((row[1:], row.new_zeros(1))) for row in rows]
            label, _ = preprocess_ragged_seqs(labels)
            label_mask, _ = preprocess_ragged_seqs(
                [torch.arange(n, device=input_ids.device) < n - 1 for n in lengths]
            )

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
                None,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                temperature=self.cfg.algorithm.sampling_params.temperature,
                packed_seq_params=packed_params,
            )

            if unwrap_model(model).post_process:
                output_rows = {
                    key: postprocess_ragged_seqs(value, packed_params, lengths)
                    for key, value in output.items()
                }
                if not self.return_loss:
                    logprob_rows = [
                        torch.cat((row.new_zeros(1), row[:-1])).detach().cpu()
                        for row in output_rows["log_probs"]
                    ]
                    # Megatron inspects the tensor device before collecting non-loss data.
                    return (
                        output["log_probs"],
                        lambda output, non_loss_data=True: logprob_rows,
                    )
                width = self.cfg.actor.model.encoder_seq_length
                output = DynamicRolloutResult.materialize_actor_batch(
                    output_rows, width, 0, input_ids.device
                )
                batch = DynamicRolloutResult.materialize_actor_batch(
                    {
                        key: batch[key]
                        for key in (
                            "advantages",
                            "loss_scales",
                            "response_mask",
                            "rollout_logprobs",
                            "recomputed_logprobs",
                            "ref_logprobs",
                        )
                        if key in batch
                    },
                    width,
                    self.tokenizer.eos_token_id,
                    input_ids.device,
                )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output

                return output, id_func

            def loss_func(output):
                curr_logprobs_ori = output["log_probs"]
                curr_logprobs = curr_logprobs_ori.clone()
                curr_logprobs[:, 1:] = curr_logprobs_ori[:, :-1].clone()
                curr_logprobs[:, 0] = 0.0  # first token has no previous token

                advantages = batch["advantages"]
                advantages *= batch["loss_scales"]
                # Prefer recomputed_logprobs, fallback to rollout_logprobs
                old_logprobs = batch.get("recomputed_logprobs")
                if old_logprobs is None:
                    old_logprobs = batch["rollout_logprobs"]
                ref_logprobs = None
                if "ref_logprobs" in batch:
                    ref_logprobs = batch["ref_logprobs"]

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    assert False, (
                        "importance_sampling_fix is not supported for dynamic rollout batch"
                    )
                    rollout_logprobs = batch["rollout_logprobs"]
                    recomputed_logprobs = batch.get("recomputed_logprobs")
                    advantages = advantages * torch.clamp(
                        (recomputed_logprobs - rollout_logprobs).exp(),
                        max=self.cfg.algorithm.importance_sampling_clip,
                    )

                mask = batch["response_mask"]

                loss, metrics_data = policy_loss(
                    task_type=self.cfg.runner.task_type,
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=curr_logprobs,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    clip_ratio_c=self.clip_ratio_c,
                    clip_ratio_low=self.clip_ratio_low,
                    clip_ratio_high=self.clip_ratio_high,
                    loss_mask=mask,
                    clip_log_ratio_min=self.cfg.algorithm.get(
                        "clip_log_ratio_min", None
                    ),
                    clip_log_ratio_max=self.cfg.algorithm.get(
                        "clip_log_ratio_max", None
                    ),
                    fast_path_zero_loss_mask=True,
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
                    kl_loss = self.loss_agg_func(kld * batch["loss_scales"], mask)
                    loss = loss + kl_loss * self.kl_beta

                # all gather
                _imp: torch.Tensor = metrics_data["actor/ratio"].clone()
                torch.distributed.all_reduce(
                    _imp,
                    torch.distributed.ReduceOp.AVG,
                    group=parallel_state.get_data_parallel_group(),
                )

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

                # add to log
                metrics_data.update(
                    {
                        "actor/final_loss": loss.detach(),
                        "actor/entropy_loss": entropy_loss.detach(),
                        "actor/kl_loss": kl_loss.detach(),
                    }
                )

                for k in sorted(metrics_data.keys()):
                    v = metrics_data[k]
                    if v is None:
                        continue
                    if k in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS:
                        v = v.detach().clone()
                        torch.distributed.all_reduce(
                            v,
                            op=torch.distributed.ReduceOp.SUM,
                            group=parallel_state.get_data_parallel_group(),
                        )
                        metrics_data[k] = v
                    else:
                        metrics_data[k] = average_losses_across_data_parallel_group([v])

                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func

    def _get_num_microbatches(
        self, batch: DynamicActorBatch, forward_only: bool
    ) -> int:
        """Include partial inference microbatches; retain the training schedule."""
        if not forward_only:
            return super()._get_num_microbatches(batch, forward_only)
        micro_batch_size = self.logprob_forward_micro_batch_size
        return max(
            1, (len(batch["input_ids"]) + micro_batch_size - 1) // micro_batch_size
        )

    def run_forward_backward(self, batch: DynamicActorBatch, forward_only=True):
        """Run the forward and backward pass on the model.

        Args:
            batch (dict): Real-length token rows and per-sequence metadata.
            forward_only (bool): If True, only run the forward pass without backpropagation.
        """
        clear_memory()
        row_count = len(batch["input_ids"])
        if forward_only:
            vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            divisor = (
                self.transformer_config.microbatch_group_size_per_vp_stage
                if vpp_size is not None and vpp_size > 1
                else 1
            )
            batch = pad_actor_batch_rows(
                batch,
                divisor,
                parallel_state.get_data_parallel_group(),
                self.torch_platform.current_device(),
            )
        if self.enable_dynamic_batch_size:
            alignment = parallel_state.get_tensor_model_parallel_world_size() * (
                2 * parallel_state.get_context_parallel_world_size()
                if parallel_state.get_context_parallel_world_size() > 1
                else 1
            )
            batch_iter, total_seqlen, num_microbatches, self.dbs_indices = (
                split_dynamic_batch_size(
                    batch=batch,
                    cp_world_size=parallel_state.get_context_parallel_world_size(),
                    vpp_world_size=parallel_state.get_virtual_pipeline_model_parallel_world_size(),
                    max_tokens_per_mbs=self.max_tokens_per_mbs,
                    microbatch_group_size_per_vp_stage=self.transformer_config.microbatch_group_size_per_vp_stage,
                    seq_len_effective=(
                        (
                            torch.tensor(list(map(len, batch["input_ids"])))
                            + alignment
                            - 1
                        )
                        // alignment
                        * alignment
                    ),
                )
            )
        else:
            total_seqlen = sum(map(len, batch["input_ids"]))
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

        return outputs[:row_count] if forward_only else outputs

    def _process_fwd_bwd_outputs(self, forward_outputs, forward_only):
        if not forward_only:
            return super()._process_fwd_bwd_outputs(forward_outputs, forward_only)
        rows = [row for microbatch in forward_outputs for row in microbatch]
        if self.enable_dynamic_batch_size:
            indices = sum(self.dbs_indices, [])
            restored = [None] * len(rows)
            for index, row in zip(indices, rows):
                restored[index] = row
            rows = restored
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            return rows
        lengths = broadcast_tensor_within_pp(
            torch.tensor([len(row) for row in rows], dtype=torch.int64)
            if rows
            else None
        )
        values = broadcast_tensor_within_pp(torch.cat(rows) if rows else None)
        return list(values.cpu().split(lengths.cpu().tolist()))

    def _setup_valid_token_scale(self, batch: Optional[DynamicActorBatch] = None):
        if batch is None:
            self.global_valid_token = (
                self.average_response_len
                * get_num_microbatches()
                * self.cfg.actor.micro_batch_size
            )
        else:
            loss_mask = batch["response_mask"]
            global_valid_token = (
                torch.stack([row.sum() for row in loss_mask])
                .sum()
                .to(self.torch_platform.current_device(), dtype=torch.float32)
            )
            torch.distributed.all_reduce(
                global_valid_token, group=parallel_state.get_data_parallel_group()
            )
            self.global_valid_token = global_valid_token
            return batch

    def run_training(self, input_channel: Channel):
        """Run the training loop for the actor."""
        assert not self.is_pipeline

        batch, _ = self.get_batch(input_channel)
        assert "recomputed_logprobs" in batch or "rollout_logprobs" in batch
        # Compute advantages and returns
        batch = self.compute_advantages_and_returns(batch)
        batch["loss_scales"] = [
            mask.to(dtype=torch.float32) for mask in batch["response_mask"]
        ]
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

        # pack traj
        scale_context = {
            "folding_scale": [],
            "enable_scale_of_group": False,
            "actor_global_batch_size": (
                self.cfg.data.rollout_batch_size
                * self.cfg.algorithm.get("group_size", 1)
                / self.cfg.algorithm.n_minibatches
            ),
            "data_parallel_world_size": parallel_state.get_data_parallel_world_size(),
        }
        for loss_scale_fn in self.loss_scale_fns:
            batch = loss_scale_fn(scale_context, batch)
        if self.pack_traj:
            batch = DynamicRolloutResult.pack_traj_batch(scale_context, batch)
        self._restore_source_normalization(batch, scale_context)
        for key in list(batch.keys()):
            if key == "idx_to_traj" or key.startswith("extra:"):
                batch.pop(key, None)

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()
        self._training_setup()

        batch = self._pad_ragged_rows(batch)

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
                global_batch_size_per_dp = len(global_batch["input_ids"])
                if global_batch_size_per_dp == 0:
                    continue
                dp_size = parallel_state.get_data_parallel_world_size()
                configure_batch_sizes(
                    rank=torch.distributed.get_rank(),
                    mbs=self.cfg.actor.micro_batch_size,
                    gbs=global_batch_size_per_dp * dp_size,
                    dp=dp_size,
                )
                num_microbatches = get_num_microbatches()
                assert (
                    num_microbatches * self.cfg.actor.micro_batch_size
                    == global_batch_size_per_dp
                )
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        # Gather weights if overlap_param_gather before the next weight sync
        self._gather_weights_among_dp()

        return rollout_metrics, training_metrics_list

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
        batch, results = self.get_batch(input_channel)
        self._load_weight_and_optimizer()
        with self.worker_timer():
            probabilities = self.inference_step(batch)
            reference = None
            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "ref_policy_state_dict must be set to compute ref_logprobs"
                )
                with cpu_weight_swap(self.model[0], self.ref_policy_state_dict):
                    reference = self.inference_step(batch)
        offset = 0
        for result in results:
            end = offset + result.num_sequence
            result.recomputed_logprobs = probabilities[offset:end]
            if reference is not None:
                result.ref_logprobs = reference[offset:end]
            self.put_result(result, output_channel)
            offset = end
        self.scheduler_offload_sync()

    # Advantages and returns
    def compute_advantages_and_returns(
        self, batch: DynamicActorBatch
    ) -> DynamicActorBatch:
        """Compute the advantages and returns.

        Args:
            batch: Real-length token rows and per-sequence metadata.
        """
        with self.worker_timer():
            if batch.get("advantages") is None:
                device = self.torch_platform.current_device()
                logprobs = batch.get("recomputed_logprobs")
                if logprobs is None:
                    logprobs = batch["rollout_logprobs"]
                reference = batch.get("ref_logprobs")
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.cfg.runner.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    advantage_mode=self.cfg.algorithm.advantage_mode,
                    rewards=batch["rewards"].to(device),
                    loss_mask=[row.to(device) for row in batch["response_mask"]],
                    num_sequence=len(batch["input_ids"]),
                    sequence_length=self.cfg.actor.model.encoder_seq_length,
                    group_size=self.cfg.algorithm.group_size,
                    idx_to_traj=batch["idx_to_traj"],
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=[row.to(device) for row in logprobs],
                    ref_logprob=[row.to(device) for row in reference]
                    if reference is not None
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = [row.cpu() for row in advantages]
        return batch
