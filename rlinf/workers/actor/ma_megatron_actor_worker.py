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

from dataclasses import replace
from typing import Optional

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
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
)
from rlinf.data.schema.reasoning_results import (
    DynamicRolloutResult,
    GroupDelivery,
    RolloutResult,
)
from rlinf.hybrid_engines.megatron.utils import (
    postprocess_ragged_seqs,
    preprocess_ragged_seqs,
)
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics_dynamic,
    masked_normalization,
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_STAT_KEYS,
    VersionedTrajectoryRecord,
    compute_staleness_metrics,
)
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
from rlinf.workers.megatron_worker import DynamicMegatronBatchMixin
from rlinf.workers.rollout.session_server.assembly import (
    assemble_group_payload,
    verify_group_assembly,
)
from rlinf.workers.rollout.session_server.payload import payload_channel_for


class MAMegatronActor(DynamicMegatronBatchMixin, MegatronActor):
    """The class for running the actor training using Megatron."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role="actor"
    ):
        super().__init__(cfg, placement)
        self._init_auto_scheduler(role)
        # Set by run_training for the duration of a training call and read
        # by _assemble_delivery.
        self._payload_channels: Optional[list[Channel]] = None
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch
        self.placement = placement

        assert self.placement_mode in [
            PlacementMode.COLLOCATED,
            PlacementMode.DISAGGREGATED,
        ], (
            "Multi-agent actor supports only collocated or disaggregated "
            f"placement, got {self.placement_mode}"
        )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)

    def _receive_rollout_result(
        self, channel: Channel
    ) -> RolloutResult | DynamicRolloutResult:
        """Fetch one rollout result, assembling delivered groups.

        Overrides DynamicMegatronBatchMixin._receive_rollout_result, which
        is a plain channel.get(). When the session server is enabled, the
        rollout buffer puts a GroupDelivery on the channel instead of the
        group's data; it is assembled here, before get_step_batch balances
        the groups across data-parallel ranks and broadcasts them to the
        model-parallel ranks.
        """
        result = channel.get()
        if isinstance(result, GroupDelivery):
            result = self._assemble_delivery(result)
        return result

    def _assemble_delivery(self, delivery: GroupDelivery) -> DynamicRolloutResult:
        """Fetch and assemble the data of one delivered group.

        Args:
            delivery: The GroupDelivery the rollout buffer put on the
                channel. The group's data is in the payload channel named
                by its payload ref.

        Returns:
            The assembled result, with the group id and task name of the
            delivery. With ``session_server.verify_assembly`` on, it is
            checked against the reference data in the last payload part.
        """
        assert self._payload_channels is not None, (
            "a group arrived as a GroupDelivery but this actor has no "
            "payload channels: run_training must receive them when the "
            "session server is enabled."
        )
        channel = payload_channel_for(
            delivery.payload_ref.locator, self._payload_channels
        )
        payload = channel.get(key=delivery.payload_ref.key)
        # A keyed get leaves an empty queue under the key; without this
        # removal the payload channel keeps one empty queue per trained
        # group. The get above has already returned the data, so the removal
        # cannot overtake it, and a group is delivered once, so no other get
        # waits on this key.
        channel.remove_key(key=delivery.payload_ref.key, async_op=True)
        result = assemble_group_payload(
            payload, not self.cfg.algorithm.recompute_logprobs
        )
        result.group_id = delivery.group_id
        result.task_name = delivery.task_name
        verify = next(part.verify for part in payload.parts if part.last)
        if verify is not None:
            verify_group_assembly(result, verify, self.cfg)
        return result

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

                if self.cfg.algorithm.use_valid_token_scale:
                    loss_scale = (
                        mask.sum()
                        / self.global_valid_token
                        * parallel_state.get_data_parallel_world_size()
                        * self.num_microbatches
                    )
                    loss *= loss_scale.item()

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

    def run_training(
        self,
        input_channel: Channel,
        version: Optional[int] = None,
        payload_channels: Optional[list[Channel]] = None,
    ):
        """Run the training loop for the actor.

        Args:
            input_channel: The channel to read rollout results from.
            version: Weight version the actor had loaded at step start, used
                for staleness statistics; None skips them (synchronous
                training passes no version).
            payload_channels: The payload channels for groups delivered as
                payload refs instead of inline results; held for the
                duration of this call, where _assemble_delivery reads them.
        """
        self._payload_channels = payload_channels

        batch, rollout_results = self.get_step_batch(input_channel)
        fixed_ppo_targets = self.cfg.critic.get("use_critic_model", False)
        assert "recomputed_logprobs" in batch or "rollout_logprobs" in batch
        batch = self.compute_advantages_and_returns(batch)
        batch["loss_scales"] = [
            mask.to(dtype=torch.float32) for mask in batch["response_mask"]
        ]
        # Advantage normalization
        if self.cfg.algorithm.normalize_advantages and not fixed_ppo_targets:
            assert False, "not implemented in multi-agent"
            mask = batch["response_mask"][:, -self.response_len :]
            batch["advantages"] = masked_normalization(
                batch["advantages"],
                mask,
                group=parallel_state.get_data_parallel_group(),
            )

        # metrics
        rollout_metrics_group = self.get_rollout_metrics_group(batch)
        rollout_metrics = self._compute_rollout_metrics(batch, rollout_metrics_group)
        staleness_metrics = self._compute_staleness_metrics(
            rollout_results, version, rollout_metrics_group
        )
        if staleness_metrics is not None:
            if rollout_metrics is None:
                rollout_metrics = {}
            rollout_metrics.update(staleness_metrics)

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
            if fixed_ppo_targets:
                # PPO weights action tokens equally across folded turns.
                batch["loss_scales"] = [mask.float() for mask in batch["response_mask"]]
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
                if self.cfg.algorithm.use_valid_token_scale:
                    self._setup_valid_token_scale(global_batch)
                assert len(global_batch["input_ids"]) > 0, (
                    "Dynamic DP load balancing must produce nonempty actor minibatches."
                )
                global_batch_size_per_dp = len(global_batch["input_ids"])
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

        if fixed_ppo_targets:
            clear_memory()

        return rollout_metrics, training_metrics_list

    def _compute_rollout_metrics(self, batch, rollout_metrics_compute_data_group):
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

    def _extract_versioned_trajectory_records(
        self, rollout_results: list[DynamicRolloutResult]
    ) -> Optional[list[VersionedTrajectoryRecord]]:
        """Extract per-trajectory weight-version records from rollout results.

        Returns None when any result lacks versions (synchronous training).
        Group indices are the position of each result in the local list, so
        they are unique within a rank; `_compute_staleness_metrics` offsets
        them per rank after gathering.
        """
        records = []
        for result_index, rollout_result in enumerate(rollout_results):
            if (
                rollout_result is None
                or rollout_result.version_start is None
                or rollout_result.version_end is None
            ):
                return None
            num_trajectories = (
                max(rollout_result.idx_to_traj) + 1 if rollout_result.idx_to_traj else 0
            )
            earliest_versions = [None] * num_trajectories
            completion_versions = [None] * num_trajectories
            generated_tokens = [0] * num_trajectories
            call_counts = [0] * num_trajectories
            cross_version_calls = [0] * num_trajectories
            turn_rewards = [0.0] * num_trajectories
            turn_counts = [0] * num_trajectories
            token_weighted_version_start_sums = [0] * num_trajectories
            for turn_index, traj_index in enumerate(rollout_result.idx_to_traj):
                turn_tokens = rollout_result.response_lengths[turn_index]
                version_start = rollout_result.version_start[turn_index]
                version_end = rollout_result.version_end[turn_index]
                earliest_versions[traj_index] = (
                    version_start
                    if earliest_versions[traj_index] is None
                    else min(earliest_versions[traj_index], version_start)
                )
                completion_versions[traj_index] = (
                    version_end
                    if completion_versions[traj_index] is None
                    else max(completion_versions[traj_index], version_end)
                )
                generated_tokens[traj_index] += turn_tokens
                call_counts[traj_index] += 1
                cross_version_calls[traj_index] += int(version_start != version_end)
                turn_rewards[traj_index] += rollout_result.rewards[turn_index]
                turn_counts[traj_index] += 1
                token_weighted_version_start_sums[traj_index] += (
                    version_start * turn_tokens
                )
            for traj_index in range(num_trajectories):
                records.append(
                    VersionedTrajectoryRecord(
                        group_index=result_index,
                        earliest_version=earliest_versions[traj_index],
                        completion_version=completion_versions[traj_index],
                        generated_tokens=generated_tokens[traj_index],
                        call_count=call_counts[traj_index],
                        cross_version_calls=cross_version_calls[traj_index],
                        reward=turn_rewards[traj_index]
                        / max(turn_counts[traj_index], 1),
                        token_weighted_version_start_sum=(
                            token_weighted_version_start_sums[traj_index]
                        ),
                    )
                )
        return records

    def _compute_staleness_metrics(
        self,
        rollout_results: list[DynamicRolloutResult],
        version: Optional[int],
        dp_group,
    ) -> Optional[dict[str, float]]:
        """Compute staleness metrics over all DP ranks, or None to skip.

        MP ranks hold broadcast copies of the same results, so records are
        gathered only across the DP group and group indices are offset by
        rank to stay unique. If any DP rank lacks versions, all ranks skip
        (the has-version flag is gathered first).
        """
        if version is None or dp_group is None:
            return None

        local_records = self._extract_versioned_trajectory_records(rollout_results)
        dp_world_size = torch.distributed.get_world_size(dp_group)
        has_version_list = [None] * dp_world_size
        torch.distributed.all_gather_object(
            has_version_list, local_records is not None, group=dp_group
        )
        if not all(flag is True for flag in has_version_list):
            return None

        local_records = local_records or []
        records_by_rank = [None] * dp_world_size
        torch.distributed.all_gather_object(
            records_by_rank, local_records, group=dp_group
        )
        # Offset group indices by the number of records on preceding ranks
        # (deterministic on every rank) so groups from different DP ranks
        # stay distinct.
        all_records = []
        group_index_offset = 0
        for rank_records in records_by_rank:
            for record in rank_records:
                all_records.append(
                    replace(record, group_index=group_index_offset + record.group_index)
                )
            group_index_offset += len(rank_records)
        return compute_staleness_metrics(all_records, version)

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
        batch, results = self.get_step_batch(input_channel)
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
        if self.cfg.critic.get("use_critic_model", False):
            assert batch.get("advantages") is not None, (
                "Critic-owned PPO advantages are missing from the actor batch."
            )
            return batch
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
