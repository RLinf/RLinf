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
from omegaconf import DictConfig

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.distillation.losses import (
    DistillationReductionContext,
    combine_distillation_and_task_loss,
    compute_distillation_loss,
)
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_loss_scales,
    policy_loss,
)
from rlinf.algorithms.utils import (
    apply_importance_sampling_correction,
    kl_penalty,
)
from rlinf.data.schema.reasoning_results import (
    DynamicRolloutResult,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    compute_rollout_metrics_dynamic,
)
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    cpu_dict,
    cpu_weight_swap,
)
from rlinf.workers.actor.distillation import distillation_model_forward
from rlinf.workers.actor.fsdp_actor_worker import (
    FSDPActor,
)


class MAFSDPActor(FSDPActor):
    """FSDP actor for Search-R1 multi-agent dynamic rollouts.

    Each DP rank consumes complete groups before balancing the retained turns.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg, placement, cfg_fsdp)
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch, (
            "MAFSDPActor requires agentloop.is_dynamic_rollout_batch=True"
        )
        assert self.enable_dp_load_balance, (
            "enable_dp_load_balance must be True when is_dynamic_rollout_batch is True"
        )
        self.placement = placement
        assert self.placement.is_collocated or (
            self.distillation_plan.separate_async and self.placement.is_disaggregated
        ), "MAFSDPActor requires collocated placement or separate_async distillation"
        if self.cfg.algorithm.get("importance_sampling_fix", False):
            raise ValueError(
                "importance_sampling_fix is not supported for dynamic rollout batch"
            )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)

    def _get_num_rollout_results_per_dp(self) -> int:
        """Return the number of DynamicRolloutResult objects each DP rank should receive.

        For multi-agent dynamic rollout, each channel item corresponds to one trajectory
        group, while `num_sequence` inside that item can vary across turns. Therefore the
        receive loop should be counted in rollout-result units, not sequence units.
        """
        return self.cfg.data.rollout_batch_size // torch.distributed.get_world_size()

    def get_rollout_metrics_group(self, batch: dict[str, torch.Tensor]):
        """Return the process group used to aggregate rollout metrics."""
        return torch.distributed.group.WORLD

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], DynamicRolloutResult]:
        result: DynamicRolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def forward_batch(
        self,
        m_batch: dict[str, torch.Tensor],
        calculate_entropy: bool = False,
        *,
        distillation: bool = False,
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        seq_length = input_ids.shape[1]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch:
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = seq_length

            if "prompt_lengths" in m_batch and "response_lengths" in m_batch:
                seq_lens = m_batch["prompt_lengths"].to(input_ids.device) + m_batch[
                    "response_lengths"
                ].to(input_ids.device)
            else:
                # Fallback: DynamicRolloutResult attention_mask is True for
                # all valid prompt+response tokens.
                seq_lens = attention_mask.to(torch.long).sum(dim=1)

            idx_starts = torch.zeros_like(seq_lens, dtype=torch.long)
            idx_ends = seq_lens.to(torch.long)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        model_inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )
        if distillation:
            packed = self.enable_dynamic_batch_size
            if packed:
                starts, ends = idx_starts, idx_ends
            else:
                ends = m_batch["attention_mask"].sum(-1)
                starts = torch.zeros_like(ends)
            return distillation_model_forward(
                self,
                model_inputs,
                m_batch,
                packed=packed,
                starts=starts,
                ends=ends,
                response_only=False,
            )
        with self.amp_context:
            outputs = self.model(**model_inputs)
        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)

        if self.enable_dynamic_batch_size:
            # unpack_fsdp_logprobs is expected to return logprobs unpacked to
            # [B, max_seq_len_unpack], aligned with token positions.
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
        else:
            if input_ids.shape[1] <= 1:
                logprobs = torch.zeros(
                    input_ids.shape,
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                # token_logprobs[:, t] corresponds to input_ids[:, t + 1]
                token_logprobs = self.compute_logprobs(
                    logits[:, :-1, :],
                    input_ids[:, 1:],
                )

                logprobs = torch.zeros(
                    input_ids.shape,
                    dtype=token_logprobs.dtype,
                    device=token_logprobs.device,
                )
                logprobs[:, 1:] = token_logprobs

        if calculate_entropy:
            if self.enable_dynamic_batch_size:
                pos_entropy = compute_entropy_from_logits(logits)

                pos_entropy = unpack_sequences(
                    pos_entropy,
                    idx_starts,
                    idx_ends,
                    max_seq_len_unpack,
                    pad_val=0,
                )

                entropy = torch.zeros(
                    pos_entropy.shape,
                    dtype=pos_entropy.dtype,
                    device=pos_entropy.device,
                )
                if pos_entropy.shape[1] > 1:
                    entropy[:, 1:] = pos_entropy[:, :-1]
            else:
                if input_ids.shape[1] <= 1:
                    entropy = torch.zeros(
                        input_ids.shape,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                else:
                    token_entropy = compute_entropy_from_logits(logits[:, :-1, :])

                    entropy = torch.zeros(
                        input_ids.shape,
                        dtype=token_entropy.dtype,
                        device=token_entropy.device,
                    )
                    entropy[:, 1:] = token_entropy

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: DynamicRolloutResult,
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

        recomputed_logprobs, ref_logprobs = None, None

        # Recompute logprobs with the actor model.
        recomputed_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == recomputed_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {recomputed_logprobs.size(0)}"
            )
            recomputed_logprobs = recomputed_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(self, self.ref_policy_state_dict, is_fsdp=True):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return recomputed_logprobs, ref_logprobs

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
        assert not self.is_pipeline, (
            "MAFSDPActor currently only supports collocated inference"
        )

        batches = []
        rollout_results = []
        total_result_size_per_dp = self._get_num_rollout_results_per_dp()

        for _ in range(total_result_size_per_dp):
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            rollout_results.append(rollout_result)

        merged_batch, num_sequence_per_group = DynamicRolloutResult.merge_batches(
            batches, adjust_traj_indices=False, return_num_sequence_per_group=True
        )
        rollout_result = DynamicRolloutResult.merge_result_list(rollout_results)

        self._load_weight_and_optimizer()
        self.model.eval()
        with self.worker_timer():
            with torch.no_grad():
                recomputed_logprobs, ref_logprobs = self.inference_step(
                    merged_batch,
                    rollout_result,
                    compute_ref_logprobs,
                )

            rollout_result.recomputed_logprobs = recomputed_logprobs

            if compute_ref_logprobs:
                rollout_result.ref_logprobs = ref_logprobs

        rollout_result_per_group = DynamicRolloutResult.split_results(
            rollout_result, num_sequence_per_group
        )
        for rollout_result in rollout_result_per_group:
            output_channel.put(rollout_result)

    def get_response_loss_mask(self, m_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the full response mask used by dynamic multi-agent rollouts."""
        return m_batch["response_mask"]

    def forward_backward_batch(
        self,
        idx: int,
        m_batch: dict[str, torch.Tensor],
        micro_batch_cnt: int,
    ) -> dict[str, torch.Tensor]:
        """Run one dynamic-rollout micro-batch with trajectory loss scaling."""
        backward_ctx = self.before_micro_batch(
            self.model,
            is_last_micro_batch=(idx + 1) == micro_batch_cnt,
        )
        for k, v in m_batch.items():
            m_batch[k] = (
                v.to(Worker.torch_device_type)
                if isinstance(v, torch.Tensor) and k != "teacher_logprobs"
                else v
            )

        # batch for forward
        distillation_forward = None
        if self.distillation_plan.enabled:
            distillation_forward = self.forward_training_batch(m_batch)
            logprobs, entropy = (
                distillation_forward.logprobs,
                distillation_forward.entropy,
            )
        else:
            logprobs, entropy = self.forward_batch(m_batch, True)

        loss_mask = self.get_response_loss_mask(m_batch)
        ref_logprobs = m_batch.get("ref_logprobs")
        loss = None
        mbs_metrics_data = {}
        if (
            not self.distillation_plan.enabled
            or self.distillation_plan.need_task_advantages
        ):
            # batch for backward
            old_logprobs = m_batch.get("recomputed_logprobs")
            if old_logprobs is None:
                old_logprobs = m_batch["rollout_logprobs"]
            advantages = m_batch["advantages"] * m_batch["loss_scales"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = self.get_response_loss_mask(m_batch)

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

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

        if self.distillation_plan.enabled:
            aligned_batch = dict(m_batch)
            if self.distillation_plan.use_policy_gradient and self.cfg.algorithm.get(
                "importance_sampling_fix", False
            ):
                old = m_batch["recomputed_logprobs"]
                weights, _, _ = apply_importance_sampling_correction(
                    torch.ones_like(old),
                    old,
                    m_batch["rollout_logprobs"],
                    self.cfg.algorithm.importance_sampling_clip,
                )
                aligned_batch["rollout_is_weights"] = weights

            aligned_batch["response_mask"] = loss_mask
            aligned_batch["teacher_valid_mask"] = m_batch["teacher_valid_mask"].to(
                loss_mask.device
            )
            # Sampled targets are small; full targets were consumed in token chunks.
            if self.distillation_plan.target_format == "sampled":
                aligned_batch["teacher_logprobs"] = m_batch["teacher_logprobs"].to(
                    loss_mask.device
                )
            distill_loss, distill_metrics = compute_distillation_loss(
                distillation_forward,
                aligned_batch,
                plan=self.distillation_plan,
                config=self.distillation_loss_config,
                reduction=DistillationReductionContext(
                    self._distillation_denominator,
                    mode=self.cfg.algorithm.loss_agg_func,
                    dp_size=self._world_size,
                    gradient_divisor=self.gradient_accumulation,
                ),
            )
            loss = combine_distillation_and_task_loss(
                loss,
                distill_loss,
                use_task_rewards=self.distillation_plan.need_task_rewards,
                coefficient=self.distillation_loss_config.distillation_loss_coef,
            )
            mbs_metrics_data.update(distill_metrics)

        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.calculate_entropy:
            entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
            if self.calculate_entropy_loss:
                loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

        kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.kl_beta > 0 and ref_logprobs is not None:
            kld = kl_penalty(logprobs, ref_logprobs, self.kl_penalty_type)
            kl_loss = self.loss_agg_func(kld * m_batch["loss_scales"], loss_mask)
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

        return mbs_metrics_data

    def _dp_load_balance_dynamic(
        self,
        batch: dict[str, torch.Tensor],
        batch_pad: dict[str, torch.Tensor],
        split_fix_chunk: int,
    ):
        return RolloutDataBalance.from_rollout_batches_dynamic(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            rollout_batch_pad=batch_pad,
            split_fix_chunk=split_fix_chunk,
            partitioning_tool=get_seqlen_balanced_partitions,
        )

    def _compute_rollout_metrics(self, batch: dict[str, torch.Tensor]):
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
                    "training_tflops": inference_tflops * 3,
                }
            )
        return rollout_metrics

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list | None]:
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )
        assert not self.is_pipeline or self.distillation_plan.enabled, (
            "MAFSDPActor currently only supports collocated training"
        )

        batches = []
        total_result_size_per_dp = self._get_num_rollout_results_per_dp()
        for _ in range(total_result_size_per_dp):
            batch, _ = self.get_batch(input_channel)
            batches.append(batch)

        global_batch = DynamicRolloutResult.merge_batches(
            batches,
            self.cfg.algorithm.group_size,
        )
        if self.distillation_plan.enabled:
            global_batch = self.prepare_distillation_targets(global_batch)
            if global_batch is None:
                self.log_warning(
                    "Skipping update: an Actor rank has no retained sequences"
                )
                return {"skipped_empty_batch": 1.0}, None
        if self.distillation_plan.need_old_logprobs:
            assert (
                "recomputed_logprobs" in global_batch
                or "rollout_logprobs" in global_batch
            )

        global_batch = self.compute_advantages_and_returns(global_batch)
        global_batch["loss_scales"] = torch.ones_like(
            global_batch["response_mask"], dtype=torch.float32
        ).masked_fill(~global_batch["response_mask"], 0)

        if (
            self.distillation_plan.need_task_advantages
            and self.cfg.algorithm.normalize_advantages
        ):
            raise AssertionError(
                "normalize_advantages is not implemented in multi-agent"
            )

        rollout_metrics = self._compute_rollout_metrics(global_batch)

        scale_context = {
            "folding_scale": [],
            "loss_agg_func": self.cfg.algorithm.loss_agg_func,
            "enable_scale_of_group": False,
            "actor_global_batch_size": (
                self.cfg.data.rollout_batch_size
                * self.cfg.algorithm.get("group_size", 1)
                / self.cfg.algorithm.n_minibatches
            ),
            "data_parallel_world_size": torch.distributed.get_world_size(),
        }
        for loss_scale_fn in self.loss_scale_fns:
            global_batch = loss_scale_fn(scale_context, global_batch)
        if self.distillation_plan.enabled:
            global_batch["distillation_loss_scales"] = global_batch[
                "loss_scales"
            ].clone()
            if self.distillation_plan.need_task_advantages:
                global_batch["distillation_loss_scales"] *= scale_context.get(
                    "distillation_group_scale", 1.0
                )
        if self.pack_traj:
            global_batch = DynamicRolloutResult.pack_traj_batch(
                scale_context, global_batch
            )
        for key in list(global_batch.keys()):
            if key == "idx_to_traj" or key.startswith("extra:"):
                global_batch.pop(key, None)

        self._load_weight_and_optimizer()

        if self.enable_dp_load_balance:
            batch_pad = DynamicRolloutResult.get_batch_pad(
                self.cfg.actor.model.encoder_seq_length,
                list(global_batch.keys()),
                batch_template=global_batch,
            )
            global_batch = self._dp_load_balance_dynamic(
                global_batch,
                batch_pad,
                self.cfg.actor.micro_batch_size,
            )

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        training_metrics_list = []
        with self.worker_timer():
            for mini_batch in mini_batches:
                if mini_batch["input_ids"].shape == torch.Size([0]):
                    continue
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)

            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        if (
            self.distillation_plan.enabled
            and not self.distillation_plan.need_task_advantages
        ):
            return batch

        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    advantage_mode=self.cfg.algorithm.get("advantage_mode", "turn"),
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    num_sequence=len(batch["input_ids"]),
                    group_size=self.cfg.algorithm.group_size,
                    idx_to_traj=batch["idx_to_traj"],
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=logprob.to(Worker.torch_device_type),
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch
