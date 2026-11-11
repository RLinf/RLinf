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
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_loss_scales,
    policy_loss,
)
from rlinf.algorithms.utils import (
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
from rlinf.workers.actor.fsdp_actor_worker import (
    FSDPActor,
)


class MAFSDPActor(FSDPActor):
    """FSDP actor for collocated or disaggregated Search-R1 rollouts."""

    @staticmethod
    def validate_placement(
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ) -> None:
        """Validate deployment requirements before launching workers."""
        if not (placement.is_collocated or placement.is_disaggregated):
            raise ValueError(
                "MAFSDPActor requires collocated or disaggregated placement"
            )
        batch_size = cfg.data.rollout_batch_size
        if (
            batch_size < placement.actor_world_size
            or batch_size % placement.actor_world_size
        ):
            raise ValueError(
                f"rollout_batch_size ({batch_size}) must be a positive multiple "
                f"of actor DP world size ({placement.actor_world_size})"
            )
        if placement.is_disaggregated:
            if placement.has_dedicated_actor_inference:
                raise ValueError(
                    "Disaggregated MAFSDPActor computes logprobs on the actor; "
                    "remove inference from cluster.component_placement"
                )
            if cfg.actor.fsdp_config.strategy != "fsdp2":
                raise ValueError("Disaggregated MAFSDPActor requires FSDP2")
            if not cfg.algorithm.recompute_logprobs:
                raise ValueError(
                    "Disaggregated MAFSDPActor requires algorithm.recompute_logprobs=True"
                )

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        self.validate_placement(cfg, placement)
        super().__init__(cfg, placement, cfg_fsdp)
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch, (
            "MAFSDPActor requires agentloop.is_dynamic_rollout_batch=True"
        )
        assert self.enable_dp_load_balance, (
            "enable_dp_load_balance must be True when is_dynamic_rollout_batch is True"
        )
        self.placement = placement
        if self.cfg.algorithm.get("importance_sampling_fix", False):
            raise ValueError(
                "importance_sampling_fix is not supported for dynamic rollout batch"
            )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)

    def _get_num_rollout_results_per_dp(self) -> int:
        """Return the number of question groups consumed by each actor DP rank."""
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

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        # FSDP backward hooks attach to the model output before temperature scaling.
        if self.cfg.algorithm.sampling_params.temperature != 1.0:
            logits = logits / self.cfg.algorithm.sampling_params.temperature

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
        num_sequence = rollout_result.num_sequence
        max_num_sequence = torch.tensor(num_sequence, device=batch["input_ids"].device)
        torch.distributed.all_reduce(
            max_num_sequence, op=torch.distributed.ReduceOp.MAX
        )
        padded_num_sequence = int(max_num_sequence.item())
        micro_batch_size = self.cfg.algorithm.logprob_forward_micro_batch_size
        if not self.enable_dynamic_batch_size:
            padded_num_sequence = (
                (padded_num_sequence + micro_batch_size - 1)
                // micro_batch_size
                * micro_batch_size
            )
        # Equal row counts keep FSDP forward collectives aligned across DP ranks.
        indices_to_keep = list(range(num_sequence)) + [0] * (
            padded_num_sequence - num_sequence
        )
        batch = {
            key: value[indices_to_keep]
            if isinstance(value, torch.Tensor)
            else [value[i] for i in indices_to_keep]
            for key, value in batch.items()
        }
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=padded_num_sequence // micro_batch_size,
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

        return (
            recomputed_logprobs[:num_sequence],
            ref_logprobs[:num_sequence] if ref_logprobs is not None else None,
        )

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
            "MAFSDPActor requires actor-side inference without a pipeline"
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
                v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
            )

        # batch for forward
        logprobs, entropy = self.forward_batch(m_batch, True)

        # batch for backward
        old_logprobs = m_batch.get("recomputed_logprobs")
        if old_logprobs is None:
            old_logprobs = m_batch["rollout_logprobs"]
        loss_mask = self.get_response_loss_mask(m_batch)
        # Cancel microbatch means; loss weights already contain global normalization.
        if self.cfg.algorithm.loss_agg_func == "token-mean":
            token_counts = loss_mask.sum(dim=-1, keepdim=True)
            reduction_scale = token_counts.sum() / token_counts.clamp_min(1)
        else:
            reduction_scale = loss_mask.shape[0]
        loss_scales = (
            m_batch["loss_scales"] * reduction_scale * self.gradient_accumulation
        )
        advantages = m_batch["advantages"] * loss_scales
        ref_logprobs = None
        if "ref_logprobs" in m_batch:
            ref_logprobs = m_batch["ref_logprobs"]

        clip_ratio = self.cfg.algorithm.ratio_clip_eps
        clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
        clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
        clip_ratio_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
        clip_ratio_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio
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
            # Padding can precede valid rows and must retain a backward graph.
            fast_path_zero_loss_mask=False,
        )

        entropy_loss = torch.tensor(0.0, device=logprobs.device)
        if self.calculate_entropy:
            entropy_loss = self.loss_agg_func(entropy * loss_scales, mask=loss_mask)
            if self.calculate_entropy_loss:
                loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

        kl_loss = torch.tensor(0.0, device=logprobs.device)
        if self.kl_beta > 0 and ref_logprobs is not None:
            kld = kl_penalty(logprobs, ref_logprobs, self.kl_penalty_type)
            kl_loss = self.loss_agg_func(kld * loss_scales, loss_mask)
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
    ) -> tuple[dict, list]:
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )
        assert not self.is_pipeline, (
            "MAFSDPActor requires full-batch training without a pipeline"
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
        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        global_batch = self.compute_advantages_and_returns(global_batch)
        global_batch["loss_scales"] = torch.ones_like(
            global_batch["advantages"]
        ).masked_fill(~global_batch["response_mask"], 0)

        if self.cfg.algorithm.normalize_advantages:
            raise AssertionError(
                "normalize_advantages is not implemented in multi-agent"
            )

        rollout_metrics = self._compute_rollout_metrics(global_batch)

        scale_context = {
            "folding_scale": [],
            "group_scale_in_loss": True,
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
            )
            global_batch = self._dp_load_balance_dynamic(
                global_batch,
                batch_pad,
                self.cfg.actor.micro_batch_size * self.cfg.algorithm.n_minibatches,
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
                valid_tokens = (
                    mini_batch["response_mask"].sum().to(Worker.torch_device_type)
                )
                torch.distributed.all_reduce(valid_tokens)
                if valid_tokens.item() == 0:
                    continue
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)

            if training_metrics_list and not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    advantage_mode=self.cfg.algorithm.advantage_mode,
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
