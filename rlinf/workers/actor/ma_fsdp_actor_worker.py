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
from rlinf.algorithms.loss_scales import restore_source_dp_normalization
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_loss_scales,
    policy_loss,
)
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.data.schema.actor_batch import (
    DynamicActorBatch,
    TokenRows,
    pad_actor_batch_rows,
)
from rlinf.data.schema.reasoning_results import (
    DynamicRolloutResult,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    balance_rollout_groups,
    compute_rollout_metrics_dynamic,
)
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
)
from rlinf.utils.torch_functionals import batch_pad_to_fixed_len
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    cpu_dict,
    cpu_weight_swap,
)
from rlinf.workers.actor.fsdp_actor_worker import (
    FSDPActor,
)


class MAFSDPActor(FSDPActor):
    """FSDP actor for Search-R1 multi-agent dynamic rollouts.

    The actor requires collocated placement because dynamic rollout batches are
    consumed and trained within the same actor group.
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
        self.placement = placement
        assert self.placement.is_collocated, (
            "Only collocated placement is supported for multi-agent actor"
        )
        if self.cfg.algorithm.get("importance_sampling_fix", False):
            raise ValueError(
                "importance_sampling_fix is not supported for dynamic rollout batch"
            )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)

    def _get_num_rollout_results_per_dp(self) -> int:
        """Count question groups; each group can contain a different turn count."""
        return self.cfg.data.rollout_batch_size // torch.distributed.get_world_size()

    def get_rollout_metrics_group(
        self, batch: DynamicActorBatch
    ) -> torch.distributed.ProcessGroup:
        """Return the process group used to aggregate rollout metrics."""
        return torch.distributed.group.WORLD

    def get_batch(
        self, channel: Channel
    ) -> tuple[DynamicActorBatch, list[DynamicRolloutResult]]:
        """Receive complete groups, balance once, and retain real-length rows."""
        results = [channel.get() for _ in range(self._get_num_rollout_results_per_dp())]
        if not channel.is_local:
            for result in results:
                result.extra_fields_train["source_dp"] = [
                    torch.distributed.get_rank()
                ] * result.num_sequence
            if self.enable_dp_load_balance:
                with self.worker_timer("group_dp_balance"):
                    results = balance_rollout_groups(
                        results, torch.distributed.group.WORLD
                    )
        batch = DynamicRolloutResult.merge_batches(
            [result.to_actor_batch() for result in results],
            self.cfg.algorithm.group_size,
        )
        return batch, results

    @staticmethod
    def _split_to_micro_batch(
        batch: DynamicActorBatch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num: int,
    ):
        if enable_dynamic_batch_size:
            iterator, _, count, indices = split_dynamic_batch_size(
                batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
                seq_len_effective=torch.tensor(list(map(len, batch["input_ids"]))),
            )
            return iterator, count, indices
        return (
            get_iterator_k_split(batch, split_num, enforce_divisible_batch=False),
            split_num,
            None,
        )

    def forward_batch(
        self,
        m_batch: DynamicActorBatch,
        calculate_entropy: bool = False,
    ) -> TokenRows | tuple[TokenRows, TokenRows]:
        """Return probabilities and optional entropy aligned with each token row."""
        device = self.torch_platform.current_device()
        rows = [row.to(device) for row in m_batch["input_ids"]]
        positions = [row.to(device) for row in m_batch["position_ids"]]
        lengths = list(map(len, rows))
        idx_starts, idx_ends = [0] * len(rows), lengths
        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch:
            for key in m_batch["multi_modal_inputs"][0]:
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]], dim=0
                ).to(device)
        if self.enable_dynamic_batch_size:
            input_ids, position_ids, attention_mask = pack_fsdp_input(
                rows,
                positions,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=self.max_tokens_per_mbs,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            width = max(lengths)
            input_ids = batch_pad_to_fixed_len(rows, width, self.tokenizer.eos_token_id)
            position_ids = batch_pad_to_fixed_len(positions, width, 0)
            attention_mask = batch_pad_to_fixed_len(
                [torch.ones_like(row, dtype=torch.bool) for row in rows], width, 0
            )
        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )
        logits = outputs.logits
        logits = logits / self.cfg.algorithm.sampling_params.temperature
        if self.enable_dynamic_batch_size:
            probabilities = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=None,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = [torch.cat((row.new_zeros(1), row[1:])) for row in probabilities]
            if calculate_entropy:
                values = unpack_sequences(
                    compute_entropy_from_logits(logits),
                    idx_starts,
                    idx_ends,
                    None,
                    pad_val=0,
                )
                entropy = [torch.cat((row.new_zeros(1), row[:-1])) for row in values]
        else:
            values = self.compute_logprobs(logits[:, :-1, :], input_ids[:, 1:])
            logprobs = [
                torch.cat((row.new_zeros(1), row[: n - 1]))
                for row, n in zip(values, lengths)
            ]
            if calculate_entropy:
                values = compute_entropy_from_logits(logits[:, :-1, :])
                entropy = [
                    torch.cat((row.new_zeros(1), row[: n - 1]))
                    for row, n in zip(values, lengths)
                ]
        return (logprobs, entropy) if calculate_entropy else logprobs

    def inference_step(
        self,
        batch: DynamicActorBatch,
        compute_ref_logprobs: bool,
    ) -> tuple[TokenRows, Optional[TokenRows]]:
        """Return detached CPU probability rows in the input sequence order."""
        count = len(batch["input_ids"])
        # A DP peer can need more forward calls than this rank has real rows.
        batch = pad_actor_batch_rows(
            batch,
            1,
            torch.distributed.group.WORLD,
            self.torch_platform.current_device(),
        )
        size = self.cfg.algorithm.logprob_forward_micro_batch_size
        split_num = max(1, (len(batch["input_ids"]) + size - 1) // size)
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=split_num,
        )
        micro_batches = list(micro_batches_iter)

        def forward_rows() -> TokenRows:
            rows = []
            for micro in micro_batches:
                rows.extend(row.detach().cpu() for row in self.forward_batch(micro))
            if dbs_indices is not None:
                rows = [rows[i] for i in get_reverse_idx(sum(dbs_indices, []))]
            return rows[:count]

        recomputed_logprobs = forward_rows()
        ref_logprobs = None
        if compute_ref_logprobs:
            with cpu_weight_swap(self, self.ref_policy_state_dict, is_fsdp=True):
                ref_logprobs = forward_rows()
        return recomputed_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """Compute real-length probabilities and retain the balanced group order."""
        assert not self.is_pipeline, "MAFSDPActor supports collocated inference."
        batch, results = self.get_batch(input_channel)
        self._load_weight_and_optimizer()
        self.model.eval()
        with self.worker_timer(), torch.no_grad():
            probabilities, reference = self.inference_step(batch, compute_ref_logprobs)
        offset = 0
        for result in results:
            end = offset + result.num_sequence
            result.recomputed_logprobs = probabilities[offset:end]
            if reference is not None:
                result.ref_logprobs = reference[offset:end]
            output_channel.put(result)
            offset = end

    def get_response_loss_mask(
        self, m_batch: DynamicActorBatch | dict[str, torch.Tensor]
    ) -> torch.Tensor | TokenRows:
        """Return the full response mask used by dynamic multi-agent rollouts."""
        return m_batch["response_mask"]

    def forward_backward_batch(
        self,
        idx: int,
        m_batch: DynamicActorBatch,
        micro_batch_cnt: int,
    ) -> dict[str, torch.Tensor]:
        """Run one dynamic-rollout micro-batch with trajectory loss scaling."""
        backward_ctx = self.before_micro_batch(
            self.model,
            is_last_micro_batch=(idx + 1) == micro_batch_cnt,
        )
        # batch for forward
        logprobs, entropy = self.forward_batch(m_batch, True)

        width = self.cfg.actor.model.encoder_seq_length
        outputs = DynamicRolloutResult.materialize_actor_batch(
            {"logprobs": logprobs, "entropy": entropy},
            width,
            0,
            self.torch_platform.current_device(),
        )
        logprobs, entropy = outputs["logprobs"], outputs["entropy"]
        m_batch = DynamicRolloutResult.materialize_actor_batch(
            {
                key: value
                for key, value in m_batch.items()
                if key
                in (
                    "recomputed_logprobs",
                    "rollout_logprobs",
                    "ref_logprobs",
                    "advantages",
                    "loss_scales",
                    "response_mask",
                )
            },
            width,
            0,
            self.torch_platform.current_device(),
        )
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
            fast_path_zero_loss_mask=True,
        )

        entropy_loss = torch.tensor(0.0, device=self.torch_platform.current_device())
        if self.calculate_entropy:
            entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
            if self.calculate_entropy_loss:
                loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

        kl_loss = torch.tensor(0.0, device=self.torch_platform.current_device())
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

    def _compute_rollout_metrics(self, batch: DynamicActorBatch):
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
            "MAFSDPActor currently only supports collocated training"
        )

        global_batch, _ = self.get_batch(input_channel)
        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        global_batch = self.compute_advantages_and_returns(global_batch)
        global_batch["loss_scales"] = [
            mask.float() for mask in global_batch["response_mask"]
        ]

        if self.cfg.algorithm.normalize_advantages:
            raise AssertionError(
                "normalize_advantages is not implemented in multi-agent"
            )

        rollout_metrics = self._compute_rollout_metrics(global_batch)

        scale_context = {
            "folding_scale": [],
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
        restore_source_dp_normalization(
            global_batch,
            scale_context,
            torch.distributed.group.WORLD,
            self.torch_platform.current_device(),
        )
        for key in list(global_batch.keys()):
            if key == "idx_to_traj" or key.startswith("extra:"):
                global_batch.pop(key, None)

        self._load_weight_and_optimizer()

        global_batch = pad_actor_batch_rows(
            global_batch,
            self.cfg.algorithm.n_minibatches * self.cfg.actor.micro_batch_size,
            torch.distributed.group.WORLD,
            self.torch_platform.current_device(),
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
                if not mini_batch["input_ids"]:
                    continue
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)

            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(
        self, batch: DynamicActorBatch
    ) -> DynamicActorBatch:
        """Compute the advantages and returns.

        Args:
            batch: Real-length token rows and per-sequence metadata.
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
                    rewards=batch["rewards"].to(self.torch_platform.current_device()),
                    loss_mask=[
                        row.to(self.torch_platform.current_device()) for row in mask
                    ],
                    num_sequence=len(batch["input_ids"]),
                    group_size=self.cfg.algorithm.group_size,
                    idx_to_traj=batch["idx_to_traj"],
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=[
                        row.to(self.torch_platform.current_device()) for row in logprob
                    ],
                    ref_logprob=[
                        row.to(self.torch_platform.current_device())
                        for row in batch["ref_logprobs"]
                    ]
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = [row.cpu() for row in advantages]

        return batch
