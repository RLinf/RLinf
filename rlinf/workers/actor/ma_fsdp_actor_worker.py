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
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

import rlinf.algorithms  # noqa: F401
import rlinf.algorithms.pear_objective  # noqa: F401
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
    GroupDelivery,
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
from rlinf.utils.metric_utils import (
    VersionedTrajectoryRecord,
    compute_staleness_metrics,
)
from rlinf.utils.placement import (
    ModelParallelComponentPlacement,
    PlacementMode,
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
from rlinf.workers.rollout.session_server.assembly import (
    assemble_group_payload,
    verify_group_assembly,
)
from rlinf.workers.rollout.session_server.payload import payload_channel_for


class MAFSDPActor(FSDPActor):
    """FSDP actor for multi-agent and multimodal VLM dynamic rollouts.

    Supports both collocated synchronous training and disaggregated async
    training (including SessionServer ``GroupDelivery`` payloads, chunked FP32
    LM head forward, ``pear_reinforce`` loss, and top-K score centering).
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
        assert self.placement.placement_mode in (
            PlacementMode.COLLOCATED,
            PlacementMode.DISAGGREGATED,
        ), (
            "Multi-agent FSDP actor supports only collocated or disaggregated "
            f"placement, got {self.placement.placement_mode}"
        )
        if self.cfg.algorithm.get("importance_sampling_fix", False):
            raise ValueError(
                "importance_sampling_fix is not supported for dynamic rollout batch"
            )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)
        # Set by run_training for the duration of a training step and read by
        # _assemble_delivery when SessionServer is enabled.
        self._payload_channels: Optional[list[Channel]] = None
        self._processor: Any = None

        pear_cfg = self.cfg.algorithm.get("pear", None) or {}
        is_pear = self.cfg.algorithm.loss_type == "pear_reinforce"
        self._pear_score_centering = bool(pear_cfg.get("score_centering", False))
        self._use_fp32_head = bool(self.cfg.actor.get("use_fp32_head", is_pear))
        self._global_token_mean = bool(pear_cfg.get("global_token_mean", is_pear))
        self._current_global_valid_tokens: Optional[float] = None

    def _processor_or_load(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.cfg.actor.model.model_path,
                trust_remote_code=self.cfg.actor.model.get("trust_remote_code", True),
            )
        return self._processor

    def _receive_rollout_result(self, channel: Channel) -> DynamicRolloutResult:
        """Fetch one rollout result, assembling delivered SessionServer groups."""
        result = channel.get()
        if isinstance(result, GroupDelivery):
            result = self._assemble_delivery(result)
        return result

    def _assemble_delivery(self, delivery: GroupDelivery) -> DynamicRolloutResult:
        """Fetch and assemble one delivered group's payload from its channel."""
        assert self._payload_channels is not None, (
            "a group arrived as a GroupDelivery but this actor has no "
            "payload channels: run_training must receive them when the "
            "session server is enabled."
        )
        channel = payload_channel_for(
            delivery.payload_ref.locator, self._payload_channels
        )
        payload = channel.get(key=delivery.payload_ref.key)
        channel.remove_key(key=delivery.payload_ref.key, async_op=True)
        session_cfg = self.cfg.get("session_server", None)
        merge_turns = bool(
            session_cfg.get("merge_continuous_turns", False)
            if session_cfg is not None
            else False
        )
        has_multimodal = any(
            part.delta_text is not None for part in payload.parts if not part.last
        )
        processor = self._processor_or_load() if has_multimodal else None
        response_budget = OmegaConf.select(
            self.cfg, "rollout.sglang.response_budget", default=None
        )
        result = assemble_group_payload(
            payload,
            not self.cfg.algorithm.recompute_logprobs,
            merge_continuous_turns=merge_turns,
            processor=processor,
            response_budget=response_budget,
        )
        result.group_id = delivery.group_id
        result.task_name = delivery.task_name
        verify = next(part.verify for part in payload.parts if part.last)
        if verify is not None:
            verify_group_assembly(result, verify, self.cfg)
        return result

    def _get_num_rollout_results_per_dp(self) -> int:
        """Count question groups; each group can contain a different turn count."""
        return self.cfg.data.rollout_batch_size // torch.distributed.get_world_size()

    def get_rollout_metrics_group(
        self, batch: DynamicActorBatch
    ) -> torch.distributed.ProcessGroup:
        """Return the process group used to aggregate rollout metrics."""
        return torch.distributed.group.WORLD

    def get_step_batch(
        self, channel: Channel
    ) -> tuple[DynamicActorBatch, list[DynamicRolloutResult]]:
        """Receive this data-parallel rank's groups for one step, balance them
        once, and keep real-length rows.

        Unlike get_batch, which receives one rollout result, this receives
        all of the step's groups and returns them as a list.
        """
        results = [
            self._receive_rollout_result(channel)
            for _ in range(self._get_num_rollout_results_per_dp())
        ]
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
        batch, results = self.get_step_batch(input_channel)
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

    def _forward_chunked_fp32_head(
        self,
        m_batch: DynamicActorBatch,
    ) -> tuple[TokenRows, TokenRows, Optional[list[torch.Tensor]]]:
        """Run unpadded backbone forward + chunked FP32 LM head on assistant tokens.

        Avoids materializing full ``[seq_len, vocab_size]`` logits on long
        multimodal episodes (160k-256k tokens).
        """
        from rlinf.models.embodiment.qwen_vlm.fp32_head import (
            response_logprobs_and_heads_from_hidden,
            response_logprobs_from_hidden,
        )

        device = self.torch_platform.current_device()
        backbone = self.model.model
        lm_head = self.model.get_output_embeddings()
        temperature = float(self.cfg.algorithm.sampling_params.temperature)

        mm_list = m_batch.get("multi_modal_inputs")
        topk_ids_list = (
            m_batch.get("rollout_topk_ids") if self._pear_score_centering else None
        )

        logprobs_rows: TokenRows = []
        entropy_rows: TokenRows = []
        train_head_rows: Optional[list[torch.Tensor]] = (
            [] if topk_ids_list is not None else None
        )

        for idx, (row, mask_row) in enumerate(
            zip(m_batch["input_ids"], m_batch["response_mask"])
        ):
            sample_ids = row.to(device).unsqueeze(0)
            sample_mask = mask_row.to(device=device, dtype=torch.bool)
            sample_attn = torch.ones_like(sample_ids, dtype=torch.long)
            resp_pos = sample_mask.nonzero(as_tuple=False).flatten()
            model_inputs: dict[str, Any] = {
                "input_ids": sample_ids,
                "attention_mask": sample_attn,
                "use_cache": False,
                "return_dict": False,
                "logits_to_keep": slice(0, 0),
            }
            if mm_list is not None and mm_list[idx] is not None:
                for key, val in mm_list[idx].items():
                    tensor = val.to(device)
                    if key == "mm_token_type_ids" and tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    model_inputs[key] = tensor

            captured_hidden: list[torch.Tensor] = []

            def _capture_backbone(_module, _args, output):
                captured_hidden.append(output[0])

            def _chunked_head_hook(_module, _args, _output):
                hidden = captured_hidden.pop()
                lm_weight = lm_head.weight
                if resp_pos.numel() == 0:
                    sel_lp = hidden.new_zeros((1, 0), dtype=torch.float32)
                    sel_ent = hidden.new_zeros((1, 0), dtype=torch.float32)
                    if train_head_rows is not None:
                        k = int(topk_ids_list[idx].shape[-1])
                        return (
                            sel_lp,
                            sel_ent,
                            hidden.new_zeros((1, 0, k), dtype=torch.float32),
                        )
                    return (sel_lp, sel_ent)
                if train_head_rows is not None:
                    head_ids = topk_ids_list[idx].to(device=device, dtype=torch.long)
                    return response_logprobs_and_heads_from_hidden(
                        hidden,
                        lm_weight,
                        sample_ids,
                        resp_pos.unsqueeze(0),
                        head_ids.unsqueeze(0),
                        temperature=temperature,
                    )
                return response_logprobs_from_hidden(
                    hidden,
                    lm_weight,
                    sample_ids,
                    resp_pos.unsqueeze(0),
                    temperature=temperature,
                )

            # Call the root module so FSDP2 unshards root-level parameters
            # (final norm and lm_head) and attaches pre-/post-backward hooks
            # to the returned chunked logprob tensors, while logits_to_keep=
            # slice(0, 0) keeps the full [seq_len, vocab] logits unallocated.
            backbone_hook = backbone.register_forward_hook(_capture_backbone)
            head_hook = lm_head.register_forward_hook(_chunked_head_hook)
            try:
                with self.amp_context:
                    head_out = self.model(**model_inputs)[0]
            finally:
                backbone_hook.remove()
                head_hook.remove()

            if train_head_rows is not None:
                sel_lp, sel_ent, sel_head = head_out
            else:
                sel_lp, sel_ent = head_out
                sel_head = None

            seq_lp = sel_lp.new_zeros(row.shape[0], dtype=torch.float32)
            seq_ent = sel_ent.new_zeros(row.shape[0], dtype=torch.float32)
            if resp_pos.numel() > 0:
                seq_lp[resp_pos] = sel_lp[0]
                seq_ent[resp_pos] = sel_ent[0]
            if train_head_rows is not None and sel_head is not None:
                train_head_rows.append(sel_head[0])

            logprobs_rows.append(seq_lp)
            entropy_rows.append(seq_ent)

        return logprobs_rows, entropy_rows, train_head_rows

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
        device = self.torch_platform.current_device()
        can_use_fp32_head = (
            self._use_fp32_head
            and hasattr(self.model, "model")
            and hasattr(self.model, "get_output_embeddings")
            and self.model.get_output_embeddings() is not None
        )
        train_head_rows: Optional[list[torch.Tensor]] = None
        if can_use_fp32_head:
            logprobs, entropy, train_head_rows = self._forward_chunked_fp32_head(
                m_batch
            )
            width = max(len(row) for row in logprobs)
        else:
            logprobs, entropy = self.forward_batch(m_batch, True)
            width = self.cfg.actor.model.encoder_seq_length

        outputs = DynamicRolloutResult.materialize_actor_batch(
            {"logprobs": logprobs, "entropy": entropy},
            width,
            0,
            device,
        )
        logprobs, entropy = outputs["logprobs"], outputs["entropy"]
        raw_topk_logprobs = m_batch.get("rollout_topk_logprobs")
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
            device,
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

        extra_loss_kwargs: dict[str, Any] = {}
        if (
            self._pear_score_centering
            and train_head_rows is not None
            and raw_topk_logprobs is not None
        ):
            from rlinf.algorithms.pear_score_centering import (
                score_centering_correction,
            )

            sc_corr = logprobs.new_zeros(logprobs.shape, dtype=torch.float32)
            sc_sampler_mass = logprobs.new_zeros(logprobs.shape, dtype=torch.float32)
            sc_train_mass = logprobs.new_zeros(logprobs.shape, dtype=torch.float32)
            for row_idx, (train_head, sampler_head) in enumerate(
                zip(train_head_rows, raw_topk_logprobs)
            ):
                resp_pos = loss_mask[row_idx].nonzero(as_tuple=False).flatten()
                if resp_pos.numel() == 0 or train_head.shape[0] != resp_pos.numel():
                    continue
                corr, s_mass, t_mass = score_centering_correction(
                    train_head,
                    sampler_head.to(device=device, dtype=torch.float32),
                )
                sc_corr[row_idx, resp_pos] = corr
                sc_sampler_mass[row_idx, resp_pos] = s_mass
                sc_train_mass[row_idx, resp_pos] = t_mass
            extra_loss_kwargs["sc_correction"] = sc_corr
            extra_loss_kwargs["sc_sampler_head_mass"] = sc_sampler_mass
            extra_loss_kwargs["sc_train_head_mass"] = sc_train_mass
        if self._current_global_valid_tokens is not None:
            extra_loss_kwargs["global_valid_tokens"] = self._current_global_valid_tokens

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
            **extra_loss_kwargs,
        )

        entropy_loss = torch.tensor(0.0, device=device)
        if self.calculate_entropy:
            entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
            if self.calculate_entropy_loss:
                loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

        kl_loss = torch.tensor(0.0, device=device)
        if self.kl_beta > 0 and ref_logprobs is not None:
            kld = kl_penalty(logprobs, ref_logprobs, self.kl_penalty_type)
            kl_loss = self.loss_agg_func(kld * m_batch["loss_scales"], loss_mask)
            loss = loss + kl_loss * self.kl_beta

        # add to log
        # scale loss for gradient accumulation and backprop
        final_loss_metric = loss.detach()
        if self._current_global_valid_tokens is not None:
            # Loss is already normalized by global valid tokens across all
            # micro-batches and DP ranks; multiply by DP world size to offset
            # FSDP's mean gradient all-reduce.
            dp_world = (
                torch.distributed.get_world_size(group=getattr(self, "_dp_group", None))
                if torch.distributed.is_initialized()
                else 1
            )
            loss = loss * float(dp_world)
        else:
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

    def _extract_versioned_trajectory_records(
        self, rollout_results: list[DynamicRolloutResult]
    ) -> Optional[list[VersionedTrajectoryRecord]]:
        """Extract per-trajectory weight-version records for staleness metrics."""
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
            earliest_versions: list[Optional[int]] = [None] * num_trajectories
            completion_versions: list[Optional[int]] = [None] * num_trajectories
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
        """Compute staleness metrics across all DP ranks, or None to skip."""
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
        all_records = []
        group_index_offset = 0
        for rank_records in records_by_rank:
            for record in rank_records:
                all_records.append(
                    replace(record, group_index=group_index_offset + record.group_index)
                )
            group_index_offset += len(rank_records)
        return compute_staleness_metrics(all_records, version)

    def _compute_global_valid_tokens(self, mini_batch: DynamicActorBatch) -> float:
        """Sum valid response tokens across this mini-batch and all DP ranks."""
        local_tokens = float(
            sum(
                int(mask.to(dtype=torch.bool).sum().item())
                for mask in mini_batch["response_mask"]
            )
        )
        token_tensor = torch.tensor(
            local_tokens,
            dtype=torch.float64,
            device=self.torch_platform.current_device(),
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                token_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=getattr(self, "_dp_group", None),
            )
        return float(token_tensor.item())

    def run_training(
        self,
        input_channel: Channel,
        do_offload: bool = False,
        version: Optional[int] = None,
        payload_channels: Optional[list[Channel]] = None,
    ) -> tuple[dict, list]:
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        self._payload_channels = payload_channels
        try:
            global_batch, results = self.get_step_batch(input_channel)
        finally:
            self._payload_channels = None

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
        staleness_metrics = self._compute_staleness_metrics(
            results, version, torch.distributed.group.WORLD
        )
        if staleness_metrics:
            if rollout_metrics is None:
                rollout_metrics = {}
            rollout_metrics.update(staleness_metrics)

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
                if self._global_token_mean:
                    self._current_global_valid_tokens = (
                        self._compute_global_valid_tokens(mini_batch)
                    )
                try:
                    mean_metric_dict = self.training_step(batch=mini_batch)
                finally:
                    self._current_global_valid_tokens = None
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
