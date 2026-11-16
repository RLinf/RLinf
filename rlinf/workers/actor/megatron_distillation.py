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

"""Alignment and reduction shared by ordinary and multi-agent Megatron actors."""

import torch
from megatron.core import parallel_state

from rlinf.algorithms.distillation.losses import (
    DistillationForwardOutput,
    DistillationReductionContext,
    combine_distillation_and_task_loss,
    compute_distillation_loss,
)
from rlinf.algorithms.distillation.megatron import (
    compute_vocab_parallel_distribution_loss,
)


def teacher_processor_args(actor, batch, *, dynamic: bool):
    """Shift target-position tensors once before the shared packing transform."""
    if not actor.distillation_plan.enabled or not actor.return_loss:
        return {}
    result = {}
    length = batch["input_ids"].shape[1]
    for key in ("teacher_logprobs", "teacher_token_ids", "teacher_valid_mask"):
        if key not in batch:
            continue
        value = batch[key]
        if (
            key == "teacher_logprobs"
            and actor.distillation_plan.target_format == "full"
        ):
            tp = parallel_state.get_tensor_model_parallel_world_size()
            rank = parallel_state.get_tensor_model_parallel_rank()
            padded = actor.cfg.actor.model.padded_vocab_size
            shard = padded // tp
            start = rank * shard
            local = torch.zeros(
                (*value.shape[:-1], shard), dtype=value.dtype, device=value.device
            )
            count = max(0, min(shard, value.shape[-1] - start))
            local[..., :count] = value[..., start : start + count]
            value = local
        aligned = torch.zeros(
            (value.shape[0], length, *value.shape[2:]),
            dtype=value.dtype,
            device=batch["input_ids"].device,
        )
        value = value.to(aligned.device)
        if dynamic:
            aligned[:, :-1] = value[:, 1:]
        else:
            aligned[:, -actor.response_len - 1 : -1] = value
        result[key] = aligned
    return result


def distribution_processor(
    actor, logits, teacher_logprobs, teacher_valid_mask, teacher_token_ids=None
):
    """Evaluate the vocabulary-parallel objective while logits remain available."""
    rank = parallel_state.get_tensor_model_parallel_rank()
    width = logits.shape[-1]
    output = compute_vocab_parallel_distribution_loss(
        logits,
        teacher_token_ids,
        teacher_logprobs,
        vocab_start=rank * width,
        vocab_end=(rank + 1) * width,
        vocab_size=actor.cfg.actor.model.padded_vocab_size,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        config=actor.distillation_loss_config,
        valid_mask=teacher_valid_mask,
    )
    return {
        name: getattr(output, name)
        for name in ("distribution_losses", "student_mass", "teacher_mass", "tail_loss")
    }


def combine_actor_distillation(
    actor,
    output,
    batch,
    task_loss,
    *,
    dynamic: bool,
    regularization_loss: torch.Tensor | None = None,
):
    """Normalize task loss and regularizers together, then add distillation."""
    fields = {}
    for source, dest in (
        ("log_probs", "logprobs"),
        ("entropy", "entropy"),
        ("distribution_losses", "distribution_losses"),
        ("student_mass", "student_mass"),
        ("teacher_mass", "teacher_mass"),
        ("tail_loss", "tail_loss"),
    ):
        value = output.get(source)
        if value is None:
            continue
        if dynamic:
            aligned = torch.zeros_like(value)
            aligned[:, 1:] = value[:, :-1]
        else:
            aligned = value[:, -actor.response_len - 1 : -1]
        fields[dest] = aligned
    aligned_batch = dict(batch)
    if actor.distillation_plan.use_policy_gradient and actor.cfg.algorithm.get(
        "importance_sampling_fix", False
    ):
        from rlinf.algorithms.utils import apply_importance_sampling_correction

        old = batch["recomputed_logprobs"]
        weights, _, _ = apply_importance_sampling_correction(
            torch.ones_like(old),
            old,
            batch["rollout_logprobs"],
            actor.cfg.algorithm.importance_sampling_clip,
        )
        aligned_batch["rollout_is_weights"] = weights

    mask = (
        batch["response_mask"]
        if dynamic
        else batch["response_mask"][:, -actor.response_len :]
    )
    aligned_batch["response_mask"] = mask
    aligned_batch["teacher_valid_mask"] = batch["teacher_valid_mask"].to(mask.device)
    if actor.distillation_plan.target_format == "sampled":
        aligned_batch["teacher_logprobs"] = batch["teacher_logprobs"].to(mask.device)
    loss, metrics = compute_distillation_loss(
        DistillationForwardOutput(**fields),
        aligned_batch,
        plan=actor.distillation_plan,
        config=actor.distillation_loss_config,
        reduction=DistillationReductionContext(
            actor._distillation_denominator,
            mode=actor.cfg.algorithm.loss_agg_func,
            dp_size=parallel_state.get_data_parallel_world_size(),
            gradient_divisor=actor.num_microbatches,
        ),
    )
    if actor.distillation_plan.need_task_rewards and regularization_loss is not None:
        task_loss = task_loss + regularization_loss
    if (
        not dynamic
        and actor.distillation_plan.need_task_rewards
        and actor.cfg.algorithm.use_valid_token_scale
    ):
        task_scale = (
            mask.sum()
            / actor.global_valid_token
            * parallel_state.get_data_parallel_world_size()
            * actor.num_microbatches
        )
        task_loss = task_loss * task_scale.item()
    combined = combine_distillation_and_task_loss(
        task_loss,
        loss,
        use_task_rewards=actor.distillation_plan.need_task_rewards,
        coefficient=actor.distillation_loss_config.distillation_loss_coef,
    )
    if (
        not actor.distillation_plan.need_task_rewards
        and regularization_loss is not None
    ):
        combined = combined + regularization_loss
    return combined, metrics
