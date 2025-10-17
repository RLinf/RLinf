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

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    entropy_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    entropy: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    **kwargs,
) -> dict:
    if reward_type == "chunk_level":
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    if logprob_type == "token_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)

    elif logprob_type == "chunk_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])

    target_shape = logprobs.shape
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)

    if entropy is not None:
        if entropy_type == "action_level":
            entropy = entropy.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        elif entropy_type == "chunk_level":
            entropy = entropy.sum(dim=-1)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "entropy": entropy,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    if kwargs["reward_type"] == "chunk_level":
        # TODO: need check
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        if kwargs["adv_type"] == "gae":
            loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)
        else:
            # NOTE: to unify with math grpo, we transpose loss mask here.
            loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz).transpose(0, 1)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs

def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"
    reward_scores = torch.zeros((seq_len, bsz), dtype=rewards.dtype, device=rewards.device)
    reward_scores[-1] = rewards  # only last token has reward

    if values is not None: # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat([values, torch.zeros((1, values.shape[-1]), dtype=values.dtype, device=values.device)], dim=0) # [seq_len+1, bsz]

    # Create done flags (episode ends at the last token)
    dones = torch.zeros_like(values, dtype=torch.bool)
    dones[-1] = True

    print(reward_scores.shape, loss_mask.shape, dones.shape, values.shape if values is not None else None)

    kwargs.update(
        {
            "rewards": reward_scores,
            "loss_mask": loss_mask,
            "dones": dones,
            "values": values,
        }
    )

    return kwargs

def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    advantages = advantages.transpose(0, 1)  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1)  # [bsz, seq_len]

    return advantages, returns

def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    batch_size: int,
    n_steps: int,
    group_size: int,
    **kwargs,
) -> dict:
    scores = torch.zeros(batch_size)
    for step in reversed(range(n_steps)):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, group_size)

    kwargs.update(
        {
            "reward_scores": scores,
            "group_size": group_size,
            "rewards": rewards,
            "dones": dones,
            "n_steps": n_steps,
            "batch_size": batch_size,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    # NOTE: to unify with math grpo, we transposed loss mask. Therefore, for embodied grpo, we need to transpose back here.
    if kwargs["adv_type"] == "grpo":
        advantages = advantages.transpose(0, 1)

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def expand_to_target_dim(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(
    array: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if loss_mask is not None:
        mean = array[loss_mask].mean()
        std = array[loss_mask].std(correction=0)
    else:
        mean = array.mean()
        std = array.std(correction=0)
    if std < 1e-8 or torch.isinf(std).any() or torch.isnan(std).any():
        array = torch.zeros_like(array)
    else:
        array = (array - mean) / (std + 1e-8)

    return array

def safe_normalize_2(array, loss_mask):
    array = array * loss_mask

    # Only normalize over valid (masked) positions
    valid_array = array[loss_mask.bool()]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
