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
    logprob: torch.Tensor, ref_logprob: torch.Tensor, kl_penalty: str
) -> torch.Tensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob: Tensor of log probabilities.
        ref_logprob: Tensor of reference log probabilities.
        kl_penalty: Type of KL penalty to compute.

    Returns:
        Tensor representing the KL penalty.

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
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        assert values is not None, "Values must be provided for GAE computation"
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


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    """
    Calculate scores for batched groups.

    Args:
        rewards: [n_steps, bsz]: reward values
        dones: [n_steps+1, bsz]: done flags

    Returns:
        dict with updated rewards (scores) and dones
    """
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
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
    It will reshape advantages and returns (if not None) to [batch_size, num_chunk, chunk_size]

    Args:
        advantages: [n_steps, bsz]: calculated advantages
        num_chunk: number of chunks
        chunk_size: size of each chunk
        returns: [n_steps, bsz] or None

    Returns:
        dict with keys "advantages" and optionally "returns", both reshaped to
        [batch_size, num_chunk, chunk_size]
    """
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Preprocess inputs before computing advantages & returns for reasoning tasks
    according to adv_type("gae", "grpo", "reinpp").

    If adv_type is "gae", it will set rewards to be of shape [seq_len, bsz],
    in which only the last token has the reward value.
    If adv_type is "grpo", it will group rewards according to group_size.
    If adv_type is "reinpp", it will unsqueeze rewards to have shape [1, bsz].
    If `values` for A-C algorithms is provided, it will be transposed to [seq_len, bsz].
    If `logprob` and `ref_logprob` are provided, they will be transposed to [seq_len, bsz].



    Args:
        rewards: [bsz]: reward for each sequence
        loss_mask: [bsz, seq_len]: mask for valid tokens
        values: [bsz, seq_len]: value estimates
        logprob: [bsz, seq_len]: log probabilities generated by current policy
        ref_logprob: [bsz, seq_len]: reference log probabilities
    """
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Post-process results for Reasoning tasks; transpose tensors back
    (by call `transpose(0, 1)`).

    Args:
        advantages: [seq_len, bsz]: calculated advantages
        returns: [seq_len, bsz] or None
    """

    advantages = advantages.transpose(0, 1)  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1)  # [bsz, seq_len]

    return advantages, returns


def preprocess_embodiment_loss_inputs(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    single_action_dim: int,
    logprob_type: Optional[str] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Preprocess inputs before computing policy loss for embodiment tasks.

    Args:
        logprobs: [bsz, num_action_chunks, action_dim]: log probabilities generated by current policy
        proximal_logprobs: [bsz, num_action_chunks, action_dim]: proximal log probabilities
        old_logprobs: [bsz, num_action_chunks, action_dim]: log probabilities generated by old policy
        advantages: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: calculated advantages
        logprob_type: str, type of logprob ("token_level", "action_level", "chunk_level")
        single_action_dim: int, action dimension for a single action chunk
        loss_mask: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: loss mask for valid
        loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: loss mask for summation
        values: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: value estimates
        prev_values: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: previous value estimates
        returns: [n_chunk_steps, bsz, num_action_chunks] or [n_chunk_steps, bsz]: calculated returns
        reward_type: str, type of reward ("chunk_level", "action_level", "token_level")

    Returns:
        dict of processed inputs, including logprobs, old_logprobs, advantages, loss_mask,
        loss_mask_sum, values, prev_values, returns.
    """
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
        proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim).sum(
            dim=-1
        )
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
    elif logprob_type == "chunk_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim).sum(
            dim=[1, 2]
        )
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])

    target_shape = logprobs.shape
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "proximal_logprobs": proximal_logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    """
    Iterate throught passed param metrics_data and convert all tensor values to float numbers.
    The tensor value must be a scalar tensor. If the value is already a float or int, it will be kept unchanged.

    Args:
        metrics_data: dict with metric names as keys and tensor/float/int as values

    Returns:
        dict with metric names as keys and float numbers as values
    """
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(
    tensor: Optional[torch.Tensor], target_shape: tuple
) -> Optional[torch.Tensor]:
    """
    Expand tensor to target shape by unsqueezing dimensions.

    Args:
        tensor: input tensor
        target_shape: target shape

    Returns:
        expanded tensor or None if input tensor is None
    """
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Get valid elements according to loss_mask and normalize the valid array.
    (by subtracting mean and dividing std)

    Args:
        array: input tensor
        loss_mask: boolean mask for valid elements

    Returns:
        normalized tensor
    """
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
