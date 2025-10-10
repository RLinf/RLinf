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


def preprocess_loss_inputs(**kwargs) -> dict:
    logprob_type = kwargs.get("logprob_type", None)
    entropy_type = kwargs.get("entropy_type", None)
    single_action_dim = kwargs.get("single_action_dim", None)

    logprobs = kwargs["logprobs"]
    old_logprobs = kwargs["old_logprobs"]
    advantages = kwargs["advantages"]
    entropy = kwargs.get("entropy", None)
    loss_mask = kwargs.get("loss_mask", None)

    bsz = logprobs.shape[0]

    if logprob_type == "token_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)

    elif logprob_type == "action_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)

    elif logprob_type == "chunk_level":
        logprobs = logprobs.sum(dim=-1)
        old_logprobs = old_logprobs.sum(dim=-1)
        advantages = advantages.sum(dim=-1)

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
        }
    )

    return kwargs


def preprocess_advantages_inputs(**kwargs) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    reward_type = kwargs.get("reward_type", None)
    if reward_type == "chunk_level":
        rewards = kwargs["rewards"]
        dones = kwargs["dones"]
        kwargs["rewards"] = rewards.sum(dim=-1, keepdim=True)
        kwargs["dones"] = dones[..., -1:]

    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    values = kwargs.get("values", None)
    loss_mask = kwargs.get("loss_mask", None)
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
    kwargs["rewards"] = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        kwargs["loss_mask"] = (
            loss_mask.transpose(1, 2).reshape(n_steps, bsz).transpose(0, 1)
        )

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    kwargs["dones"] = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        kwargs["values"] = flattened_values_full[: n_steps + 1]

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def calculate_scores(**kwargs):
    rewards = kwargs["rewards"]
    bsz = kwargs["batch_size"]
    n_steps = kwargs["n_steps"]
    dones = kwargs["dones"]
    group_size = kwargs["group_size"]

    scores = torch.zeros(bsz)
    for step in reversed(range(n_steps)):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, group_size)

    kwargs.update({"reward_scores": scores})

    return kwargs


def postprocess_advantages_outputs(**kwargs):
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    advantages = kwargs["advantages"]
    returns = kwargs.get("returns", None)
    num_chunk = kwargs["num_chunk"]
    chunk_size = kwargs["chunk_size"]

    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res
