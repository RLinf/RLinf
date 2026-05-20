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

import torch
import torch.nn.functional as F


def prepare_edac_actions(
    actions: torch.Tensor,
    edac_eta: float,
    use_crossq: bool,
) -> torch.Tensor:
    """Prepare critic actions for optional EDAC regularization."""
    if edac_eta > 0.0 and use_crossq:
        raise NotImplementedError("EDAC regularizer is not supported with CrossQ")
    if edac_eta <= 0.0:
        return actions
    return actions.detach().requires_grad_(True)


def compute_edac_critic_diversity_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute EDAC-style action-gradient diversity loss for Q ensembles.

    Args:
        q_values: Critic outputs whose last dimension indexes Q heads.
        actions: Actions used to produce ``q_values``. Must require gradients.
        eps: Numerical stability epsilon for gradient normalization.

    Returns:
        Scalar loss that penalizes aligned action gradients across Q heads.
    """
    num_q_heads = q_values.shape[-1]
    if num_q_heads < 2:
        return q_values.sum() * 0.0

    if not actions.requires_grad:
        raise ValueError("actions must require gradients to compute EDAC loss")
    if not q_values.requires_grad:
        raise ValueError("q_values must require gradients to compute EDAC loss")

    action_grads = []
    for q_id in range(num_q_heads):
        grad = torch.autograd.grad(
            q_values[..., q_id].sum(),
            actions,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            raise ValueError(
                "q_values must be connected to actions to compute EDAC loss"
            )
        action_grads.append(grad.reshape(grad.shape[0], -1))

    grads = torch.stack(action_grads, dim=1).float()
    normalized_grads = F.normalize(grads, p=2, dim=-1, eps=eps)
    similarity = torch.matmul(normalized_grads, normalized_grads.transpose(-1, -2))

    mask = ~torch.eye(num_q_heads, dtype=torch.bool, device=q_values.device)
    off_diagonal_similarity = similarity[:, mask].reshape(
        similarity.shape[0], num_q_heads, num_q_heads - 1
    )
    per_sample_loss = off_diagonal_similarity.sum(dim=(-1, -2)) / (
        num_q_heads * (num_q_heads - 1)
    )
    return per_sample_loss.mean()


def compute_sac_critic_loss(
    q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    actions: torch.Tensor | None,
    edac_eta: float = 0.0,
    edac_grad_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SAC/RLPD critic loss with optional EDAC regularization."""
    target_q_values = target_q_values.detach().to(
        device=q_values.device,
        dtype=q_values.dtype,
    )
    critic_mse_loss = F.mse_loss(q_values, target_q_values.expand_as(q_values))
    critic_loss = critic_mse_loss
    metrics = {
        "mse_loss": critic_mse_loss.detach().item(),
        "q_data": q_values.mean().item(),
    }

    if edac_eta > 0.0:
        if actions is None:
            raise ValueError("actions must be provided when EDAC is enabled")
        edac_loss = compute_edac_critic_diversity_loss(
            q_values,
            actions,
            eps=edac_grad_eps,
        )
        edac_loss = edac_loss.to(dtype=critic_mse_loss.dtype)
        critic_loss = critic_loss + edac_eta * edac_loss
        metrics["edac_diversity_loss"] = edac_loss.detach().item()
        metrics["edac_eta"] = edac_eta

    return critic_loss, metrics
