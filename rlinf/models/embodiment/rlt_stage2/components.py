"""Core Stage 2 components for RLT.

This module keeps the original Stage 2 structure lightweight:
- a frozen VLA backbone provides reference actions and embeddings
- a frozen RL token encoder compresses embeddings into z_rl
- a direct Gaussian actor predicts action chunks conditioned on VLA references
- a twin-Q critic scores chunk actions for TD3-style updates
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transition import TransitionSource


class MLP(nn.Module):
    """Simple MLP used by the actor and Q networks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DirectGaussianActor(nn.Module):
    """Direct Gaussian actor over VLA-conditioned action chunks.

    The actor conditions on:
    - RL state x = [z_rl, s^p]
    - VLA reference action chunk a_tilde

    It directly predicts the action chunk mean. During sampling, a small fixed
    standard deviation is used, matching the RLT Gaussian actor formulation.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        sigma: float = 0.1,
        ref_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_chunk_dim = action_chunk_dim
        self.sigma = sigma
        self.ref_dropout = ref_dropout

        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def _apply_ref_dropout(self, a_tilde: Tensor) -> Tensor:
        if self.ref_dropout <= 0.0:
            return a_tilde

        keep_mask = (
            torch.rand(a_tilde.shape[0], 1, device=a_tilde.device) >= self.ref_dropout
        )
        return a_tilde * keep_mask

    def forward(
        self,
        x: Tensor,
        a_tilde: Tensor,
        deterministic: bool = False,
        *,
        apply_ref_dropout: bool | None = None,
        apply_action_noise: bool | None = None,
    ) -> Tensor:
        if apply_ref_dropout is None:
            apply_ref_dropout = False
        if apply_action_noise is None:
            apply_action_noise = not deterministic

        a_tilde_input = self._apply_ref_dropout(a_tilde) if apply_ref_dropout else a_tilde
        action = self.mlp(torch.cat([x, a_tilde_input], dim=-1))

        if apply_action_noise and self.sigma > 0.0:
            action = action + torch.randn_like(action) * self.sigma
        return action.clamp(-1.0, 1.0)


class QNetwork(nn.Module):
    """Single Q network: (x, a) -> scalar."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        return self.mlp(torch.cat([x, a], dim=-1))


class TwinQCritic(nn.Module):
    """Twin Q critic with target copies."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)
        self.q2 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for param in self.q1_target.parameters():
            param.requires_grad_(False)
        for param in self.q2_target.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(x, a), self.q2(x, a)

    def q_min(self, x: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(x, a)
        return torch.min(q1, q2)

    @torch.no_grad()
    def target_q_min(self, x: Tensor, a: Tensor) -> Tensor:
        q1 = self.q1_target(x, a)
        q2 = self.q2_target(x, a)
        return torch.min(q1, q2)

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for online, target in (
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ):
            for src_param, dst_param in zip(
                online.parameters(),
                target.parameters(),
                strict=True,
            ):
                dst_param.data.lerp_(src_param.data, tau)


@torch.no_grad()
def compute_td_target(
    *,
    rewards: Tensor,
    dones: Tensor,
    next_x: Tensor,
    next_a_tilde: Tensor,
    target_actor: DirectGaussianActor,
    critic: TwinQCritic,
    gamma: float,
    chunk_length: int,
) -> Tensor:
    """Compute chunk TD target using the target actor and target critic."""
    discount_powers = gamma ** torch.arange(
        chunk_length, device=rewards.device, dtype=rewards.dtype
    )
    chunk_return = (rewards * discount_powers).sum(dim=-1, keepdim=True)

    next_a = target_actor(
        next_x,
        next_a_tilde,
        deterministic=False,
        apply_ref_dropout=False,
        apply_action_noise=True,
    )

    next_q = critic.target_q_min(next_x, next_a)
    bootstrap = (gamma**chunk_length) * (1.0 - dones) * next_q
    return chunk_return + bootstrap


def critic_loss(q1: Tensor, q2: Tensor, q_target: Tensor) -> Tensor:
    return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)


def chunk_delta_loss(a: Tensor, a_tilde: Tensor) -> Tensor:
    if a.ndim != 3 or a_tilde.ndim != 3:
        raise ValueError(
            "chunk_delta_loss expects action tensors with shape [B, T, A], got "
            f"{tuple(a.shape)} and {tuple(a_tilde.shape)}."
        )

    if a.shape[1] <= 1:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    pred_delta = a[:, 1:, :] - a[:, :-1, :]
    ref_delta = a_tilde[:, 1:, :] - a_tilde[:, :-1, :]
    return F.mse_loss(pred_delta, ref_delta)


def actor_loss(
    q_value: Tensor,
    a: Tensor,
    a_tilde: Tensor,
    *,
    action_chunk: Tensor | None = None,
    source_chunk: Tensor | None = None,
    bc_weight: float,
    q_weight: float,
    delta_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if action_chunk is None:
        action_chunk = a_tilde
    if source_chunk is None:
        bc_target = a_tilde
        human_mask = torch.zeros(a.shape[:2], device=a.device, dtype=torch.bool)
    else:
        source_chunk = source_chunk.to(device=a.device)
        human_mask = torch.logical_or(
            source_chunk == int(TransitionSource.HUMAN),
            source_chunk == int(TransitionSource.MIXED),
        )
        bc_target = torch.where(human_mask[..., None], action_chunk, a_tilde)

    bc_error = (a - bc_target).square().mean(dim=-1)
    bc_loss = bc_error.mean()
    ref_error = (a - a_tilde).square().mean(dim=-1)
    human_error = (a - action_chunk).square().mean(dim=-1)
    policy_mask = ~human_mask
    bc_ref_loss = (
        ref_error[policy_mask].mean()
        if policy_mask.any()
        else torch.zeros((), device=a.device, dtype=a.dtype)
    )
    bc_human_loss = (
        human_error[human_mask].mean()
        if human_mask.any()
        else torch.zeros((), device=a.device, dtype=a.dtype)
    )
    delta_loss = chunk_delta_loss(a, bc_target)
    q_mean = q_value.mean()
    total_loss = (
        bc_weight * bc_loss
        - q_weight * q_mean
        + delta_weight * delta_loss
    )
    metrics = {
        "bc_loss": bc_loss.detach(),
        "bc_ref_loss": bc_ref_loss.detach(),
        "bc_human_loss": bc_human_loss.detach(),
        "delta_loss": delta_loss.detach(),
        "human_mask_ratio": human_mask.to(torch.float32).mean().detach(),
        "q_mean": q_mean.detach(),
    }
    return total_loss, metrics
