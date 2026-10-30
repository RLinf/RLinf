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

"""SiLU MLPs for SuperMimic. Not registered in EMBODIED_MODEL."""

from __future__ import annotations

import torch
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    OBS_DIM,
    RESIDUAL_DIM,
    STATE_DIM,
    XI_DIM,
    critic_input_dim,
)
from rlinf.envs.sim.g1_mimic.features import (
    error_floor,
    frame_local_error,
    heading_invariant_features,
    normalize_xi,
    residual_target,
)


def _mlp_forward(
    owner: nn.Module, net: nn.Module, features: torch.Tensor
) -> torch.Tensor:
    """Run one linear stack.

    ``owner.bf16_matmul`` autocasts the stack and casts the result back to
    ``features.dtype``. Heading features, integration and losses stay outside.
    """
    if not getattr(owner, "bf16_matmul", False):
        return net(features)
    with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16):
        out = net(features)
    return out.to(dtype=features.dtype)


class RunningNorm(nn.Module):
    """Welford mean and variance. Updates do not track gradients.

    zita-ch freezes the statistics after ``max_count`` samples. ``eps`` is
    1e-2 on inputs and 1e-4 on the world-model residual.
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = float(max_count)
        self.eps = float(eps)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Fold one batch into the running statistics."""
        flat = values.detach().reshape(-1, values.shape[-1]).float()
        if flat.numel() == 0 or float(self.count) >= self.max_count:
            return
        count = flat.shape[0]
        batch_mean = flat.mean(0)
        batch_var = flat.var(0, unbiased=False)
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean.add_(delta * count / total)
        self.var.copy_(
            (
                self.var * self.count
                + batch_var * count
                + delta.square() * self.count * count / total
            )
            / total
        )
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        """Standard deviation including ``eps``."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize ``values`` on the last dimension."""
        return (values - self.mean) / self.std()


def _mlp(sizes: list[int], last_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        if i < len(sizes) - 2:
            layers.extend((linear, nn.SiLU()))
        else:
            if last_gain is not None:
                nn.init.xavier_uniform_(linear.weight, gain=last_gain)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
    return nn.Sequential(*layers)


class DynamicsNet(nn.Module):
    """Paper Table A.4: MLP hidden widths 2048, 2048, 2048. Last layer is zero.

    ``input_norm`` applies zita-ch's fixed ξ scale and a running norm to the
    MLP input. ``output_norm`` turns the MLP output back into a physical
    residual with the running mean and standard deviation of one-step targets.
    """

    def __init__(self, input_norm: bool = False, output_norm: bool = False):
        super().__init__()
        self.net = _mlp(
            [
                STATE_DIM + ACTION_DIM + XI_DIM,
                2048,
                2048,
                2048,
                RESIDUAL_DIM,
            ]
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        width = STATE_DIM + ACTION_DIM + XI_DIM
        self.in_norm = RunningNorm(width) if input_norm else None
        self.out_norm = RunningNorm(RESIDUAL_DIM, eps=1e-4) if output_norm else None

    def features(
        self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Heading-free state, action, and ξ. Running mode scales ξ first."""
        if self.in_norm is not None:
            xi = normalize_xi(xi)
        return torch.cat((heading_invariant_features(state), action, xi), dim=-1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Residual ``[B, 70]`` of the next state from heading-free features."""
        feat = self.features(state, action, xi)
        if self.in_norm is not None:
            feat = self.in_norm(feat)
        out = _mlp_forward(self, self.net, feat)
        if self.out_norm is None:
            return out
        return out * self.out_norm.std() + self.out_norm.mean

    @torch.no_grad()
    def update_norm(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        xi: torch.Tensor,
        nxt: torch.Tensor,
        frame: str,
    ) -> None:
        """Fold one real transition into the input and output running norms."""
        if self.in_norm is not None:
            self.in_norm.update(self.features(state, action, xi))
        if self.out_norm is not None:
            self.out_norm.update(residual_target(state, nxt, frame))


def running_state_loss(pred: torch.Tensor, true: torch.Tensor, dynamics, frame: str):
    """zita-ch's state loss: frame-local error divided by the running std.

    Each component is at least :func:`error_floor`. The result is the mean
    over the 70 components and over the batch, matching their per-step term.
    """
    err = frame_local_error(pred, true, frame)
    scale = torch.maximum(dynamics.out_norm.std(), error_floor(err))
    return (err / scale).square().mean()


def _normalize_history(norm: RunningNorm | None, history: torch.Tensor, obs_dim: int):
    """Normalize each observation frame. A missing norm leaves history as stored."""
    if norm is None:
        return history
    flat = history.reshape(-1, HISTORY_LEN, obs_dim)
    return norm(flat).reshape(history.shape)


class MotionEncoder(nn.Module):
    """Reference-motion encoder: the 680/689/730-d goal to a 32-d tanh latent."""

    def __init__(self, goal_dim: int, input_norm: bool = False):
        super().__init__()
        self.net = _mlp([goal_dim, 2048, 1024, 512, 256, LATENT_DIM])
        self.goal_norm = RunningNorm(goal_dim) if input_norm else None

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        """Latent ``[B, 32]`` of a goal ``[B, goal_dim]``."""
        if self.goal_norm is not None:
            goal = self.goal_norm(goal)
        return torch.tanh(_mlp_forward(self, self.net, goal))


class MotionDecoder(nn.Module):
    """Shared decoder: a goal latent and the observation history to an action."""

    def __init__(self, obs_dim: int = OBS_DIM, input_norm: bool = False):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.net = _mlp(
            [LATENT_DIM + HISTORY_LEN * obs_dim, 2048, 1024, 512, 256, ACTION_DIM],
            last_gain=0.01,
        )
        self.obs_norm = RunningNorm(obs_dim) if input_norm else None

    def forward(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Action ``[B, 29]`` from a latent and a flat 10-frame history."""
        history = _normalize_history(self.obs_norm, history, self.obs_dim)
        return _mlp_forward(self, self.net, torch.cat((latent, history), dim=-1))


class CriticNet(nn.Module):
    """MLP(256, 128) on one prebuilt motion-critic feature vector."""

    def __init__(self, in_dim: int, input_norm: bool = False):
        super().__init__()
        self.net = _mlp([in_dim, 256, 128, 1])
        self.norm = RunningNorm(in_dim) if input_norm else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Value ``[B]`` from ``motion_critic_input`` features."""
        if self.norm is not None:
            features = self.norm(features)
        return _mlp_forward(self, self.net, features).squeeze(-1)


class TerminationHead(nn.Module):
    """Failure logit from height and tilt errors (Eq. 1 termination head).

    Args:
        bias_init: Output bias at construction; ``None`` keeps PyTorch's init.
    """

    def __init__(self, bias_init: float | None = None):
        super().__init__()
        self.net = _mlp([4, 64, 64, 1])
        if bias_init is not None:
            nn.init.constant_(self.net[-1].bias, float(bias_init))

    def forward(self, term_feat: torch.Tensor) -> torch.Tensor:
        """Failure logit ``[B]`` from ``termination_head_input`` features."""
        return _mlp_forward(self, self.net, term_feat).squeeze(-1)


class VelocityEncoder(nn.Module):
    """MLP(256, 256) on the normalized command and the 10-frame observation."""

    def __init__(self, obs_dim: int = OBS_DIM, obs_norm: RunningNorm | None = None):
        super().__init__()
        self.obs_dim = int(obs_dim)
        # Shared with the decoder. Registering it here would move the module
        # off the decoder and write the buffers twice.
        object.__setattr__(self, "obs_norm", obs_norm)
        self.net = _mlp([3 + HISTORY_LEN * obs_dim, 256, 256, LATENT_DIM])

    def forward(self, command: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Latent ``[B, 32]`` of a normalized command and the history."""
        history = _normalize_history(self.obs_norm, history, self.obs_dim)
        packed = torch.cat((command, history), dim=-1)
        return torch.tanh(_mlp_forward(self, self.net, packed))


class VelocityCritic(nn.Module):
    """One critic for the velocity interface: heading state, command, and ξ."""

    def __init__(self, input_norm: bool = False):
        super().__init__()
        width = STATE_DIM + 3 + XI_DIM
        self.net = _mlp([width, 256, 128, 1])
        self.norm = RunningNorm(width) if input_norm else None

    def features(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Heading state, command, and ξ. Running mode scales ξ first."""
        if self.norm is not None:
            xi = normalize_xi(xi)
        return torch.cat((heading_invariant_features(state), command, xi), dim=-1)

    @torch.no_grad()
    def update_norm(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> None:
        """Fold velocity-critic inputs into the running norm."""
        if self.norm is not None:
            self.norm.update(self.features(state, command, xi))

    def forward(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Value ``[B]`` of a state under a velocity command."""
        packed = self.features(state, command, xi)
        if self.norm is not None:
            packed = self.norm(packed)
        return _mlp_forward(self, self.net, packed).squeeze(-1)


class SuperMimicModels(nn.Module):
    """Every SuperMimic network: dynamics, termination, policy, critics.

    Args:
        goal_dim: Width of the motion goal (680, 689 or 730).
        obs_dim: Width of one observation frame (93 or 99).
        velocity: Also build the velocity encoder and its critic.
        termination_bias_init: Initial termination-head bias; ``None`` keeps the default.
        critic_features: ``root4`` or ``zita``. Selects the motion-critic input width.
        input_norm: Running normalization on policy, critic and world-model inputs.
        output_norm: Running mean and std of the world-model residual.
    """

    def __init__(
        self,
        goal_dim: int,
        obs_dim: int = OBS_DIM,
        velocity: bool = False,
        termination_bias_init: float | None = None,
        critic_features: str = "root4",
        input_norm: bool = False,
        output_norm: bool = False,
    ):
        super().__init__()
        self.dynamics = DynamicsNet(input_norm, output_norm)
        self.encoder = MotionEncoder(goal_dim, input_norm)
        self.decoder = MotionDecoder(obs_dim, input_norm)
        self.critic = CriticNet(critic_input_dim(goal_dim, critic_features), input_norm)
        self.termination = TerminationHead(termination_bias_init)
        shared = self.decoder.obs_norm
        self.velocity_encoder = VelocityEncoder(obs_dim, shared) if velocity else None
        self.velocity_critic = VelocityCritic(input_norm) if velocity else None
        self.set_bf16_matmul(False)

    @torch.no_grad()
    def update_policy_norm(self, obs: torch.Tensor, goal: torch.Tensor) -> None:
        """Fold the observation frame and goal seen by the policy into the norms."""
        if self.decoder.obs_norm is not None:
            self.decoder.obs_norm.update(obs)
        if self.encoder.goal_norm is not None:
            self.encoder.goal_norm.update(goal)

    def set_bf16_matmul(self, enabled: bool) -> None:
        """Autocast every linear stack. Weights, Adam and stored states stay float32."""
        flag = bool(enabled)
        self.bf16_matmul = flag
        for module in (
            self.dynamics,
            self.encoder,
            self.decoder,
            self.critic,
            self.termination,
            self.velocity_encoder,
            self.velocity_critic,
        ):
            if module is not None:
                module.bf16_matmul = flag

    def policy(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Motion-tracking action from the history and the goal."""
        return self.decoder(self.encoder(goal), history)

    def velocity_policy(
        self, history: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """Velocity-tracking action from the history and the command."""
        if self.velocity_encoder is None:
            raise RuntimeError(
                "velocity_policy needs SuperMimicModels(velocity=True); these "
                "models were built with velocity=False"
            )
        return self.decoder(self.velocity_encoder(command, history), history)

    def wm_params(self) -> list[nn.Parameter]:
        """Parameters fit by the world-model loss (dynamics and termination head)."""
        return list(self.dynamics.parameters()) + list(self.termination.parameters())

    def actor_params(self) -> list[nn.Parameter]:
        """Parameters of the motion policy (encoder and decoder)."""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def velocity_actor_params(self) -> list[nn.Parameter]:
        """Velocity encoder parameters; empty without velocity tracking."""
        if self.velocity_encoder is None:
            return []
        return list(self.velocity_encoder.parameters())

    def critic_params(self) -> list[nn.Parameter]:
        """Parameters of the motion critic."""
        return list(self.critic.parameters())

    def velocity_critic_params(self) -> list[nn.Parameter]:
        """Velocity critic parameters; empty without velocity tracking."""
        if self.velocity_critic is None:
            return []
        return list(self.velocity_critic.parameters())
