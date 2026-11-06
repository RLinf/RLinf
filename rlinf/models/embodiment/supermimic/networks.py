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

"""SiLU MLPs of SuperMimic: world model, motion policy, critics, termination."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    OBS_DIM,
    RESIDUAL_DIM,
    STATE_DIM,
    XI_DIM,
)
from rlinf.envs.sim.g1_mimic.features import (
    error_floor,
    frame_local_error,
    heading_invariant_features,
    residual_target,
    scale_context,
)
from rlinf.models.embodiment.supermimic.normalization import RunningMeanStd


def _mlp_forward(
    owner: nn.Module, net: nn.Module, features: torch.Tensor
) -> torch.Tensor:
    """Run one linear stack, autocast to ``owner.autocast_dtype`` when set.

    The result is cast back to ``features.dtype``. Heading features,
    integration, kinematics and losses stay outside the autocast region.
    """
    dtype = getattr(owner, "autocast_dtype", None)
    if dtype is None:
        return net(features)
    with torch.autocast(device_type=features.device.type, dtype=dtype):
        out = net(features)
    return out.to(dtype=features.dtype)


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


def _fold(norm: RunningMeanStd | None, values, group) -> None:
    if norm is not None and values is not None:
        norm.fold(values, group)


class DynamicsNet(nn.Module):
    """Paper Table A.4: MLP hidden widths 2048, 2048, 2048. Last layer is zero.

    ``input_norm`` scales the context and applies a running norm to the MLP
    input. ``output_norm`` maps the MLP output back to a physical residual
    with the running mean and standard deviation of one-step targets.
    """

    def __init__(self, input_norm: bool = False, output_norm: bool = False):
        super().__init__()
        width = STATE_DIM + ACTION_DIM + XI_DIM
        self.net = _mlp([width, 2048, 2048, 2048, RESIDUAL_DIM])
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.in_norm = RunningMeanStd(width) if input_norm else None
        self.out_norm = RunningMeanStd(RESIDUAL_DIM, eps=1e-4) if output_norm else None

    def features(
        self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Heading-free state, action, and context. Running mode scales the context."""
        if self.in_norm is not None:
            xi = scale_context(xi)
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

    def norm_rows(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        xi: torch.Tensor,
        nxt: torch.Tensor,
        frame: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Rows the input and output norms fold. ``None`` where a norm is off."""
        feat = None if self.in_norm is None else self.features(state, action, xi)
        target = None if self.out_norm is None else residual_target(state, nxt, frame)
        return feat, target

    @torch.no_grad()
    def fold_norm_rows(
        self,
        feat: torch.Tensor | None,
        target: torch.Tensor | None,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Fold the rows of :meth:`norm_rows`, merged over ``group``."""
        _fold(self.in_norm, feat, group)
        _fold(self.out_norm, target, group)


def running_state_loss(pred: torch.Tensor, true: torch.Tensor, dynamics, frame: str):
    """Frame-local state error divided by the running residual std.

    Each component's scale is at least :func:`error_floor`. The result is the
    mean over the 70 components and over the batch.
    """
    err = frame_local_error(pred, true, frame)
    scale = torch.maximum(dynamics.out_norm.std(), error_floor(err))
    return (err / scale).square().mean()


def _normalize_history(norm: RunningMeanStd | None, history: torch.Tensor, obs_dim):
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
        self.goal_norm = RunningMeanStd(goal_dim) if input_norm else None

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
        self.obs_norm = RunningMeanStd(obs_dim) if input_norm else None

    def forward(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Action ``[B, 29]`` from a latent and a flat 10-frame history."""
        history = _normalize_history(self.obs_norm, history, self.obs_dim)
        return _mlp_forward(self, self.net, torch.cat((latent, history), dim=-1))


class CriticNet(nn.Module):
    """MLP(256, 128) on one prebuilt motion-critic feature vector."""

    def __init__(self, in_dim: int, input_norm: bool = False):
        super().__init__()
        self.net = _mlp([in_dim, 256, 128, 1])
        self.norm = RunningMeanStd(in_dim) if input_norm else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Value ``[B]`` from motion-critic features."""
        if self.norm is not None:
            features = self.norm(features)
        return self.value_mlp(features)

    def value_mlp(self, features: torch.Tensor) -> torch.Tensor:
        """MLP value. The caller applies ``self.norm`` when that module exists."""
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

    def __init__(self, obs_dim: int = OBS_DIM, obs_norm: RunningMeanStd | None = None):
        super().__init__()
        self.obs_dim = int(obs_dim)
        # Shared with the decoder. Registering it here as well would store and
        # update the same buffers under two names.
        object.__setattr__(self, "obs_norm", obs_norm)
        self.net = _mlp([3 + HISTORY_LEN * obs_dim, 256, 256, LATENT_DIM])

    def forward(self, command: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Latent ``[B, 32]`` of a normalized command and the history."""
        history = _normalize_history(self.obs_norm, history, self.obs_dim)
        packed = torch.cat((command, history), dim=-1)
        return torch.tanh(_mlp_forward(self, self.net, packed))


class VelocityCritic(nn.Module):
    """One critic for the velocity interface: heading state, command, and context."""

    def __init__(self, input_norm: bool = False):
        super().__init__()
        width = STATE_DIM + 3 + XI_DIM
        self.net = _mlp([width, 256, 128, 1])
        self.norm = RunningMeanStd(width) if input_norm else None

    def features(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Heading state, command, and context. Running mode scales the context."""
        if self.norm is not None:
            xi = scale_context(xi)
        return torch.cat((heading_invariant_features(state), command, xi), dim=-1)

    @torch.no_grad()
    def fold_norm(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        xi: torch.Tensor,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Fold velocity-critic inputs into the running norm, merged over ``group``.

        A rank without rows still has to call this when other ranks have rows.
        """
        if self.norm is not None:
            self.norm.fold(self.features(state, command, xi), group)

    def forward(
        self, state: torch.Tensor, command: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """Value ``[B]`` of a state under a velocity command."""
        packed = self.features(state, command, xi)
        if self.norm is not None:
            packed = self.norm(packed)
        return _mlp_forward(self, self.net, packed).squeeze(-1)
