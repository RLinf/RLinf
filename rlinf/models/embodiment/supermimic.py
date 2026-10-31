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

"""SiLU MLPs for SuperMimic. Not registered in EMBODIED_MODEL.

Trainer and evaluator construct ``SuperMimicModels`` directly. This is not
a generic embodied policy, so it is not added to that registry.
"""

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


def _batch_rows(values: torch.Tensor) -> int:
    """Rows of ``values`` once the leading dimensions are flattened."""
    if values.numel() == 0:
        return 0
    return int(values.reshape(-1, values.shape[-1]).shape[0])


def _as_count(value: torch.Tensor, host: float | None) -> float:
    """Python count. ``host`` avoids reading a CUDA scalar back to the CPU."""
    return float(value) if host is None else host


def _chan_merge(
    n_a: torch.Tensor,
    mean_a: torch.Tensor,
    var_a: torch.Tensor,
    n_b: torch.Tensor,
    mean_b: torch.Tensor,
    var_b: torch.Tensor,
    host_a: float | None = None,
    host_b: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two population-variance batches. A non-positive count is skipped."""
    n_a = n_a.reshape(())
    n_b = n_b.reshape(())
    if _as_count(n_b, host_b) <= 0.0:
        return n_a, mean_a, var_a
    if _as_count(n_a, host_a) <= 0.0:
        return n_b, mean_b, var_b
    total = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / total)
    var = (var_a * n_a + var_b * n_b + delta.square() * (n_a * n_b / total)) / total
    return total, mean, var


def merge_welford_batches(
    counts: list[torch.Tensor],
    means: list[torch.Tensor],
    variances: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chan-merge population batches into one count, mean, and variance.

    A zero count is skipped. In exact arithmetic the result is the mean and
    population variance of the concatenated rows. Ranks must fold this once
    into the same previous state; averaging the already-updated variances
    double-counts that state and zeroes a variance that is only across ranks.
    """
    if not counts:
        raise ValueError("merge_welford_batches needs at least one batch")
    acc_n = counts[0].detach().reshape(()).to(dtype=torch.float64) * 0
    acc_mean = means[0].detach().to(dtype=torch.float64) * 0
    acc_var = variances[0].detach().to(dtype=torch.float64) * 0
    for count, mean, var in zip(counts, means, variances, strict=True):
        acc_n, acc_mean, acc_var = _chan_merge(
            acc_n,
            acc_mean,
            acc_var,
            count.detach().to(dtype=torch.float64),
            mean.detach().to(dtype=torch.float64),
            var.detach().to(dtype=torch.float64),
        )
    return acc_n, acc_mean, acc_var


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

    ``update`` folds one local batch. ``update_from_stats`` folds a batch the
    caller has already merged across ranks. zita-ch freezes the statistics
    after ``max_count`` samples. ``eps`` is 1e-2 on inputs and 1e-4 on the
    world-model residual.
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = float(max_count)
        self.eps = float(eps)
        # Mirror of ``count`` kept on the CPU. The buffer is the checkpoint
        # value; this copy is what the freeze check reads, so a hot update
        # does not wait for the GPU to finish the previous kernel.
        self._count_cpu = 0.0

        def _restore_cpu_count(module, _incompatible) -> None:
            module._count_cpu = float(module.count.detach().cpu())

        self.register_load_state_dict_post_hook(_restore_cpu_count)

    def batch_stats(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count, mean, and population variance of one batch.

        An empty batch has count 0. One row has variance 0.
        """
        flat = values.detach().reshape(-1, values.shape[-1]).float()
        if flat.shape[0] == 0:
            zeros = torch.zeros(self.mean.shape, device=flat.device, dtype=flat.dtype)
            return torch.zeros((), device=flat.device, dtype=flat.dtype), zeros, zeros
        return (
            torch.tensor(flat.shape[0], device=flat.device, dtype=flat.dtype),
            flat.mean(0),
            flat.var(0, unbiased=False),
        )

    def frozen(self) -> bool:
        """Whether ``max_count`` samples have already been folded."""
        return self._count_cpu >= self.max_count

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Fold one local batch into the running statistics."""
        if self.frozen():
            return
        rows = _batch_rows(values)
        if rows == 0:
            return
        count, mean, var = self.batch_stats(values)
        self.update_from_stats(count, mean, var, host_count=float(rows))

    @torch.no_grad()
    def update_from_stats(
        self,
        count: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        host_count: float | None = None,
    ) -> None:
        """Fold one batch given its count, mean, and population variance.

        A count of 0 leaves the buffers unchanged. At ``max_count`` the fold
        is skipped, as in ``update``. ``host_count`` is that count already
        known on the CPU; without it the count tensor is read back once.
        """
        if self.frozen():
            return
        if host_count is None:
            host_count = float(count)
        if host_count <= 0.0:
            return
        count = count.detach().to(device=self.count.device, dtype=self.count.dtype)
        mean = mean.detach().to(device=self.mean.device, dtype=self.mean.dtype)
        var = var.detach().to(device=self.var.device, dtype=self.var.dtype)
        total, new_mean, new_var = _chan_merge(
            self.count,
            self.mean,
            self.var,
            count,
            mean,
            var,
            host_a=self._count_cpu,
            host_b=host_count,
        )
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total.reshape(()))
        added = torch.tensor(self._count_cpu, dtype=self.count.dtype) + torch.tensor(
            host_count, dtype=self.count.dtype
        )
        self._count_cpu = float(added)

    def std(self) -> torch.Tensor:
        """Standard deviation including ``eps``."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize ``values`` on the last dimension."""
        return (values - self.mean) / self.std()


def _fold_norm(
    norm: RunningNorm | None, values: torch.Tensor, parallel: object | None
) -> None:
    """Fold ``values`` into ``norm``. ``parallel`` merges the batch across ranks."""
    if norm is None:
        return
    if parallel is None:
        norm.update(values)
    else:
        parallel.update_norm(norm, values)


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
        parallel: object | None = None,
    ) -> None:
        """Fold one real transition into the input and output running norms.

        ``parallel`` merges each rank's batch before the fold. ``None`` uses
        this process's rows alone.
        """
        if self.in_norm is not None:
            _fold_norm(self.in_norm, self.features(state, action, xi), parallel)
        if self.out_norm is not None:
            _fold_norm(self.out_norm, residual_target(state, nxt, frame), parallel)


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
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        xi: torch.Tensor,
        parallel: object | None = None,
    ) -> None:
        """Fold velocity-critic inputs into the running norm.

        ``parallel`` merges each rank's batch before the fold. ``None`` uses
        this process's rows alone. An empty batch still has to be passed in
        when other ranks have rows.
        """
        if self.norm is not None:
            _fold_norm(self.norm, self.features(state, command, xi), parallel)

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
    def update_policy_norm(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        parallel: object | None = None,
    ) -> None:
        """Fold the observation frame and goal seen by the policy into the norms.

        ``parallel`` merges each rank's batch before the fold.
        """
        _fold_norm(self.decoder.obs_norm, obs, parallel)
        _fold_norm(self.encoder.goal_norm, goal, parallel)

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
