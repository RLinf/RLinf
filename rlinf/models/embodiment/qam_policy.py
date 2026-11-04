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


"""Flow policies and independent ensemble critics for QAM.

Architecture follows QAM commit 2726d767, implemented with PyTorch. This module
has no dependency on the reference repository or its JAX training state.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.base_policy import BasePolicy


class QAMMLP(nn.Module):
    """Dense, GELU(tanh), optional LayerNorm, then a linear output."""

    def __init__(self, input_dim, hidden_dims, output_dim, layer_norm=False):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.norms = nn.ModuleList(
            nn.LayerNorm(d, eps=1e-6) if layer_norm else nn.Identity()
            for d in hidden_dims
        )
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = norm(F.gelu(layer(x), approximate="tanh"))
        return self.layers[-1](x)


class QAMVectorField(QAMMLP):
    def forward(self, observations, actions, times):
        return super().forward(torch.cat((observations, actions, times), dim=-1))


class QAMEnsemble(nn.Module):
    """Batched independent critics; output shape is (ensemble, *batch)."""

    def __init__(self, input_dim, hidden_dims, num_qs, layer_norm=True):
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.scales = nn.ParameterList()
        self.offsets = nn.ParameterList()
        self.num_qs = num_qs
        self.layer_norm = layer_norm
        for a, b in zip(dims, dims[1:]):
            weight = nn.Parameter(torch.empty(num_qs, a, b))
            nn.init.uniform_(weight, -math.sqrt(6 / (a + b)), math.sqrt(6 / (a + b)))
            self.weights.append(weight)
            self.biases.append(nn.Parameter(torch.zeros(num_qs, 1, b)))
        if layer_norm:
            for d in hidden_dims:
                self.scales.append(nn.Parameter(torch.ones(num_qs, 1, d)))
                self.offsets.append(nn.Parameter(torch.zeros(num_qs, 1, d)))

    def forward(self, observations, actions):
        shape = observations.shape[:-1]
        x = torch.cat((observations, actions), -1).reshape(-1, self.weights[0].shape[1])
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, weight) + bias
            if i < len(self.weights) - 1:
                x = F.gelu(x, approximate="tanh")
                if self.layer_norm:
                    x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
                    x = x * self.scales[i] + self.offsets[i]
        return x.reshape(self.num_qs, *shape)


class QAMPolicy(nn.Module, BasePolicy):
    """QAM model implementing RLinf's embodied policy contract."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.action_dim = int(cfg.action_dim)
        self.horizon = int(cfg.horizon_length)
        self.full_action_dim = self.action_dim * self.horizon
        self.flow_steps = int(cfg.flow_steps)
        if self.flow_steps < 1 or self.horizon < 1:
            raise ValueError("QAM requires positive horizon_length and flow_steps")
        for key in ("residual", "fql_alpha", "edit_scale"):
            if cfg.get(key, False):
                raise ValueError(f"This QAM implementation requires {key}=0/False")
        if not cfg.get("action_chunking", True):
            raise ValueError("This QAM implementation requires action_chunking=True")
        field_args = (
            cfg.obs_dim + self.full_action_dim + 1,
            cfg.actor_hidden_dims,
            self.full_action_dim,
            cfg.actor_layer_norm,
        )
        self.actor_fast = QAMVectorField(*field_args)
        self.actor_slow = QAMVectorField(*field_args)
        self.critic = QAMEnsemble(
            cfg.obs_dim + self.full_action_dim,
            cfg.value_hidden_dims,
            cfg.num_qs,
            cfg.value_layer_norm,
        )
        self.target_actor_slow = copy.deepcopy(self.actor_slow).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)

    def forward(self, **kwargs):
        return self.default_forward(**kwargs)

    def default_forward(self, observations, actions, times, **kwargs):
        return self.actor_fast(observations, actions, times)

    def flow_actions(self, observations, noises):
        x = noises
        field = self.actor_slow if self.cfg.inv_temp == 0 else self.actor_fast
        for i in range(self.flow_steps):
            t = torch.full_like(x[..., :1], i / self.flow_steps)
            x = x + field(observations, x, t) / self.flow_steps
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_actions(self, observations, generator=None, noises=None):
        n = int(self.cfg.best_of_n)
        shape = (*observations.shape[:-1], n, self.full_action_dim)
        if noises is None:
            noises = torch.randn(shape, device=observations.device, generator=generator)
        expanded_obs = observations.unsqueeze(-2).expand(
            *shape[:-1], observations.shape[-1]
        )
        actions = self.flow_actions(expanded_obs, noises)
        if n == 1:
            return actions.squeeze(-2)
        indices = self.critic(expanded_obs, actions).mean(0).argmax(-1)
        return actions.gather(
            -2, indices[..., None, None].expand(*indices.shape, 1, self.full_action_dim)
        ).squeeze(-2)

    @torch.no_grad()
    def predict_action_batch(self, env_obs=None, observations=None, **kwargs):
        if observations is None:
            observations = env_obs["states"]
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=next(self.parameters()).device
        )
        actions = self.sample_actions(observations)
        chunks = actions.reshape(-1, self.horizon, self.action_dim)
        return chunks.cpu().numpy(), {"forward_inputs": {"observations": observations}}


def build_qam_model(cfg, torch_dtype=None):
    """Build the FP32 QAM model; higher action derivatives require FP32."""
    if torch_dtype not in (None, torch.float32):
        raise ValueError("QAM currently requires precision=fp32")
    return QAMPolicy(cfg)
