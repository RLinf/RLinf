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
import torch.nn as nn
from torch.distributions.normal import Normal

from .batch_renorm import BatchRenorm


class FlowTActor(nn.Module):
    """
    Transformer-based Flow Matching Actor for SAC
    Uses transformer architecture with cross-attention between action and observation
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.log_std_min = -5
        self.log_std_max = 2
        self.use_batch_norm = use_batch_norm

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Action input projection (projects action_dim -> d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.action_dim)
        self.velocity_log_std_head = nn.Linear(self.d_model, self.action_dim)

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, train=False, log_grad=False):
        if log_grad:
            self.grad_norms.clear()

        batch_size = obs.shape[0]
        device = obs.device

        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps

        # 1. Observation encoding (memory)
        # obs: [batch_size, obs_dim] -> obs_emb: [batch_size, d_model]
        if self.use_batch_norm:
            obs = self.bn_obs(obs, train)
        obs_emb = self.obs_encoder(obs)
        # Add sequence dimension: [batch_size, 1, d_model]
        obs_emb = obs_emb.unsqueeze(1)

        x_current = torch.randn((batch_size, self.action_dim), device=device)

        # Calculate x0 log probability under N(0, I) using torch.distributions
        initial_dist = Normal(torch.zeros_like(x_current), torch.ones_like(x_current))
        total_log_prob = initial_dist.log_prob(x_current).sum(dim=1, keepdim=True)

        # 3. Flow Matching iterative refinement
        for step in range(self.denoising_steps):
            # 3a. Project current action to embedding space
            # x_current: [batch_size, action_dim] -> x_input: [batch_size, 1, action_dim]
            if self.use_batch_norm:
                x_bn = self.bn_action(x_current, train)
            else:
                x_bn = x_current
            x_input_bn = x_bn.unsqueeze(1)
            # action_emb: [batch_size, 1, d_model]
            action_emb = self.action_proj(x_input_bn)

            # 3b. Add time embedding
            time_value = torch.full(
                (batch_size, 1, 1),
                step / self.denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            # time_emb: [batch_size, 1, d_model]
            time_emb = self.time_embedding(time_value)

            # 3c. Combine action and time to form query (tgt)
            # input_emb: [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # 3d. Create diagonal mask
            # For a single query position (seq_len=1), we don't need
            # to mask future positions. A mask of 0s is fine.
            # PyTorch's mask should be (L, L) -> (1, 1)
            diagonal_mask = torch.zeros(1, 1, device=device)

            # 3e. Transformer forward pass
            output = input_emb
            for layer in self.transformer_layers:
                # tgt=output, memory=obs_emb, tgt_mask=diagonal_mask
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output is [batch_size, 1, d_model], squeeze to [batch_size, d_model]
            output = output.squeeze(1)

            # 3f. Predict velocity
            velocity_mean = self.velocity_mean_head(output)
            velocity_log_std = self.velocity_log_std_head(output)

            # Clamp log_std
            velocity_log_std = torch.tanh(velocity_log_std)
            velocity_log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (velocity_log_std + 1)
            velocity_std = torch.exp(velocity_log_std)

            # 3g. Sample velocity
            u_dist = Normal(velocity_mean, velocity_std)
            predicted_velocity = u_dist.rsample()

            velocity_log_prob = u_dist.log_prob(predicted_velocity).sum(
                dim=-1, keepdim=True
            )
            total_log_prob += velocity_log_prob

            # 3i. Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_current = x_current + predicted_velocity * DELTA_T

            # Add gradient logging hook in style of Actor
            if log_grad:
                current_step_for_hook = step
                x_current.register_hook(
                    lambda grad, s=current_step_for_hook: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        # 4. Apply tanh transformation and scaling
        y_t = torch.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias

        # 5. Add Jacobian correction for tanh
        # Use 1e-6 to match JAX implementation
        tanh_correction = torch.sum(
            torch.log(self.action_scale * (1 - y_t**2) + 1e-6), dim=-1, keepdim=True
        )
        total_log_prob -= tanh_correction

        return action, total_log_prob.detach()


class JaxFlowTActor(nn.Module):
    """
    JAX-style Flow Matching Actor (uses noise sampling instead of distribution sampling)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
        noise_std_head=False,
        log_std_min_train=-5,
        log_std_max_train=2,
        log_std_min_rollout=-5,
        log_std_max_rollout=2,
        noise_std_train=0.3,
        noise_std_rollout=0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm
        # Whether to use fixed noise std, otherwise predict std via velocity_log_std_head
        self.noise_std_head = noise_std_head
        # Different noise std for train/rollout, smaller noise during rollout.
        self.log_std_min_train = log_std_min_train
        self.log_std_max_train = log_std_max_train
        self.log_std_min_rollout = log_std_min_rollout
        self.log_std_max_rollout = log_std_max_rollout
        # Fixed noise std added directly to actions
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Action input projection (projects action_dim -> d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.action_dim)
        # Use a specific head to predict velocity_log_std
        if self.noise_std_head:
            self.velocity_log_std_head = nn.Linear(self.d_model, self.action_dim)

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, train=False, log_grad=False):
        if log_grad:
            self.grad_norms.clear()

        batch_size = obs.shape[0]
        device = obs.device

        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps

        # 1. Observation encoding (memory)
        # obs: [batch_size, obs_dim] -> obs_emb: [batch_size, d_model]
        if self.use_batch_norm:
            obs = self.bn_obs(obs, train)
        obs_emb = self.obs_encoder(obs)
        # Add sequence dimension: [batch_size, 1, d_model]
        obs_emb = obs_emb.unsqueeze(1)

        x_current = torch.randn((batch_size, self.action_dim), device=device)

        # Calculate x0 log probability under N(0, I) using torch.distributions
        initial_dist = Normal(torch.zeros_like(x_current), torch.ones_like(x_current))
        total_log_prob = initial_dist.log_prob(x_current).sum(dim=1, keepdim=True)

        if self.noise_std_head:
            log_std_min = self.log_std_min_train if train else self.log_std_min_rollout
            log_std_max = self.log_std_max_train if train else self.log_std_max_rollout
        else:
            noise_std = self.noise_std_train if train else self.noise_std_rollout

        # 3. Flow Matching iterative refinement
        for step in range(self.denoising_steps):
            # 3a. Project current action to embedding space
            # x_current: [batch_size, action_dim] -> x_input: [batch_size, 1, action_dim]
            if self.use_batch_norm:
                x_bn = self.bn_action(x_current, train)
            else:
                x_bn = x_current
            x_input_bn = x_bn.unsqueeze(1)
            # action_emb: [batch_size, 1, d_model]
            action_emb = self.action_proj(x_input_bn)

            # 3b. Add time embedding
            time_value = torch.full(
                (batch_size, 1, 1),
                step / self.denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            # time_emb: [batch_size, 1, d_model]
            time_emb = self.time_embedding(time_value)

            # 3c. Combine action and time to form query (tgt)
            # input_emb: [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # 3d. Create diagonal mask
            # For a single query position (seq_len=1), we don't need
            # to mask future positions. A mask of 0s is fine.
            # PyTorch's mask should be (L, L) -> (1, 1)
            diagonal_mask = torch.zeros(1, 1, device=device)

            # 3e. Transformer forward pass
            output = input_emb
            for layer in self.transformer_layers:
                # tgt=output, memory=obs_emb, tgt_mask=diagonal_mask
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output is [batch_size, 1, d_model], squeeze to [batch_size, d_model]
            output = output.squeeze(1)

            # Choice A: use NN predicted velocity_log_std, add noise to velocity
            if self.noise_std_head:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)
                velocity_log_std = self.velocity_log_std_head(output)

                # Clamp log_std
                velocity_log_std = torch.tanh(velocity_log_std)
                velocity_log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (
                    velocity_log_std + 1
                )
                velocity_std = torch.exp(velocity_log_std)

                # 3g. Sample velocity (JAX style: sample noise first, then add)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample()

                predicted_velocity = velocity_mean + velocity_std * noise

                velocity_log_prob = noise_dist.log_prob(noise).sum(dim=-1, keepdim=True)
                total_log_prob += velocity_log_prob

                # 3i. Flow Matching update: x_{t+1} = x_t + v_t * Δt
                x_current = x_current + predicted_velocity * DELTA_T

            # Choice B: use fixed noise_std, add noise to action
            else:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)

                # 3g. Euler step (no noise, deterministic)
                x_next_mean = x_current + velocity_mean * DELTA_T

                # 3h. Add noise to actions (not velocity)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample((batch_size, self.action_dim)).to(device)
                x_current = x_next_mean + noise_std * noise

                # 3i. calculate log prob
                step_log_prob = (
                    Normal(x_next_mean, noise_std)
                    .log_prob(x_current)
                    .sum(dim=-1, keepdim=True)
                )
                total_log_prob += step_log_prob

            # Add gradient logging hook in style of Actor
            if log_grad:
                current_step_for_hook = step
                x_current.register_hook(
                    lambda grad, s=current_step_for_hook: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        # 4. Apply tanh transformation and scaling
        y_t = torch.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias

        # 5. Add Jacobian correction for tanh
        tanh_correction = torch.sum(
            torch.log(self.action_scale * (1 - y_t**2) + 1e-6), dim=-1, keepdim=True
        )
        total_log_prob -= tanh_correction

        return action, total_log_prob
