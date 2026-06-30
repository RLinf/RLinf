# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0 model for PyTorch, aligned with JAX models/pi0.py.

Flow matching model for continuous action generation.
"""

from __future__ import annotations

import logging

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, model, pointnet, siglip
from .pi0_config import Pi0Config
from .utils import _str_to_dtype

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] - true where next token starts a new autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(
        pos.dtype
    )


class Pi0(model.BaseModel):
    """Pi0 flow matching model for continuous action generation."""

    def __init__(self, config: Pi0Config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pcd = config.pcd
        self.embed_dtype = _str_to_dtype(config.dtype)
        self._config = config

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)

        # Gemma LLM with dual experts
        # Expert 0 (PaliGemma) uses regular RMSNorm; Expert 1 (Action Expert) may use adaRMS
        adarms = [False, config.pi05]
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=adarms,
            use_gradient_checkpointing=True,
        )

        # SigLIP vision encoder
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            remat=True,
            dtype_mm=config.dtype,
        )

        action_expert_width = action_expert_config.width
        self.action_dim = config.action_dim

        # Action input projection
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        # Optional PointNet
        if config.pcd:
            pointnet_config = pointnet.get_config(config.pointnet_variant)
            self.pointnet = pointnet.UncoloredPointNet(
                n_coordinates=pointnet_config.n_coordinates,
                output_dim=pointnet_config.output_dim,
                hidden_dim=pointnet_config.hidden_dim,
                hidden_depth=pointnet_config.hidden_depth,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

    def embed_prefix(
        self, obs: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix (images + language + optional point cloud).

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask (all False for prefix)
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Embed images through SigLIP
        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name])  # (B, num_patches, width)
            tokens.append(image_tokens)

            # Image tokens use bidirectional attention
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Add point cloud tokens
        if self.pcd and obs.pcd_xyz is not None:
            # pcd_xyz: (B, 16, 2025, 3)
            # PointNet expects (B, num_points, 3)
            B = obs.pcd_xyz.shape[0]
            pcd_flat = obs.pcd_xyz.reshape(B, -1, 3)  # (B, 16*2025, 3)
            pcd_tokens = self.pointnet(pcd_flat)  # (B, 16, 2048)
            # Reshape to match expected dimensions
            if pcd_tokens.dim() == 2:
                pcd_tokens = pcd_tokens.unsqueeze(1)  # (B, 1, 2048)

            tokens.append(pcd_tokens)
            input_mask.append(
                torch.ones(
                    pcd_tokens.shape[:2], dtype=torch.bool, device=pcd_tokens.device
                )
            )
            ar_mask += [False] * pcd_tokens.shape[1]

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed the suffix (state + noisy actions + time embedding).

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            input_mask: (B, S)
            ar_mask: (S,)
            adarms_cond: (B, emb_dim) or None
        """
        input_mask = []
        tokens = []

        B = noisy_actions.shape[0]

        # Cast to embed_dtype to ensure consistency between training and inference
        # (during training FSDP2 casts forward inputs, but inference may pass float32).
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)

        if not self.pi05:
            # Add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                torch.ones(B, 1, dtype=torch.bool, device=state_token.device)
            )

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        if self.pi05:
            # Time MLP for adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Mix timestep + action through MLP
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon
            )
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = F.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(
            torch.ones(
                action_expert_tokens.shape[:2],
                dtype=torch.bool,
                device=action_expert_tokens.device,
            )
        )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)

        # Build ar_mask with correct length matching input_mask.shape[1]
        ar_mask = torch.zeros(
            input_mask.shape[1], dtype=torch.bool, device=tokens.device
        )
        if not self.pi05:
            ar_mask[:2] = True
        else:
            ar_mask[0] = True

        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Returns:
            loss: (B, action_horizon) per-timestep MSE loss
        """
        B = actions.shape[0]
        device = actions.device

        # Preprocess first (requries float32 for image ops),
        # then cast to model dtype for FSDP2 mixed precision compatibility.
        observation = model.preprocess_observation(observation, train=train, rng=rng)

        embed_dtype = self.embed_dtype
        observation = model._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(
                actions.shape, device=device, dtype=dtype, generator=rng
            )
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((B,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass for prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return torch.mean(torch.square(v_t - u_t), dim=-1)

    def sample_actions(
        self,
        observation: model.Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        observation = model.preprocess_observation(observation, train=False)

        dt = -1.0 / num_steps
        B = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(
                B, self.action_horizon, self.action_dim, device=device, generator=rng
            )

        # Pre-fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1

        _, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )

        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, t_tensor
            )

            # Build attention mask: suffix attends to both prefix and suffix
            suffix_len = suffix_tokens.shape[1]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_len
            )
            full_attn_mask = torch.cat(
                [prefix_to_suffix_mask, suffix_attn_mask], dim=-1
            )

            suffix_positions = (
                torch.sum(prefix_mask, dim=-1)[:, None]
                + torch.cumsum(suffix_mask.int(), dim=-1)
                - 1
            )

            _, suffix_out = self.llm(
                [None, suffix_tokens],
                positions=suffix_positions,
                mask=full_attn_mask,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )[0]

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def forward(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward computes loss."""
        return self.compute_loss(
            observation, actions, train=train, rng=rng, noise=noise, time=time
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to the activation
                checkpoint. Currently honors ``use_reentrant`` (default ``False``), so
                the FSDP ``gradient_checkpointing_use_reentrant`` setting is respected.
        """
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = kwargs.get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
