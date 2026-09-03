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

Flow-matching VLA: assembles Gemma / SigLIP / action expert and exposes the
RLinf SFT ``forward(ForwardType.SFT, data=...)`` entry point. Task
subclasses (eval / RL / DAgger / DSRL) inherit this module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.modules import gemma, model, pointnet, siglip
from rlinf.models.embodiment.openpi_rlinf.modules.utils import (
    ComputeDtypeLinear,
    _str_to_dtype,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.rlt_config import OpenPiPytorchRLTConfig


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
    """Pi0 flow-matching model: network assembly plus RLinf SFT forward."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_env_dim: int | None = None,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
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
            use_gradient_checkpointing=False,
        )

        # SigLIP vision encoder
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            use_gradient_checkpointing=False,
            dtype_mm=config.dtype,
        )

        action_expert_width = action_expert_config.width
        self.action_dim = config.action_dim

        # Action / time projections. Cast input to the FSDP compute dtype
        # inside each module so ``precision: fp32`` (fp32 master) can still
        # run the original ``param_dtype: bf16`` matmuls.
        self.action_in_proj = ComputeDtypeLinear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = ComputeDtypeLinear(
                action_expert_width, action_expert_width
            )
            self.time_mlp_out = ComputeDtypeLinear(
                action_expert_width, action_expert_width
            )
        else:
            self.state_proj = ComputeDtypeLinear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = ComputeDtypeLinear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = ComputeDtypeLinear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = ComputeDtypeLinear(
            action_expert_width, config.action_dim
        )

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
        self._init_rlinf_runtime(
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            rlt_cfg=rlt_cfg,
        )
        # PI0Pytorch.__init__ sets this globally so fp32 action/value heads
        # use TF32. OpenPI RL keeps this even though it un-compiles sample_actions.
        torch.set_float32_matmul_precision("high")

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

        # Embed images through SigLIP in IMAGE_KEYS order (not dict iteration
        # order). Official OpenPI preprocess rebuilds cameras this way, then
        # concatenates ``list(observation.images.values())``.
        image_names = [name for name in model.IMAGE_KEYS if name in obs.images]
        image_names.extend(name for name in obs.images if name not in model.IMAGE_KEYS)
        for name in image_names:
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

        if not self.pi05:
            # Official PI0Pytorch: upcast state only when state_proj is fp32.
            state = obs.state
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_token = self.state_proj(state)[:, None, :]
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
            loss: (B, action_horizon, action_dim) per-element MSE
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

        v_t = self.velocity_from_suffix(suffix_out[:, -self.action_horizon :])

        return torch.square(v_t - u_t)

    def build_prefix_cache(
        self, observation: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Embed prefix tokens and run one LLM pass to build the KV cache.

        The caller is responsible for preprocessing the observation (image
        resize/pad, mask defaults) — this method only consumes the prepared
        observation so it can be shared between the eval Euler sampler and the
        RL train-time forward where the observation has already been built.

        Returns:
            prefix_out:  (B, prefix_len, paligemma_width) paligemma-side hidden states.
            prefix_mask: (B, prefix_len) bool mask of valid prefix positions.
            kv_cache:    per-layer KV cache to feed into subsequent suffix passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        return outputs[0], prefix_mask, kv_cache

    def run_suffix(
        self,
        observation: model.Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One suffix forward pass (action expert) given the prefix KV cache.

        Returns the action-expert hidden states sliced to the last
        ``action_horizon`` positions: (B, action_horizon, action_expert_width).
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, t_tensor
        )
        suffix_len = suffix_tokens.shape[1]
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn_mask = torch.cat([prefix_to_suffix_mask, suffix_attn_mask], dim=-1)
        suffix_positions = (
            torch.sum(prefix_mask, dim=-1)[:, None]
            + torch.cumsum(suffix_mask.int(), dim=-1)
            - 1
        )
        outputs, _ = self.llm(
            [None, suffix_tokens],
            positions=suffix_positions,
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        # Official PI0Pytorch casts suffix hidden states to fp32 before
        # ``action_out_proj`` / the value head.
        return outputs[1][:, -self.action_horizon :].to(dtype=torch.float32)

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        """Project action-expert hidden states to a velocity prediction v_t."""
        return self.action_out_proj(suffix_out_act.to(dtype=torch.float32))

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """OpenPI ``PaliGemmaWithExpertModel.to_bfloat16_for_selected_params``.

        OpenPI calls ``self.to(dtype)`` on ``paligemma_with_expert`` only
        (SigLIP + PaliGemma Gemma + action-expert Gemma), then restores a
        subset of those weights to fp32. Action / value heads live outside
        that module and must not be converted — so this uses ``llm`` +
        ``img``, never ``self.to()``.
        """
        # Same branch order as OpenPI: bf16 converts then falls through to
        # the fp32 restore; float32 converts and returns.
        if precision == "bfloat16":
            self.llm.to(dtype=torch.bfloat16)
            self.img.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.llm.to(dtype=torch.float32)
            self.img.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # 1:1 with OpenPI's substring list. Names come from
        # ``openpi_pytorch_to_openpi_rlinf``:
        #   patch_embedding.weight/bias  -> img.stem.weight/bias
        #   position_embedding.weight    -> img.pos_embedding
        #   input_layernorm              -> pre_attention_norms
        #   post_attention_layernorm     -> pre_ffw_norms
        #   model.norm                   -> final_norms
        #     (language_model.norm + gemma_expert.model.norm)
        params_to_keep_float32 = [
            "img.stem.weight",
            "img.stem.bias",
            "img.pos_embedding",
            "pre_attention_norms",
            "pre_ffw_norms",
            "final_norms",
        ]

        # OpenPI iterates paligemma_with_expert.named_parameters(); llm + img
        # are that module. Do not scan action/value heads.
        for name, param in self.named_parameters():
            if not (name.startswith("llm.") or name.startswith("img.")):
                continue
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

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

        _, prefix_mask, kv_cache = self.build_prefix_cache(observation)

        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            suffix_out_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_out_act)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def _init_rlinf_runtime(
        self,
        *,
        num_steps: int,
        action_env_dim: int | None,
        action_chunk: int | None,
        config_name: str,
        state_indices: Sequence[int] | None,
        rlt_cfg: OpenPiPytorchRLTConfig | None,
    ) -> None:
        """Attach RLinf SFT knobs (num_steps, optional RLT) without a wrapper."""
        self.num_steps = num_steps
        self.action_env_dim = (
            action_env_dim if action_env_dim is not None else self.action_dim
        )
        self.action_chunk = action_chunk
        self.config_name = config_name
        self.state_indices = list(state_indices) if state_indices else None
        # Workers (NFT, FSDP wrap) read ``model.config.num_steps``.
        self.config = SimpleNamespace(
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            action_env_dim=self.action_env_dim,
            config_name=config_name,
        )
        self.rlt_cfg = rlt_cfg or OpenPiPytorchRLTConfig()
        if self.rlt_cfg.use_rlt:
            from rlinf.models.embodiment.modules.rlt_token_transformer import (
                RLTTokenTransformer,
            )

            self.rlt_module = RLTTokenTransformer(
                input_dim=self.rlt_cfg.rlt_input_dim,
                embed_dim=self.rlt_cfg.rlt_embed_dim,
                prefix_seq_len=self.rlt_cfg.rlt_prefix_seq_len,
                num_layers=self.rlt_cfg.rlt_num_layers,
                num_heads=self.rlt_cfg.rlt_num_heads,
                mlp_ratio=self.rlt_cfg.rlt_mlp_ratio,
            ).to(dtype=next(self.parameters()).dtype)
        self._mark_fsdp_wrap_names()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _no_split_modules(self) -> list[str] | None:
        # FSDP1 FlatParameter requires uniform dtype per wrap unit, even with
        # use_orig_params=True. Dual-expert Block also mixes frozen/trainable
        # params, so the experiment yaml must set use_orig_params=True.
        # Mirror OpenPI: wrap GemmaRMSNorm / vision embeddings separately so
        # fp32 islands are not flattened with bf16.
        #   RMSNorm / Embedder     -- OpenPI GemmaRMSNorm
        #   Block                  -- leftover attn+mlp is bf16 after RMSNorm
        #   Encoder1DBlock         -- SigLIP layer, all bf16
        #   Encoder                -- leftover post_layernorm is bf16
        #   SigLIPViT              -- leftover pos_embedding is fp32 after
        #                            stem / head wrap
        names = [
            "Block",
            "Encoder1DBlock",
            "Encoder",
            "RMSNorm",
            "Embedder",
            "SigLIPViT",
        ]
        if self.rlt_cfg.use_rlt:
            names.append("RLTSelfAttentionLayer")
        return names

    @property
    def _no_split_names(self) -> list[str] | None:
        return [
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "time_mlp_in",
            "time_mlp_out",
            "stem",  # fp32 patch embedding (OpenPI SiglipVisionEmbeddings)
            "head",  # bf16 multi-modal projector
        ]

    def _mark_fsdp_wrap_names(self) -> None:
        """Mark modules so RLinf's FSDP lambda policy can find leaf projects."""
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def _require_rlt(self) -> None:
        if not self.rlt_cfg.use_rlt or not hasattr(self, "rlt_module"):
            raise ValueError("RLT operation requires actor.model.openpi.use_rlt=True.")

    def _select_rlt_prefix_embeddings(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        lang_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rlt_cfg.rlt_image_only and lang_tokens is not None:
            num_image_tokens = prefix_output.shape[1] - lang_tokens.shape[1]
            prefix_output = prefix_output[:, :num_image_tokens]
            prefix_mask = prefix_mask[:, :num_image_tokens]
        return prefix_output, prefix_mask

    def _rlt_forward(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module(prefix_output, rlt_mask)

    def _encode_rlt_flat(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module.encode_flat(prefix_output, rlt_mask)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any]:
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions); "
                    f"got length {len(data)}."
                )
            observation, actions = data
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions

    def _observation_to_device(self, observation: Any) -> model.Observation:
        observation = model.Observation.from_observation_like(observation)
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return model.Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == self.action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        raise ValueError(
            "SFT actions must arrive normalized + padded to the model action "
            f"dim {self.action_dim}; got last dim {actions.shape[-1]}."
        )

    def _reduce_sft_loss(
        self, per_element_loss: torch.Tensor, use_action_chunk_loss: bool
    ) -> torch.Tensor:
        """Mean flow-matching MSE, optionally restricted to the env action chunk.

        ``per_element_loss`` is ``[B, action_horizon, action_dim]``. DAgger
        matches the old OpenPI wrapper by dropping padded action dims and
        unexecuted horizon steps before the mean.
        """
        if use_action_chunk_loss:
            horizon = (
                self.action_chunk
                if self.action_chunk is not None
                else per_element_loss.shape[1]
            )
            env_dim = (
                self.action_env_dim
                if self.action_env_dim is not None
                else per_element_loss.shape[-1]
            )
            per_element_loss = per_element_loss[:, :horizon, :env_dim]
        return per_element_loss.mean()

    def _sft_forward_with_rlt_prefix(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VLA loss while retaining the prefix hidden states for RLT."""
        batch_size = actions.shape[0]
        device = actions.device

        observation = model.preprocess_observation(observation, train=True)
        observation = model._observation_to_dtype(observation, self.embed_dtype)
        actions = actions.to(dtype=self.embed_dtype)
        dtype = actions.dtype

        noise = torch.randn(actions.shape, device=device, dtype=dtype)
        time = (
            torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            .sample((batch_size,))
            .to(device=device, dtype=dtype)
        )
        time = time * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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
        v_t = self.velocity_from_suffix(suffix_out[:, -self.action_horizon :])
        loss = torch.square(v_t - u_t)
        prefix_out, prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_out.detach(), prefix_mask, observation.tokenized_prompt
        )
        return loss, prefix_out, prefix_mask

    def sft_forward(
        self, data: Any, use_action_chunk_loss: bool = False, **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Flow-matching SFT loss. Shared by SFT, DAgger, and PPO co-train."""
        del kwargs
        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()
        observation, actions = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        if not self.rlt_cfg.use_rlt:
            per_element_loss = self.compute_loss(observation, actions, train=True)
            return self._reduce_sft_loss(per_element_loss, use_action_chunk_loss)

        per_element_loss, prefix_output, prefix_mask = (
            self._sft_forward_with_rlt_prefix(observation, actions)
        )
        vla_loss = self._reduce_sft_loss(per_element_loss, use_action_chunk_loss)
        rlt_loss, _ = self._rlt_forward(prefix_output, prefix_mask)
        return {
            "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
        }

    def freeze_vlm(self, freeze_action_expert: bool = False) -> int:
        """Freeze PaliGemma (vision + LLM expert-0). Optionally freeze expert-1.

        OpenPI also puts the frozen paligemma trunk in ``eval()``.
        """
        self.img.eval()
        frozen = 0
        for p in self.img.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        llm = self.llm
        for p in llm.embedder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        expert_ids = (0, 1) if freeze_action_expert else (0,)
        for block in llm.layers:
            for expert_id in expert_ids:
                for sub in (
                    block.pre_attention_norms[expert_id],
                    block.pre_ffw_norms[expert_id],
                    block.mlps[expert_id],
                ):
                    for p in sub.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen += 1
                attn = block.attn
                for proj_list in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                    proj = proj_list[expert_id]
                    if proj is None:
                        continue
                    for p in proj.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen += 1
        for expert_id in expert_ids:
            if llm.final_norms[expert_id] is not None:
                for p in llm.final_norms[expert_id].parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
        if freeze_action_expert:
            for name in (
                "action_in_proj",
                "action_out_proj",
                "state_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
                "time_mlp_in",
                "time_mlp_out",
            ):
                module = getattr(self, name, None)
                if module is None:
                    continue
                for p in module.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
        return frozen

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}."
            )
        return self.sft_forward(**kwargs)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None, **kwargs
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

    def gradient_checkpointing_disable(self, **kwargs):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
