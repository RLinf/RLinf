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

"""StreamingVLA's training-only SFP model.

The SFP objective is derived from the StreamingVLA OpenPI training fork. The
model owns its projections and flow-matching logic, while reusing RLinf's
unchanged, self-contained Gemma and SigLIP building blocks. It does not patch
the installed ``openpi`` or ``transformers`` packages.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma, siglip
from rlinf.models.embodiment.openpi_rlinf.pi0_model.utils import _str_to_dtype

from .openpi_compat.preprocessing import preprocess_observation_pytorch
from .sfp import (
    compute_sfp_flow_targets,
    create_sinusoidal_pos_embedding,
    make_attention_masks_and_position_ids,
    sample_beta,
)


class SVLAPytorch(nn.Module):
    """Pi0.5 backbone with StreamingVLA's one-token SFP training objective."""

    def __init__(
        self,
        config: Any,
        *,
        sigma: float = 0.16,
        noise_decay: float = 4.0,
        require_action_states: bool = True,
    ) -> None:
        super().__init__()
        if not bool(getattr(config, "pi05", False)):
            raise ValueError("StreamingVLA SFP training requires a Pi0.5 config.")
        if not bool(getattr(config, "use_sfp", False)):
            raise ValueError("StreamingVLA only supports use_sfp=True.")

        self.config = config
        self.sigma = float(sigma)
        self.noise_decay = float(noise_decay)
        self.require_action_states = bool(require_action_states)
        self.embed_dtype = _str_to_dtype(config.dtype)

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=[False, True],
            use_gradient_checkpointing=False,
        )
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            use_gradient_checkpointing=False,
            dtype_mm=config.dtype,
        )
        # Match the reference's mixed layout: Transformer/SigLIP parameters are
        # bf16 while SFP projections, trajectories, velocities, and loss stay
        # fp32. Loading a safetensors checkpoint preserves these target dtypes.
        self.llm.to(dtype=self.embed_dtype)
        self.img.to(dtype=self.embed_dtype)

        expert_width = action_expert_config.width
        self.action_in_proj = nn.Linear(config.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.action_dim)
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)
        self._init_weights()
        torch.set_float32_matmul_precision("high")

    def _init_weights(self) -> None:
        """Initialize projections using the Pi0.5 reference convention."""
        for layer in (
            self.action_in_proj,
            self.action_out_proj,
            self.time_mlp_in,
            self.time_mlp_out,
        ):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable activation checkpointing in private Gemma/SigLIP components."""
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = bool(kwargs.get("use_reentrant", False))
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False

    def is_gradient_checkpointing_enabled(self) -> bool:
        """Return whether Gemma activation checkpointing is enabled."""
        return bool(self.llm.gradient_checkpointing)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample reference ``Beta(1.5, 1.0)`` timesteps in fp32."""
        return (sample_beta(1.5, 1.0, batch_size, device) * 0.999 + 0.001).float()

    def _preprocess_observation(self, observation: Any, *, train: bool) -> Any:
        """Apply the isolated StreamingVLA image preprocessing implementation."""
        return preprocess_observation_pytorch(observation, train=train)

    def embed_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images and language into the PaliGemma prefix."""
        embeddings: list[Tensor] = []
        pad_masks: list[Tensor] = []
        categories: list[bool] = []
        for image, image_mask in zip(images, image_masks, strict=True):
            if image.ndim != 4:
                raise ValueError(f"image must be rank 4, got {tuple(image.shape)}.")
            if image.shape[1] == 3:
                image = image.permute(0, 2, 3, 1)
            image_embedding, _ = self.img(image)
            batch_size, token_count = image_embedding.shape[:2]
            embeddings.append(image_embedding)
            pad_masks.append(image_mask[:, None].expand(batch_size, token_count))
            categories.extend([False] * token_count)

        language_embedding = self.llm.embed(language_tokens)
        embeddings.append(language_embedding)
        pad_masks.append(language_masks.bool())
        categories.extend([False] * language_embedding.shape[1])

        prefix = torch.cat(embeddings, dim=1)
        prefix_pad = torch.cat(pad_masks, dim=1).bool()
        prefix_categories = torch.tensor(
            categories, dtype=torch.bool, device=prefix.device
        )[None, :].expand(prefix.shape[0], -1)
        return prefix, prefix_pad, prefix_categories

    def embed_suffix(
        self, noisy_position: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the single SFP position token and adaRMS condition."""
        position_embedding = self.action_in_proj(
            noisy_position.to(dtype=self.action_in_proj.weight.dtype)
        )
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        ).to(dtype=self.time_mlp_in.weight.dtype)
        condition = F.silu(self.time_mlp_in(time_embedding))
        condition = F.silu(self.time_mlp_out(condition))

        pad_mask = torch.ones(
            position_embedding.shape[:2],
            dtype=torch.bool,
            device=position_embedding.device,
        )
        categories = torch.ones_like(pad_mask)
        return position_embedding, pad_mask, categories, condition

    def forward(
        self,
        observation: Any,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Return the unreduced fp32 SFP loss with shape ``(B, 1, D)``."""
        action_states = observation.action_states
        if self.require_action_states and action_states is None:
            raise ValueError(
                "StreamingVLA requires action_states in every training sample."
            )

        processed = self._preprocess_observation(observation, train=True)
        images = list(processed.images.values())
        image_masks = list(processed.image_masks.values())
        if (
            processed.tokenized_prompt is None
            or processed.tokenized_prompt_mask is None
        ):
            raise ValueError("StreamingVLA requires tokenized prompts and masks.")

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time = time.to(device=actions.device, dtype=torch.float32)
        _, _, _, x_t, u_t, _ = compute_sfp_flow_targets(
            actions,
            action_states,
            time,
            noise,
            action_horizon=self.config.action_horizon,
            sigma=self.sigma,
            noise_decay=self.noise_decay,
        )

        prefix, prefix_pad, prefix_categories = self.embed_prefix(
            images,
            image_masks,
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
        )
        suffix, suffix_pad, suffix_categories, condition = self.embed_suffix(x_t, time)
        pad_mask = torch.cat([prefix_pad, suffix_pad], dim=1)
        categories = torch.cat([prefix_categories, suffix_categories], dim=1)
        attention, positions = make_attention_masks_and_position_ids(
            pad_mask, categories
        )

        outputs, _ = self.llm(
            [prefix, suffix],
            positions,
            attention,
            [None, condition],
        )
        suffix_output = outputs[1]
        if suffix_output is None:
            raise RuntimeError("Gemma action expert returned no suffix output.")
        velocity = self.action_out_proj(suffix_output[:, -1:].float())
        return F.mse_loss(u_t, velocity.float(), reduction="none")
