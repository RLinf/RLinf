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

"""Frozen Prefix feature adapter for OpenPI evaluation models."""

from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.openpi.modules.model import preprocess_observation


class OpenPiPrefixMixin:
    """Expose Prefix features without adding parameters to the OpenPI model."""

    @property
    def z_dim(self) -> int:
        """Width of the frozen Prefix-FT feature exposed to Stage 2."""
        return int(
            self.rlt_cfg.rlt_embed_dim
            if self._resolved_prefix_pool() == "rlt_token"
            else self.rlt_cfg.rlt_input_dim
        )

    def _resolved_prefix_pool(self) -> str:
        from rlinf.models.embodiment.prefix.config import resolve_prefix_pool

        return resolve_prefix_pool(
            use_rlt=self.rlt_cfg.use_rlt,
            prefix_pool=self.rlt_cfg.prefix_pool,
        )

    def _stage2_requires_rlt(self) -> bool:
        return self._resolved_prefix_pool() == "rlt_token"

    def _encode_vlm_prefix_flat(
        self, prefix_output: torch.Tensor, prefix_mask: torch.Tensor
    ) -> torch.Tensor:
        from rlinf.models.embodiment.prefix.pool import pool_prefix

        pool = self._resolved_prefix_pool()
        if pool not in ("masked_mean", "mean", "last"):
            pool = "masked_mean" if self.rlt_cfg.rlt_use_mask else "mean"
        mask = prefix_mask if pool in ("masked_mean", "last") else None
        return pool_prefix(prefix_output, mask, mode=pool)

    def _encode_stage2_z(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        lang_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        selected_prefix, selected_mask = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, lang_tokens
        )
        if self._stage2_requires_rlt():
            return self._encode_rlt_flat(selected_prefix, selected_mask)
        return self._encode_vlm_prefix_flat(selected_prefix, selected_mask)

    @torch.no_grad()
    def extract_prefix_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head."""
        if self._stage2_requires_rlt():
            self._require_rlt()
        observation = self.env_obs_to_observation(env_obs)

        prepared_observation = preprocess_observation(observation, train=False)
        prefix_output, prefix_mask, kv_cache = self.build_prefix_cache(
            prepared_observation
        )
        z_rl = self._encode_stage2_z(
            prefix_output, prefix_mask, prepared_observation.tokenized_prompt
        ).to(dtype=torch.float32)

        model_actions = self._sample_actions_from_prefix_cache(
            prepared_observation,
            prefix_mask,
            kv_cache,
        )
        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }
