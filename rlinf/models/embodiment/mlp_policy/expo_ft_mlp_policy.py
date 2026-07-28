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

import torch
from torch.distributions.normal import Normal

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


class ExpoFTMLPPolicy(RLTMLPPolicy):
    """EXPO-FT residual actor and REDQ critic on top of RLT features."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        fixed_std: float = 1.0,
        num_q_heads: int = 10,
        num_base_candidates: int = 8,
        num_edit_samples: int = 8,
        num_min_qs: int = 2,
        edit_scale: float = 0.2,
        residual_logstd_min: float = -20.0,
        residual_logstd_max: float = 2.0,
    ):
        super().__init__(
            z_dim=z_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            ref_num_action_chunks=ref_num_action_chunks,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
            fixed_std=fixed_std,
            num_q_heads=num_q_heads,
        )
        self.num_q_heads = int(num_q_heads)
        self.num_base_candidates = int(num_base_candidates)
        self.num_edit_samples = int(num_edit_samples)
        self.num_min_qs = int(num_min_qs)
        self.edit_scale = float(edit_scale)
        self.logstd_range = (float(residual_logstd_min), float(residual_logstd_max))
        if self.num_base_candidates <= 0:
            raise ValueError(
                f"num_base_candidates must be positive, got {num_base_candidates}."
            )
        if self.num_edit_samples < 0:
            raise ValueError(
                f"num_edit_samples must be non-negative, got {num_edit_samples}."
            )
        if self.num_min_qs <= 0 or self.num_min_qs > self.num_q_heads:
            raise ValueError(
                "num_min_qs must be in [1, num_q_heads], got "
                f"{self.num_min_qs} for {self.num_q_heads} heads."
            )
        if self.edit_scale <= 0:
            raise ValueError(f"edit_scale must be positive, got {self.edit_scale}.")

    def _get_base_chunks(self, obs: dict) -> torch.Tensor:
        if "base_chunks" in obs:
            base_chunks = obs["base_chunks"]
            if base_chunks.dim() == 3:
                base_chunks = base_chunks[:, None]
            if base_chunks.dim() != 4:
                raise ValueError(
                    "base_chunks must have shape [B, N, H, A], got "
                    f"{tuple(base_chunks.shape)}."
                )
        else:
            ref_chunk = obs["ref_chunk"]
            if ref_chunk.dim() == 2:
                ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, self.step_action_dim)
            base_chunks = ref_chunk[:, None]
        base_chunks = base_chunks[:, : self.num_base_candidates, : self.chunk_len]
        return base_chunks.reshape(base_chunks.shape[0], base_chunks.shape[1], -1)

    def _actor_state_from_base(
        self,
        obs: dict,
        base_actions: torch.Tensor,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        base_actions = self._flatten_batch(base_actions)
        if apply_reference_dropout:
            base_actions = self._maybe_drop_reference(
                base_actions, reference_dropout_prob
            )
        return torch.cat([base_actions, self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def residual_forward(
        self,
        obs: dict,
        base_actions: torch.Tensor,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_state = self._actor_state_from_base(
            obs,
            base_actions,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        feat = self.backbone(actor_state)
        residual_mean = self.actor_mean(feat)
        residual_logstd = self.actor_logstd(feat)
        residual_logstd = torch.tanh(residual_logstd)
        residual_logstd = self.logstd_range[0] + 0.5 * (
            self.logstd_range[1] - self.logstd_range[0]
        ) * (residual_logstd + 1)
        residual_std = torch.exp(residual_logstd)
        probs = Normal(residual_mean, residual_std)
        raw_residual = residual_mean if deterministic else probs.rsample()
        residual = torch.tanh(raw_residual)
        log_probs = probs.log_prob(raw_residual)
        log_probs = log_probs - torch.log(1 - residual.pow(2) + 1e-6)
        log_probs = log_probs - torch.log(
            torch.as_tensor(self.edit_scale, device=log_probs.device, dtype=log_probs.dtype)
        )
        edited_actions = self._flatten_batch(base_actions) + self.edit_scale * residual
        return edited_actions, log_probs, residual

    def sac_forward(
        self,
        obs,
        base_actions: torch.Tensor | None = None,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        select_top_q: bool = False,
        q_indices: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        if select_top_q:
            action, info = self.select_top_q_actions(
                obs,
                q_indices=q_indices,
                deterministic_residual=deterministic,
            )
            return action, torch.zeros_like(action), info
        if base_actions is None:
            base_actions = self._get_ref_chunk(obs)
        action, chunk_logprobs, residual = self.residual_forward(
            obs,
            base_actions,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
            deterministic=deterministic,
        )
        return action, chunk_logprobs, residual

    def _critic_values_for_candidates(
        self,
        obs: dict,
        candidates: torch.Tensor,
        *,
        q_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_candidates, action_dim = candidates.shape
        critic_state = self._critic_state(obs)
        critic_state = critic_state[:, None].expand(-1, num_candidates, -1)
        q_values = self.q_head(
            critic_state.reshape(batch_size * num_candidates, -1),
            candidates.reshape(batch_size * num_candidates, action_dim),
        ).reshape(batch_size, num_candidates, -1)
        if q_indices is not None:
            q_values = q_values.index_select(dim=-1, index=q_indices)
        return q_values.min(dim=-1).values

    def _q_indices(self, *, mode: str) -> torch.Tensor | None:
        if self.num_min_qs >= self.num_q_heads:
            return None
        device = next(self.parameters()).device
        if mode == "eval":
            return torch.arange(self.num_min_qs, device=device)
        return torch.randperm(self.num_q_heads, device=device)[: self.num_min_qs]

    def select_top_q_actions(
        self,
        obs: dict,
        *,
        q_indices: torch.Tensor | None = None,
        deterministic_residual: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_chunks = self._get_base_chunks(obs)
        batch_size, num_base, flat_action_dim = base_chunks.shape
        num_edits = min(self.num_edit_samples, num_base)
        candidate_chunks = [base_chunks]

        residual_abs_mean = base_chunks.new_tensor(0.0)
        if num_edits > 0:
            edit_bases = base_chunks[:, :num_edits]
            flat_edit_bases = edit_bases.reshape(batch_size * num_edits, flat_action_dim)
            repeat_obs = {
                key: value[:, None]
                .expand(-1, num_edits, *value.shape[1:])
                .reshape(batch_size * num_edits, *value.shape[1:])
                if torch.is_tensor(value) and value.shape[0] == batch_size
                else value
                for key, value in obs.items()
            }
            edited, _, residual = self.residual_forward(
                repeat_obs,
                flat_edit_bases,
                deterministic=deterministic_residual,
            )
            candidate_chunks.append(edited.reshape(batch_size, num_edits, -1))
            residual_abs_mean = residual.detach().abs().mean()

        candidates = torch.cat(candidate_chunks, dim=1)
        candidate_scores = self._critic_values_for_candidates(
            obs, candidates, q_indices=q_indices
        )
        selected_indices = candidate_scores.argmax(dim=1)
        selected = candidates[torch.arange(batch_size, device=candidates.device), selected_indices]
        info = {
            "candidate_scores": candidate_scores,
            "selected_indices": selected_indices,
            "residual_abs_mean": residual_abs_mean,
        }
        return selected, info

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs=env_obs)
        selected, selection_info = self.select_top_q_actions(
            obs,
            q_indices=self._q_indices(mode=mode),
            deterministic_residual=(mode == "eval"),
        )
        chunk_actions = self._format_chunk_actions(selected)

        forward_inputs = {"action": selected, "model_action": selected}
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": torch.zeros_like(selected[..., :1]),
            "prev_values": selection_info["candidate_scores"].max(dim=1).values[:, None],
            "forward_inputs": forward_inputs,
            "expo_ft_selected_indices": selection_info["selected_indices"],
        }
        return chunk_actions, result
