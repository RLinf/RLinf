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

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.compact_encoders import (
    CompactMultiQHead,
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy
from rlinf.models.embodiment.openpi_rlinf.env_io import EnvIO
from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class Pi0DSRLConfig:
    state_dim: int = 8
    action_noise_dim: int = 32
    num_q_heads: int = 10
    image_latent_dim: int = 64
    state_latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (128, 128, 128)


class Pi0DSRL(EnvIO, Pi0):
    """Off-policy DSRL: SAC in noise space with a frozen Pi0 decoder."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_env_dim: int | None = None,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        dsrl_cfg: Pi0DSRLConfig | None = None,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
        )
        cfg = dsrl_cfg or Pi0DSRLConfig()
        self.dsrl_cfg = cfg
        dsrl_dtype = torch.bfloat16
        input_dim = cfg.state_latent_dim + cfg.image_latent_dim
        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=input_dim,
            output_dim=cfg.action_noise_dim,
            hidden_dims=cfg.hidden_dims,
            low=None,
            high=None,
            action_horizon=self.action_horizon,
        ).to(dtype=dsrl_dtype)
        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=cfg.image_latent_dim, image_size=64
        ).to(dtype=dsrl_dtype)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=cfg.state_dim, hidden_dim=cfg.state_latent_dim
        ).to(dtype=dsrl_dtype)
        self.critic_image_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=cfg.image_latent_dim, image_size=64
        ).to(dtype=dsrl_dtype)
        self.critic_state_encoder = CompactStateEncoder(
            state_dim=cfg.state_dim, hidden_dim=cfg.state_latent_dim
        ).to(dtype=dsrl_dtype)
        self.q_head = CompactMultiQHead(
            state_dim=cfg.state_latent_dim,
            image_dim=cfg.image_latent_dim,
            action_dim=cfg.action_noise_dim,
            hidden_dims=cfg.hidden_dims,
            num_q_heads=cfg.num_q_heads,
            output_dim=1,
        ).to(dtype=dsrl_dtype)
        self._mark_fsdp_wrap_names()

    @property
    def _no_split_modules(self) -> list[str] | None:
        names = super()._no_split_modules or []
        return list(names) + [
            "GaussianPolicy",
            "CompactMultiQHead",
            "LightweightImageEncoder64",
            "CompactStateEncoder",
        ]

    def _normalize_dsrl_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "images" in obs:
            return obs
        if "main_images" in obs:
            return {"images": [obs["main_images"]], "states": obs["states"]}
        raise ValueError(
            f"Invalid DSRL obs format: {obs.keys()}. Expected 'images' or 'main_images'."
        )

    def _preprocess_dsrl_images(self, images, train=False):
        del train
        agentview_img = images[0] if isinstance(images, list) else images
        if agentview_img.shape[-1] == 3:
            agentview_img = agentview_img.permute(0, 3, 1, 2)
        if agentview_img.dtype == torch.uint8:
            agentview_img = agentview_img.float() / 255.0
        elif agentview_img.min() < 0:
            agentview_img = (agentview_img + 1.0) / 2.0
        agentview_img = agentview_img.clamp(0.0, 1.0)
        resized = F.interpolate(
            agentview_img, size=(64, 64), mode="bilinear", align_corners=False
        )
        return (resized * 2.0 - 1.0).unsqueeze(1)

    def _preprocess_states(self, states):
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        return states.to(torch.bfloat16)

    def _actor_features(self, obs: dict[str, Any], train: bool = False):
        obs = self._normalize_dsrl_obs(obs)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])
        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        image_features = self.actor_image_encoder(images)
        state_features = self.actor_state_encoder(states)
        return torch.cat([state_features, image_features], dim=-1)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        dsrl_obs = {"images": [env_obs["main_images"]], "states": env_obs["states"]}
        noise_actions, noise_logprob, _ = self.sac_forward(
            obs=dsrl_obs, train=False, mode=mode
        )
        observation = self.env_obs_to_observation(env_obs)
        noise = noise_actions.to(dtype=torch.float32)
        if noise.shape[-1] != self.action_dim:
            pad = self.action_dim - noise.shape[-1]
            if pad < 0:
                noise = noise[..., : self.action_dim]
            else:
                noise = F.pad(noise, (0, pad))
        actions, result = self._predict_eval(observation, noise=noise)
        result["prev_logprobs"] = noise_logprob
        # SAC trains on noise, not the decoded env action. Keep the unflattened
        # [B, H, noise_dim] tensor so ``sac_q_forward`` can take ``actions[:, 0]``.
        result["forward_inputs"]["action"] = noise_actions
        return actions, result

    def forward(self, forward_type: ForwardType = ForwardType.SAC, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"{type(self).__name__} only supports SAC / SAC_Q; got {forward_type!r}."
        )

    def sac_forward(
        self, obs=None, data=None, train=False, return_dist_params=False, **kwargs
    ):
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        features = self._actor_features(obs, train=train)
        deterministic = kwargs.get("mode", "train") == "eval"
        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )
        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)
        return action_noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")
        obs = self._normalize_dsrl_obs(obs)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])
        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        actions = actions.to(device=device, dtype=torch.bfloat16)
        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()
        if actions.dim() == 3:
            actions = actions[:, 0, :]
        return self.q_head(state_features, image_features, actions)
