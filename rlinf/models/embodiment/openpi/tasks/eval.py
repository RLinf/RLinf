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

from typing import Any, Literal, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi.env_io import EnvIO
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig
from rlinf.models.embodiment.openpi.sfp_config import OpenPiPytorchSfpConfig


def env_reset_mask(
    dones: Any | None,
    episode_starts: Any | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Combine batched done and episode-start flags into a reset mask."""
    reset = None
    for field_name, flags in (("dones", dones), ("episode_starts", episode_starts)):
        if flags is None:
            continue
        mask = torch.as_tensor(flags, device=device)
        if mask.dtype != torch.bool:
            mask = mask != 0
        while mask.ndim > 1:
            mask = mask.any(dim=-1)
        if mask.shape != (batch_size,):
            raise ValueError(
                f"SFP {field_name} must collapse to shape {(batch_size,)}; "
                f"got {tuple(mask.shape)}."
            )
        reset = mask if reset is None else reset | mask
    return reset


class Pi0Eval(EnvIO, Pi0):
    """Inference-only: flow-matching Euler, SFP trajectory sampling, or RTC."""

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
        sfp_cfg: OpenPiPytorchSfpConfig | None = None,
        rtc_enabled: bool = False,
        rtc_guidance_mode: str = "approx",
        rtc_guidance_clip: float = 5.0,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            rlt_cfg=rlt_cfg,
            sfp_cfg=sfp_cfg,
        )
        if self.sfp_cfg.use_sfp and rtc_enabled:
            raise ValueError(
                "actor.model.openpi.use_sfp cannot be combined with RTC: "
                "RTC guidance integrates the flow-matching velocity field."
            )
        self.rtc_enabled = rtc_enabled
        self.rtc_guidance_mode = rtc_guidance_mode
        self.rtc_guidance_clip = rtc_guidance_clip
        # Env-space cumulative action, shape (B, action_env_dim). SFP only.
        self._sfp_env_action_states: torch.Tensor | None = None

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        rtc_context=None,
        dones: Any | None = None,
        episode_starts: Any | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values, kwargs
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use a training task wrapper (rl / dagger / dsrl) for train rollouts."
            )
        if self.sfp_cfg.use_sfp:
            if rtc_context is not None or self.rtc_enabled:
                raise ValueError(
                    "SFP eval cannot use RTC guidance; disable runner.rtc / "
                    "openpi.rtc_enabled."
                )
            return self._predict_sfp_eval(
                env_obs, dones=dones, episode_starts=episode_starts
            )

        observation = self.env_obs_to_observation(env_obs)
        if rtc_context is not None and self.rtc_enabled:
            if self.rtc_guidance_mode != "approx":
                raise NotImplementedError(
                    f"Unsupported RTC guidance mode: {self.rtc_guidance_mode!r}"
                )
            return self._predict_eval_with_rtc(
                observation, rtc_context=rtc_context, noise=noise, rng=rng
            )
        return self._predict_eval(observation, noise=noise, rng=rng)

    def _predict_sfp_eval(
        self,
        env_obs: dict[str, Any],
        *,
        dones: Any | None,
        episode_starts: Any | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.openpi.sampling.sfp_sampler import (
            sample_sfp_actions,
        )

        batch_size = self._sfp_batch_size(env_obs)
        self._prepare_sfp_action_states(batch_size, dones, episode_starts)
        observation = self.env_obs_to_observation(env_obs)
        model_actions = sample_sfp_actions(self, observation)
        actions = self.decode_actions(model_actions, observation.state)
        self._accumulate_sfp_action_states(actions)
        batch = actions.shape[0]
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(batch, -1).contiguous(),
                "model_action": model_actions.reshape(batch, -1).contiguous(),
            },
            "model_actions": model_actions,
        }

    def _sfp_batch_size(self, env_obs: dict[str, Any]) -> int:
        states = env_obs["states"]
        if hasattr(states, "shape"):
            return int(states.shape[0])
        return int(np.asarray(states).shape[0])

    def _prepare_sfp_action_states(
        self,
        batch_size: int,
        dones: Any | None,
        episode_starts: Any | None,
    ) -> None:
        env_dim = int(self.action_env_dim)
        device = self.device
        if (
            self._sfp_env_action_states is None
            or self._sfp_env_action_states.shape[0] != batch_size
            or self._sfp_env_action_states.shape[-1] != env_dim
        ):
            self._sfp_env_action_states = torch.zeros(
                batch_size, env_dim, device=device, dtype=torch.float32
            )
        else:
            self._sfp_env_action_states = self._sfp_env_action_states.to(device)
        reset = env_reset_mask(dones, episode_starts, batch_size, device)
        if reset is not None:
            self._sfp_env_action_states[reset] = 0

    def _accumulate_sfp_action_states(self, env_actions: torch.Tensor) -> None:
        if self._sfp_env_action_states is None:
            raise RuntimeError("SFP action_states buffer was not prepared.")
        executed = env_actions.to(
            device=self._sfp_env_action_states.device, dtype=torch.float32
        )
        self._sfp_env_action_states = self._sfp_env_action_states + executed.sum(dim=1)

    def _predict_eval_with_rtc(
        self,
        observation,
        *,
        rtc_context,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
            sample_actions_with_rtc_guidance,
        )

        model_actions = sample_actions_with_rtc_guidance(
            self,
            observation,
            rtc_context,
            num_steps=self.num_steps,
            noise=noise,
            rng=rng,
            guidance_clip=self.rtc_guidance_clip,
        )
        actions = self.decode_actions(model_actions, observation.state)
        B = actions.shape[0]
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
            "model_actions": model_actions,
        }

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head."""
        self._require_rlt()
        observation = self.env_obs_to_observation(env_obs)

        prepared_observation = preprocess_observation(observation, train=False)
        prefix_output, prefix_mask, kv_cache = self.build_prefix_cache(
            prepared_observation
        )
        rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, prepared_observation.tokenized_prompt
        )
        z_rl = self._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
            dtype=torch.float32
        )

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

    def _sample_actions_from_prefix_cache(
        self,
        observation,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                generator=rng,
            )

        x_t = noise
        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_out)
            x_t = x_t + dt * v_t
            t += dt
        return x_t
