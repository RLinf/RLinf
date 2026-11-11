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

"""Hybrid Flow-SDE WAM policies using the embodied OpenPI PPO contract."""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.data.schema.embodied_types import split_batch_value
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy
from rlinf.models.embodiment.cosmos3.env_io import Cosmos3EnvIO
from rlinf.models.embodiment.cosmos3.sampling import bridge, rl_sampler
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

logger = get_logger()


class Cosmos3RL(Cosmos3EnvIO, Cosmos3Policy):
    """Generate joint video/actions and score one stored stochastic action step.

    Future video follows the ODE. One action update uses Flow-SDE, with the
    current video latent included in its conditioning state. PPO re-evaluates
    that exact transition and uses the existing chunk-level advantage pipeline.
    """

    def __init__(self, model: torch.nn.Module, model_cfg: DictConfig):
        super().__init__(
            model,
            lr_multipliers=dict(model_cfg.get("lr_multipliers", {})),
            ema_enabled=False,
        )
        self.model_cfg = model_cfg
        self.rl_cfg = model_cfg.cosmos3
        self.num_steps = int(model_cfg.get("num_inference_steps", 4))
        self.action_chunk = int(model_cfg.num_action_chunks)
        self.valid_action_dim = int(self.rl_cfg.get("valid_action_dim", 10))
        self._action_normalizer = None
        self.value_head = ValueHead(
            input_dim=int(model.net.hidden_size),
            hidden_sizes=list(self.rl_cfg.get("value_hidden_dims", [512, 256, 128])),
            output_dim=1,
            activation="relu",
            bias_last=True,
        ).float()

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"Cosmos3RL does not support forward_type={forward_type}."
        )

    def _value(self, features: torch.Tensor) -> torch.Tensor:
        if self.rl_cfg.get("detach_critic_input", True):
            features = features.detach()
        dtype = next(self.value_head.parameters()).dtype
        return self.value_head(features.to(dtype))[:, 0].float()

    def _score_mask(self, context: bridge.WAMContext) -> torch.Tensor:
        mask = context.action_mask[:, context.action_offset :].reshape(
            context.state.shape[0], *context.action_shape
        )
        mask = mask.clone()
        mask[:, self.action_chunk :] = False
        mask[:, :, self.valid_action_dim :] = False
        return mask

    def _context(
        self, replay: dict[str, torch.Tensor], seeds: list[int]
    ) -> bridge.WAMContext:
        context = bridge.prepare_context(self.omni, self.native_batch(replay), seeds)
        limit = int(self.model_cfg.get("max_prompt_length", 1024))
        if any(len(tokens) > limit for tokens in context.text_tokens):
            raise ValueError(f"WAM prompt exceeds max_prompt_length={limit} tokens.")
        return context

    def _predict_sde_step(
        self,
        replay: dict[str, torch.Tensor],
        state: torch.Tensor,
        timestep: torch.Tensor,
        micro_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        sizes = [
            min(micro_batch_size, batch_size - start)
            for start in range(0, batch_size, micro_batch_size)
        ]
        # Rebuild each pack as actor replay does, using the existing noisy state.
        outputs = [
            bridge.predict_velocity(
                self.omni, self._context(inputs, [0] * size), state_slice, timestep
            )
            for size, inputs, state_slice in zip(
                sizes, split_batch_value(replay, sizes), state.split(sizes)
            )
        ]
        velocities, features = zip(*outputs)
        return torch.cat(velocities), torch.cat(features)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = True,
        *,
        logprob_micro_batch_size: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample all WAM steps, recording only the selected PPO transition."""
        del kwargs
        if mode not in ("train", "eval"):
            raise ValueError(f"Expected mode train or eval, got {mode!r}.")
        if logprob_micro_batch_size is not None and logprob_micro_batch_size < 1:
            raise ValueError("Expected a positive logprob_micro_batch_size.")
        # Density recomputation must use the same deterministic denoiser mode.
        self.omni.eval()
        replay = self.replay_observation(env_obs)
        batch_size = replay["observation/image"].shape[0]
        seeds = torch.randint(0, 2**31, (batch_size,)).tolist()
        context = self._context(replay, seeds)
        if self.action_chunk > context.action_shape[0]:
            raise ValueError(
                "Executed action chunk exceeds the WAM generation horizon."
            )
        times = rl_sampler.get_timesteps(self.num_steps, self.device)
        selected = random.randrange(self.num_steps) if mode == "train" else -1
        state = bridge.enforce_condition(context, context.state)
        score_mask = self._score_mask(context)
        values = []
        old_logprob = None
        for index in range(self.num_steps):
            if (
                index == selected
                and logprob_micro_batch_size is not None
                and batch_size > logprob_micro_batch_size
            ):
                velocity, features = self._predict_sde_step(
                    replay, state, times[index], logprob_micro_batch_size
                )
            else:
                velocity, features = bridge.predict_velocity(
                    self.omni, context, state, times[index]
                )
            if compute_values:
                values.append(self._value(features))
            next_state = state + (times[index + 1] - times[index]) * velocity
            if index == selected:
                actions_pre = bridge.action_latents(context, state)
                action_velocity = bridge.action_latents(context, velocity)
                mean, std = rl_sampler.sample_mean_var(
                    actions_pre,
                    action_velocity,
                    torch.full(
                        (batch_size,), index, device=self.device, dtype=torch.long
                    ),
                    noise_method="flow_sde",
                    noise_level=float(self.rl_cfg.noise_level),
                    timesteps=times,
                )
                sampled_action = mean + std * torch.randn_like(mean)
                next_state[:, context.action_offset :] = sampled_action.reshape(
                    batch_size, -1
                )
                next_state = bridge.enforce_condition(context, next_state)
                actions_next = bridge.action_latents(context, next_state)
                old_logprob = rl_sampler.gaussian_logprob(
                    actions_next, mean, std, mask=score_mask
                )
                replay["wam_state"] = state.clone()
                replay["wam_action_next"] = actions_next.clone()
                replay["denoise_inds"] = torch.full(
                    (batch_size,), index, device=self.device, dtype=torch.long
                )
            state = bridge.enforce_condition(context, next_state)
        model_actions = bridge.action_latents(context, state)
        actions = self.decode_actions(model_actions)
        if not torch.isfinite(actions).all():
            raise ValueError("WAM generated nonfinite environment actions.")
        replay["action"] = actions.reshape(batch_size, -1).clone()
        replay["model_action"] = model_actions.reshape(batch_size, -1).clone()
        return actions, {
            "prev_logprobs": None
            if old_logprob is None
            else old_logprob[
                :, : self.action_chunk, : self.valid_action_dim
            ].contiguous(),
            "prev_values": torch.stack(values, dim=1).mean(dim=1, keepdim=True)
            if values
            else torch.zeros(batch_size, 1, device=self.device),
            "forward_inputs": replay,
            "model_actions": model_actions,
        }

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_values: bool = True,
        *,
        return_sde_params: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Re-evaluate the cached state/output without resampling the WAM path."""
        del kwargs
        self.omni.eval()
        state = forward_inputs["wam_state"].detach().float()
        action_next = forward_inputs["wam_action_next"].detach().float()
        indices = forward_inputs["denoise_inds"].long()
        context = self._context(forward_inputs, [0] * state.shape[0])
        times = rl_sampler.get_timesteps(self.num_steps, state.device)
        velocity, features = bridge.predict_velocity(
            self.omni, context, state, times[indices]
        )
        mean, std = rl_sampler.sample_mean_var(
            bridge.action_latents(context, state),
            bridge.action_latents(context, velocity),
            indices,
            noise_method="flow_sde",
            noise_level=float(self.rl_cfg.noise_level),
            timesteps=times,
        )
        score_mask = self._score_mask(context)
        logprobs = rl_sampler.gaussian_logprob(action_next, mean, std, mask=score_mask)
        scored_actions = (
            slice(None),
            slice(self.action_chunk),
            slice(self.valid_action_dim),
        )
        output = {
            "logprobs": logprobs[scored_actions].contiguous(),
            "values": self._value(features)
            if compute_values
            else torch.zeros(state.shape[0], device=state.device),
            "entropy": torch.zeros(state.shape[0], 1, device=state.device),
        }
        if return_sde_params:
            output.update(
                sde_mean=mean[scored_actions],
                sde_std=std[scored_actions],
                sde_mask=score_mask[scored_actions],
            )
        return output


def load_ppo_checkpoint(model: Cosmos3Policy, checkpoint_path: str) -> None:
    """Warm-start WAM from SFT or PPO weights, allowing only a new value head."""
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith("omni.net_ema.")
    }
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("value_head.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"WAM checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}."
        )
    logger.info(
        "Loaded WAM checkpoint %s; initialized value parameters=%d",
        checkpoint_path,
        len(result.missing_keys),
    )
