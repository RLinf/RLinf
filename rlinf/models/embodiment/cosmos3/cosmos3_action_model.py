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

"""Cosmos3 action model: SFT forward, native HF sampling and actor replay."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.cosmos3.cosmos3_policy import (
    Cosmos3Policy,
    decode_prompts,
)


def shifted_timesteps(
    num_steps: int, shift: float, device: torch.device
) -> torch.Tensor:
    """Return the descending flow schedule used for stochastic actor sampling."""
    if num_steps < 1 or shift <= 0:
        raise ValueError(f"Expected num_steps>=1 and shift>0, got {num_steps}, {shift}")
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return shift * base / (1.0 + (shift - 1.0) * base)


def flow_sde_transition(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Pi0-style Gaussian transition in Cosmos3's flattened state."""
    delta = t - t_next
    if bool((delta <= 0).any()):
        raise ValueError("Cosmos3 flow timesteps must strictly decrease")
    # Match Pi0's finite first-step variance: use the next timestep in the
    # denominator when t=1 instead of dividing by zero.
    t_safe = torch.where(t >= 1.0, t_next, t).clamp(max=1.0 - 1e-4)
    sigma = noise_level * torch.sqrt(t / (1.0 - t_safe))
    std = torch.sqrt(delta) * sigma
    x0 = x_t - t * velocity
    x1 = x_t + (1.0 - t) * velocity
    mean = (1.0 - t_next) * x0 + (t_next - sigma.square() * delta / (2.0 * t)) * x1
    return mean, std


def action_transition_logprobs(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    horizon: int,
    model_action_dim: int,
    raw_action_dim: int,
    env_action_dim: int,
) -> torch.Tensor:
    """Return chunk-shaped logprobs whose sum is the selected flow transition logprob."""
    action_width = horizon * model_action_dim
    sample_action = sample[:, -action_width:].reshape(-1, horizon, model_action_dim)
    mean_action = mean[:, -action_width:].reshape(-1, horizon, model_action_dim)
    std_action = std[:, -action_width:].reshape(-1, 1, 1)
    sigma = std_action.clamp_min(1e-12)
    logprob = (
        -torch.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((sample_action - mean_action) / sigma).square()
    )[..., :raw_action_dim].sum(dim=-1)
    result = torch.zeros(
        (sample.shape[0], horizon, env_action_dim),
        device=sample.device,
        dtype=torch.float32,
    )
    result[..., 0] = logprob.float()
    return result


class Cosmos3ActionModel(Cosmos3Policy, nn.Module, BasePolicy):
    """Train and sample the native OmniMoT action model."""

    _no_split_modules: list[str] = [
        "MoTDecoderLayer",
        "PackedAttentionMoT",
        "Cosmos3VFMNetwork",
    ]
    _fsdp_wrap_embeddings: bool = False

    def __init__(
        self,
        model: nn.Module,
        *,
        model_cfg: Any = None,
        lr_multipliers: dict[str, float] | None = None,
        ema_enabled: bool = True,
    ):
        super().__init__()
        self.omni = model
        self.model_cfg = model_cfg
        self.lr_multipliers = dict(lr_multipliers or {})
        self._ema_enabled = bool(ema_enabled) and bool(
            getattr(getattr(model, "config", None), "ema", None)
            and model.config.ema.enabled
        )
        self._global_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.omni.parameters()).device

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accept EMA-enabled SFT weights in an EMA-disabled RL/eval replica."""
        if not self._ema_enabled:
            state_dict = state_dict.copy()
            for key in tuple(state_dict):
                if key.startswith("omni.net_ema."):
                    del state_dict[key]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_global_step(self, step: int) -> None:
        step = int(step)
        if self._ema_enabled and step == self._global_step + 1:
            self._update_ema(self._global_step)
        self._global_step = step

    @torch.no_grad()
    def _update_ema(self, iteration: int) -> None:
        model = self.omni
        if not getattr(model, "config", None) or not model.config.ema.enabled:
            return
        net = getattr(model, "net", None)
        net_ema = getattr(model, "net_ema", None)
        worker = getattr(model, "net_ema_worker", None)
        if net is None or net_ema is None or worker is None:
            return
        if getattr(model, "_uses_aux_loss_free_load_balancing", False) or getattr(
            model, "_uses_ema_router_bias", False
        ):
            for sync_fn in ("sync_expert_biases_to_ema", "sync_router_biases_to_ema"):
                fn = getattr(model, sync_fn, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - best-effort buffer sync
                        pass
        beta = model.ema_beta(iteration)
        worker.update_average(net, net_ema, beta=beta)

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported Cosmos3 forward type: {forward_type}")

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the native Cosmos flow-matching SFT loss."""
        torch.compiler.cudagraph_mark_step_begin()
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError(
                "Cosmos3 sft_forward requires `data` from the SFT dataloader."
            )
        from cosmos_framework.utils.misc import to as _cosmos_to

        data = _cosmos_to(data, "cuda")
        self._ensure_vision_device_patch()
        output_batch, loss = self.omni.training_step(data, self._global_step)
        result: dict[str, Any] = {"loss": loss}
        if isinstance(output_batch, dict):
            for key, value in output_batch.items():
                if key != "loss" and torch.is_tensor(value) and value.numel() == 1:
                    result[key] = value.detach()
        return result

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        return None

    def _ensure_vision_device_patch(self) -> None:
        """Cosmos builds image batches on CPU; move frames to the VAE device."""
        if getattr(self, "_vae_patch_applied", False):
            return
        import functools

        self._vae_patch_applied = True
        original = type(self.omni)._encode_vision_item

        def _patched(omni_self, state, *args, **kwargs):
            if hasattr(state, "device") and state.device.type == "cpu":
                state = state.to(self.device)
            return original(omni_self, state, *args, **kwargs)

        self.omni._encode_vision_item = functools.partial(_patched, self.omni)

    def _option(self, name: str, default):
        return (
            self.model_cfg.get(name, default) if self.model_cfg is not None else default
        )

    def _set_device(self, device: torch.device) -> None:
        self.omni.tensor_kwargs = {**self.omni.tensor_kwargs, "device": device}
        self.omni.tensor_kwargs_fp32 = {
            **self.omni.tensor_kwargs_fp32,
            "device": device,
        }
        sampler = getattr(self.omni, "sampler", None)
        if sampler is not None and getattr(sampler, "tensor_kwargs", None) is not None:
            sampler.tensor_kwargs = {**sampler.tensor_kwargs, "device": device}

    def prepare_for_inference(self, device: torch.device) -> None:
        """Move the rollout replica after the shared CPU model construction."""
        self.omni.to(device)
        self._set_device(device)

    def _build_batch(self, images: torch.Tensor, prompts: torch.Tensor) -> dict:
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.inference.inference import _merge_data_batches

        horizon = int(self._option("num_action_chunks", 16))
        batches = []
        for image, prompt in zip(images, decode_prompts(prompts), strict=True):
            batches.append(
                build_action_batch(
                    video=image.cpu().permute(2, 0, 1).unsqueeze(1).contiguous(),
                    action=torch.zeros(horizon, int(self.omni.config.max_action_dim)),
                    raw_action_dim=int(self._option("raw_action_dim", 10)),
                    prompt=prompt,
                    view_point=str(self._option("action_view_point", "ego_view")),
                    domain_name=str(self._option("domain_name", "libero")),
                    model_mode=ModelMode.POLICY,
                    action_chunk_size=horizon,
                    fps=int(self._option("fps", 20)),
                    resolution=str(self._option("resolution", "256")),
                    input_video_key=self.omni.input_video_key,
                    device=self.device,
                )
            )
        return _merge_data_batches(batches)

    @torch.no_grad()
    def generate_eval_actions(self, images: torch.Tensor, prompts: torch.Tensor):
        """Use Cosmos native deterministic sampler for HF evaluation."""
        self._set_device(self.device)
        self._ensure_vision_device_patch()
        batch = self._build_batch(images, prompts)
        n = images.shape[0]
        steps = int(self._option("num_inference_steps", 30))
        shift = float(self._option("shift", 10.0))
        output = self.omni.generate_samples_from_batch(
            batch,
            seed=[int(self._option("seed", 1140))] * n,
            num_steps=steps,
            shift=shift,
            guidance=float(self._option("guidance", 1.5)),
        )
        if len(output.get("action", [])) != n:
            raise RuntimeError("Cosmos3 native eval returned no action batch")
        return output["action"]

    @torch.no_grad()
    def sample_train_actions(self, images: torch.Tensor, prompts: torch.Tensor):
        """Sample an SDE trajectory and save one transition for GRPO replay."""
        self._set_device(self.device)
        self._ensure_vision_device_patch()
        batch = self._build_batch(images, prompts)
        n = images.shape[0]
        horizon = int(self._option("num_action_chunks", 16))
        raw_dim = int(self._option("raw_action_dim", 10))
        steps = int(self._option("num_inference_steps", 30))
        shift = float(self._option("shift", 10.0))
        seeds = torch.randint(0, 2**31 - 1, (n,)).tolist()
        plans, clean, cond, _, initial, reference, mask, has_actions = (
            self.omni._prepare_inference_data(batch, seed=seeds)
        )
        if not has_actions:
            raise RuntimeError("Cosmos3 policy plan has no action target")
        if getattr(self.omni.config, "sound_gen", False):
            raise ValueError("Cosmos3 action replay requires a vision-action model")
        x = torch.stack([value.float() for value in initial])
        ref = torch.stack([value.float() for value in reference])
        cond_mask = torch.stack([value.float() for value in mask])
        model_dim = int(clean.x0_tokens_action[0].shape[-1])
        if clean.x0_tokens_action[0].shape[0] != horizon or raw_dim > model_dim:
            raise ValueError("Cosmos3 action shape does not match rollout config")
        action_width = horizon * model_dim
        cond_mask[:, -action_width:].reshape(n, horizon, model_dim)[:, :, raw_dim:] = 1
        ref[:, -action_width:].reshape(n, horizon, model_dim)[:, :, raw_dim:] = 0
        x = cond_mask * ref + (1 - cond_mask) * x
        times = shifted_timesteps(steps, shift, self.device)
        noise_level = float(self._option("noise_level", 0.5))
        if noise_level <= 0:
            raise ValueError("Cosmos3 RL noise_level must be positive")
        selected = int(torch.randint(steps, (1,)).item())
        saved_x = saved_next = old_logprobs = None
        for step in range(steps):
            t = times[step].expand(n)
            velocity = torch.stack(
                self.omni._get_velocity(
                    noise_x=list(x.unbind(0)),
                    timestep=t[:, None],
                    text_tokens=cond,
                    sequence_plans=plans,
                    gen_data_clean=clean,
                    has_noisy_actions=True,
                )
            ).float()
            mean, std = flow_sde_transition(
                x, velocity, t[:, None], times[step + 1].expand(n, 1), noise_level
            )
            x_next = mean + std * torch.randn_like(mean)
            x_next = cond_mask * ref + (1 - cond_mask) * x_next
            if step == selected:
                saved_x, saved_next = x.cpu(), x_next.cpu()
                old_logprobs = action_transition_logprobs(
                    x_next,
                    mean,
                    std,
                    horizon=horizon,
                    model_action_dim=model_dim,
                    raw_action_dim=raw_dim,
                    env_action_dim=7,
                ).cpu()
            x = x_next
        model_actions = x[:, -action_width:].reshape(n, horizon, model_dim)[
            :, :, :raw_dim
        ]
        return list(model_actions.unbind(0)), {
            "prev_logprobs": old_logprobs,
            "prev_values": torch.zeros((n, 1), dtype=torch.float32),
            "forward_inputs": {
                "images": images,
                "prompts": prompts,
                "latent": saved_x,
                "next_latent": saved_next,
                "step": torch.full((n,), selected, dtype=torch.long),
            },
        }

    def default_forward(
        self,
        forward_inputs: dict,
        compute_values: bool = False,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Recompute selected rollout transition with current actor parameters."""
        del kwargs
        if compute_values:
            raise NotImplementedError(
                "Cosmos3 actor supports GRPO, not GAE value learning"
            )
        x = forward_inputs["latent"].float()
        x_next = forward_inputs["next_latent"].float()
        step = forward_inputs["step"].long()
        n = x.shape[0]
        self._set_device(x.device)
        self._ensure_vision_device_patch()
        batch = self._build_batch(forward_inputs["images"], forward_inputs["prompts"])
        plans, clean, cond, _, _, _, _, has_actions = self.omni._prepare_inference_data(
            batch, seed=[0] * n
        )
        if not has_actions:
            raise RuntimeError("Cosmos3 replay has no action target")
        if getattr(self.omni.config, "sound_gen", False):
            raise ValueError("Cosmos3 action replay requires a vision-action model")
        times = shifted_timesteps(
            int(self._option("num_inference_steps", 30)),
            float(self._option("shift", 10.0)),
            x.device,
        )
        t = times[step]
        velocity = torch.stack(
            self.omni._get_velocity(
                noise_x=list(x.unbind(0)),
                timestep=t[:, None],
                text_tokens=cond,
                sequence_plans=plans,
                gen_data_clean=clean,
                has_noisy_actions=True,
            )
        ).float()
        mean, std = flow_sde_transition(
            x,
            velocity,
            t[:, None],
            times[step + 1][:, None],
            float(self._option("noise_level", 0.5)),
        )
        logprobs = action_transition_logprobs(
            x_next,
            mean,
            std,
            horizon=int(self._option("num_action_chunks", 16)),
            model_action_dim=int(clean.x0_tokens_action[0].shape[-1]),
            raw_action_dim=int(self._option("raw_action_dim", 10)),
            env_action_dim=7,
        )
        output = {"logprobs": logprobs}
        if compute_entropy:
            output["entropy"] = torch.zeros_like(logprobs)
        return output
