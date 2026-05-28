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

"""RLinf policy wrapper for the XR0 VLA model.

XR0 uses Qwen3-VL as the vision-language backbone and a DiT with rectified
flow for continuous action prediction.  This module adapts the raw XR0 model
to RLinf's ``BasePolicy`` interface for rollout and training.
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger

from .utils import ACTION_DIM, denormalize_action, resize_image


class XR0ForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf policy wrapper for XR0 VLA checkpoints.

    Wraps the XR0 model (Qwen3-VL + DiT) and exposes the
    ``predict_action_batch`` / ``default_forward`` interface required by
    RLinf's rollout and actor workers.

    Args:
        xr0_model: The XR0 model instance (stub or real).
        action_dim: Action dimensionality (default 32).
        num_action_chunks: Number of action timesteps per chunk (default 30).
        num_steps: Number of rectified flow denoising steps.
        action_mean: Per-timestep action mean for denormalization.
        action_std: Per-timestep action std for denormalization.
        noise_level: Noise level for flow-SDE (default 0.5).

    TODO: Add π_RL-style learnable noise network (ExploreNoiseNet) for
    Flow-Noise method. Currently uses fixed noise_level. See lingbotvla
    for reference implementation with flow_noise / flow_sde / flow_cps.
    TODO: Add Flow-SDE method (ODE-to-SDE conversion) for better RL
    exploration. See lingbotvla.sample_mean_var_val for reference.
    TODO: Add value head (ValueHead MLP) for PPO critic. Currently
    values are stub zeros. Use rlinf.models.embodiment.modules.value_head.
    """

    def __init__(
        self,
        xr0_model: nn.Module,
        action_dim: int = ACTION_DIM,
        num_action_chunks: int = 30,
        num_steps: int = 5,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        noise_level: float = 0.5,
    ):
        super().__init__()
        self.logger = get_logger()

        self.xr0_model = xr0_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_steps = int(num_steps)
        self.noise_level = float(noise_level)

        # Action normalization stats
        self.register_buffer(
            "action_mean",
            torch.from_numpy(action_mean).float() if action_mean is not None else None,
        )
        self.register_buffer(
            "action_std",
            torch.from_numpy(action_std).float() if action_std is not None else None,
        )

        # Qwen3-VL processor (lazy-loaded on first use)
        self._processor = None

        # FSDP wrap name for every submodule
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    # ------------------------------------------------------------------
    # FSDP hints
    # ------------------------------------------------------------------

    @property
    def processor(self):
        """Lazy-load Qwen3-VL processor on first access."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-4B-Instruct"
            )
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "DecoderLayer",
            "Qwen3VLDecoderLayer",
            "Qwen3VLVisionBlock",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "vlm",
            "dit",
            "state_projector",
            "action_projector",
            "action_output_layer",
        ]

    # ------------------------------------------------------------------
    # Log-probability helpers (for RL training)
    # ------------------------------------------------------------------

    @staticmethod
    def get_logprob_norm(
        sample: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probability of *sample* under N(mu, sigma).

        When sigma == 0 (deterministic step), returns 0 for that element.

        Returns:
            Tensor of same shape as *sample*.
        """
        sample = sample.float()
        mu = mu.float()
        sigma = sigma.float()

        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

        log_prob = -0.5 * ((sample - mu) / sigma_safe) ** 2
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    @staticmethod
    def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
        """Entropy of N(0, sigma).  Returns 0 where sigma == 0."""
        sigma = sigma.float()
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * sigma_safe**2)
        entropy = torch.where(mask, torch.zeros_like(entropy), entropy)
        return entropy

    # ------------------------------------------------------------------
    # Step-by-step denoising (for RL chain recording)
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        vlm_batch: dict[str, torch.Tensor],
        state_tensor: torch.Tensor,
        device: torch.device,
        mode: Literal["train", "eval"] = "train",
    ) -> dict[str, Any]:
        """Run rectified flow denoising step-by-step and record the chain.

        During ``mode="train"``, one random denoising step is made stochastic
        (nonzero std) so its log-prob can be used for the policy gradient.
        All other steps are deterministic (std=0, logprob=0).

        Args:
            vlm_batch: VLM processor output (input_ids, pixel_values, ...).
            state_tensor: ``(B, 1, STATE_DIM)`` proprioceptive state.
            device: Target device.
            mode: ``"train"`` or ``"eval"``.

        Returns:
            Dict with ``actions``, ``chains``, ``prev_logprobs``,
            ``denoise_inds``, ``prev_values``.
        """
        batch_size = state_tensor.shape[0]

        # Determine if stub or real model
        is_stub = hasattr(self.xr0_model, "generate")

        if is_stub:
            return self._sample_actions_stub(
                batch_size, device, mode
            )

        # --- Real model: step-by-step denoising ---
        # VLM forward to get KV-cache
        vlm_inputs = {
            k: v.to(device) for k, v in vlm_batch.items() if isinstance(v, torch.Tensor)
        }
        vlm_outputs = self.xr0_model.vlm(**vlm_inputs, use_cache=True)
        past_key_values = list(vlm_outputs.past_key_values)

        # Position ids for DiT (continue from VLM's last position)
        action_len = self.num_action_chunks
        state_len = state_tensor.shape[1]  # typically 1
        q_len = action_len + state_len + 1  # +1 for sink token
        position_ids = (
            torch.arange(0, q_len, device=device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )

        # Attention mask
        cache_mask = vlm_inputs.get(
            "attention_mask",
            torch.ones(batch_size, 1, device=device, dtype=torch.long),
        )
        cache_mask = cache_mask[:, None, :].expand(-1, q_len, -1)

        # Build causal mask for DiT tokens
        s_len = state_len + 1
        a_len = action_len
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        local_window = getattr(self.xr0_model, "local_window", 4)
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)],
            dim=0,
        )
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

        # State embedding
        state_embed = self.xr0_model.state_projector(
            state_tensor.to(device=device, dtype=torch.bfloat16)
        )

        # Position embeddings (RoPE)
        dummy_action = torch.zeros(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        position_embeds = self.xr0_model.rotary_emb(dummy_action, position_ids)

        # Action mask (all ones)
        action_mask = torch.ones(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        # Timestep schedule: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] for 5 steps
        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # Pick one random step for stochastic gradient
        if mode == "train":
            denoise_ind = random.randint(0, self.num_steps - 1)
        else:
            denoise_ind = -1  # all deterministic

        # Denoising loop
        x_t = torch.randn(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        chains = [x_t.detach().clone()]
        log_probs = []
        values = []

        for idx in range(self.num_steps):
            t_val = timesteps[idx].item()
            delta = (timesteps[idx] - timesteps[idx + 1]).item()

            # DiT forward: predict velocity
            t_tensor = torch.full(
                (batch_size, 1, 1), t_val, device=device, dtype=torch.bfloat16
            )
            v_t = self.xr0_model.dit_forward(
                x_t, t_tensor, action_mask, state_embed,
                position_embeds, past_key_values, attn_mask,
            )

            # x0 prediction from velocity
            x0_pred = x_t - v_t * t_tensor

            # Deterministic mean for this step
            w0 = 1 - t_val + delta
            w1 = t_val - delta
            x_t_mean = x0_pred * w0 + (x_t + v_t * (1 - t_val)) * w1

            # Stochastic step (only at denoise_ind)
            # TODO: Replace fixed sigma with learnable ExploreNoiseNet
            # (π_RL Flow-Noise method). See lingbotvla for reference.
            if mode == "train" and idx == denoise_ind:
                sigma = self.noise_level * math.sqrt(delta)
                x_t_std = torch.full_like(x_t, sigma)
                noise = torch.randn_like(x_t)
                x_t = x_t_mean + noise * x_t_std
                log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            else:
                x_t_std = torch.zeros_like(x_t)
                x_t = x_t_mean
                log_prob = torch.zeros_like(x_t)

            chains.append(x_t.detach().clone())
            log_probs.append(log_prob)
            # TODO: Compute real value at each denoising step using
            # ValueHead on DiT hidden states (see lingbotvla pattern).
            values.append(torch.zeros(batch_size, device=device))

        # Aggregate: (B, num_steps+1, action_len, action_dim)
        chains_tensor = torch.stack(chains, dim=1)
        log_probs_tensor = torch.stack(log_probs, dim=1)

        # Pick logprob at the chosen denoise step, average over action dims
        if denoise_ind >= 0:
            prev_logprobs = log_probs_tensor[:, denoise_ind].mean(dim=[1, 2])
        else:
            prev_logprobs = torch.zeros(batch_size, device=device)

        # TODO: Aggregate per-step values into prev_values (see lingbotvla
        # pattern: values = torch.stack(values, dim=1).mean(...)).
        prev_values = torch.zeros(batch_size, device=device)

        return {
            "actions": x_t[:, :action_len, : self.action_dim],
            "chains": chains_tensor,
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "denoise_inds": torch.full((batch_size,), denoise_ind, dtype=torch.long),
        }

    def _sample_actions_stub(
        self,
        batch_size: int,
        device: torch.device,
        mode: str,
    ) -> dict[str, Any]:
        """Stub version: generate random actions with dummy chain."""
        action_len = self.num_action_chunks

        # Random final action
        actions = torch.randn(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        # Build a dummy chain (num_steps+1 states)
        chains = []
        for _ in range(self.num_steps + 1):
            chains.append(
                torch.randn(
                    (batch_size, action_len, self.action_dim),
                    device=device, dtype=torch.bfloat16,
                )
            )
        chains_tensor = torch.stack(chains, dim=1)

        denoise_ind = random.randint(0, self.num_steps - 1) if mode == "train" else -1

        return {
            "actions": actions,
            "chains": chains_tensor,
            # TODO: For stub, prev_logprobs/prev_values are zeros.
            # Real model will compute these from the chain.
            "prev_logprobs": torch.zeros(batch_size, device=device),
            "prev_values": torch.zeros(batch_size, device=device),
            "denoise_inds": torch.full(
                (batch_size,), denoise_ind, dtype=torch.long
            ),
        }

    # ------------------------------------------------------------------
    # Rollout-time: predict_action_batch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict a batch of actions from environment observations.

        Returns:
            Tuple of ``(actions, result)`` where *actions* is ``[B, C, D]``
            and *result* has ``prev_logprobs``, ``prev_values``,
            ``forward_inputs``.
        """
        device = next(self.parameters()).device

        images = env_obs["main_images"]
        states = env_obs["states"]
        task_descriptions = env_obs.get("task_descriptions", [""] * len(images))
        batch_size = len(images)

        # State tensor: (B, 1, STATE_DIM)
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)

        # Build VLM batch
        vlm_batch = self._build_vlm_batch(
            images, task_descriptions, state_tensor, device
        )

        # Run step-by-step denoising with chain recording
        outputs = self.sample_actions(vlm_batch, state_tensor, device, mode=mode)

        # Denormalize actions
        actions_np = outputs["actions"].float().cpu().numpy()
        if self.action_mean is not None and self.action_std is not None:
            mean = self.action_mean.cpu().numpy()
            std = self.action_std.cpu().numpy()
            actions_np = denormalize_action(actions_np, mean, std)

        actions_np = actions_np[:, : self.num_action_chunks, :]

        # Build forward_inputs for training replay
        forward_inputs: dict[str, Any] = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
        }
        for k, v in vlm_batch.items():
            if isinstance(v, torch.Tensor):
                forward_inputs[k] = v.detach().cpu()
        forward_inputs["state"] = state_tensor.detach().cpu()
        forward_inputs["action"] = torch.from_numpy(
            actions_np.reshape(batch_size, -1).astype(np.float32)
        )

        result = {
            "prev_logprobs": outputs["prev_logprobs"].cpu(),
            "prev_values": outputs["prev_values"].cpu(),
            "forward_inputs": forward_inputs,
        }
        return actions_np, result

    # ------------------------------------------------------------------
    # Training-time: default_forward
    # ------------------------------------------------------------------

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Replay the denoising chain and recompute logprobs/entropy.

        During RL training, the actor worker calls this with the stored
        ``forward_inputs`` from rollout.  It replays the single stochastic
        denoising step under the *current* policy weights to get the
        ``logprobs`` needed for the PPO/GRPO policy ratio.
        """
        if forward_inputs is None:
            forward_inputs = {}

        device = next(self.parameters()).device

        chains = forward_inputs.get("chains")  # (B, num_steps+1, C, D)
        denoise_inds = forward_inputs.get("denoise_inds")  # (B,)

        if chains is None or denoise_inds is None:
            # TODO: This fallback should not happen in normal usage.
            # Indicates forward_inputs was not properly built.
            batch_size = 1
            for v in forward_inputs.values():
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    batch_size = v.shape[0]
                    break
            return {
                "logprobs": torch.zeros(batch_size, device=device),
                "values": torch.zeros(batch_size, device=device),
                "entropy": torch.zeros(batch_size, device=device),
            }

        batch_size = chains.shape[0]
        denoise_ind = int(denoise_inds[0].item())

        if denoise_ind < 0:
            # Eval mode: no stochastic step, return zeros
            return {
                "logprobs": torch.zeros(batch_size, device=device),
                "values": torch.zeros(batch_size, device=device),
                "entropy": torch.zeros(batch_size, device=device),
            }

        is_stub = hasattr(self.xr0_model, "generate")

        if is_stub:
            # TODO: For stub model, return dummy logprobs/entropy.
            # Real model replays the chain below.
            return {
                "logprobs": torch.zeros(batch_size, device=device),
                "values": torch.zeros(batch_size, device=device),
                "entropy": torch.zeros(batch_size, device=device),
            }

        # --- Real model: replay the chain ---
        chains = chains.to(device)

        # Rebuild VLM batch from stored inputs
        vlm_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vlm_batch = {}
        for k in vlm_keys:
            if k in forward_inputs:
                vlm_batch[k] = forward_inputs[k].to(device)

        state_tensor = forward_inputs.get("state")
        if state_tensor is not None:
            state_tensor = state_tensor.to(device)

        # VLM forward to get current KV-cache
        vlm_outputs = self.xr0_model.vlm(**vlm_batch, use_cache=True)
        past_key_values = list(vlm_outputs.past_key_values)

        # Build position ids and attention mask (same as sample_actions)
        action_len = self.num_action_chunks
        state_len = state_tensor.shape[1] if state_tensor is not None else 1
        q_len = action_len + state_len + 1
        position_ids = (
            torch.arange(0, q_len, device=device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )

        cache_mask = vlm_batch.get(
            "attention_mask",
            torch.ones(batch_size, 1, device=device, dtype=torch.long),
        )
        cache_mask = cache_mask[:, None, :].expand(-1, q_len, -1)

        s_len = state_len + 1
        a_len = action_len
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        local_window = getattr(self.xr0_model, "local_window", 4)
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)],
            dim=0,
        )
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

        state_embed = self.xr0_model.state_projector(
            state_tensor.to(dtype=torch.bfloat16)
        )

        dummy_action = torch.zeros(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        position_embeds = self.xr0_model.rotary_emb(dummy_action, position_ids)

        action_mask = torch.ones(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # Replay: get x_t and x_{t+1} from the recorded chain
        x_t = chains[:, denoise_ind]  # (B, C, D)
        x_next = chains[:, denoise_ind + 1]  # recorded next state

        t_val = timesteps[denoise_ind].item()
        delta = (timesteps[denoise_ind] - timesteps[denoise_ind + 1]).item()

        # Current policy's velocity prediction
        t_tensor = torch.full(
            (batch_size, 1, 1), t_val, device=device, dtype=torch.bfloat16
        )
        v_t = self.xr0_model.dit_forward(
            x_t, t_tensor, action_mask, state_embed,
            position_embeds, past_key_values, attn_mask,
        )

        # Current policy's mean and std
        x0_pred = x_t - v_t * t_tensor
        w0 = 1 - t_val + delta
        w1 = t_val - delta
        x_t_mean = x0_pred * w0 + (x_t + v_t * (1 - t_val)) * w1
        sigma = self.noise_level * math.sqrt(delta)
        x_t_std = torch.full_like(x_t, sigma)

        # Log-prob of recorded next state under current policy
        logprobs = self.get_logprob_norm(x_next, x_t_mean, x_t_std)
        logprobs = logprobs.mean(dim=[1, 2])  # (B,)

        # Entropy
        entropy = self.gaussian_entropy(x_t_std)
        entropy = entropy.mean(dim=[1, 2])  # (B,)

        # Values (stub for now)
        # TODO: Implement value head using ValueHead MLP from
        # rlinf.models.embodiment.modules.value_head. Feed the DiT hidden
        # states (or VLM prefix output) through value_head to get real
        # value estimates for PPO critic.
        values = torch.zeros(batch_size, device=device)

        return {
            "logprobs": logprobs.float(),
            "values": values,
            "entropy": entropy.float(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vlm_batch(
        self,
        images: np.ndarray,
        task_descriptions: list[str],
        state_tensor: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert env observations to Qwen3-VL processor format."""
        pil_images = []
        for img_np in images:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            pil_img = resize_image(pil_img, factor=32, max_pixels=90000)
            pil_images.append(pil_img)

        messages = []
        for i, pil_img in enumerate(pil_images):
            instruction = task_descriptions[i] if i < len(task_descriptions) else ""
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "\n# Ego View\n"},
                            {"type": "image", "image": pil_img},
                            {
                                "type": "text",
                                "text": (
                                    "\nGenerate robot actions"
                                    " for the task:\n"
                                    + instruction
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<bot></bot>"}],
                    },
                ]
            )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )

        batch = {
            k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        batch["state"] = state_tensor.to(device=device, dtype=torch.bfloat16)
        return batch

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Enable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_enable"):
                self.xr0_model.vlm.gradient_checkpointing_enable(**kwargs)
            elif hasattr(self.xr0_model.vlm, "model") and hasattr(
                self.xr0_model.vlm.model, "visual"
            ):
                self.xr0_model.vlm.model.visual.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_disable"):
                self.xr0_model.vlm.gradient_checkpointing_disable()
