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

from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

from .model.xr0_model import XR0
from .utils import ACTION_DIM, STATE_DIM


class XR0ForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf policy wrapper for XR0 VLA checkpoints.

    Wraps the vendored ``XR0`` model (Qwen3-VL + DiT) and exposes the
    ``predict_action_batch`` / ``default_forward`` interface required by
    RLinf's rollout and actor workers.

    Args:
        xr0_model: The vendored ``XR0`` model instance.
        action_dim: Action dimensionality (default 32).
        num_action_chunks: Number of action timesteps per chunk (default 30).
        num_steps: Number of rectified flow denoising steps.
    """

    def __init__(
        self,
        xr0_model: XR0,
        action_dim: int = ACTION_DIM,
        num_action_chunks: int = 30,
        num_steps: int = 5,
    ):
        super().__init__()
        self.logger = get_logger()

        self.xr0_model = xr0_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_steps = int(num_steps)

    # ------------------------------------------------------------------
    # FSDP hints
    # ------------------------------------------------------------------

    @property
    def _no_split_modules(self) -> list[str]:
        """Module class names that FSDP should not shard across ranks."""
        return [
            "DecoderLayer",
            "Qwen3VLDecoderLayer",
            "Qwen3VLVisionBlock",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        """Named submodules that FSDP should keep on a single rank."""
        return [
            "vlm",
            "dit",
            "state_projector",
            "action_projector",
            "action_output_layer",
        ]

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

        Args:
            env_obs: Observation dict with keys ``main_images``, ``states``,
                ``task_descriptions``.
            mode: ``"train"`` or ``"eval"``.
            compute_values: Whether to compute value estimates (unused in v1).

        Returns:
            Tuple of ``(actions, result)`` where *actions* is a numpy array of
            shape ``[B, num_action_chunks, action_dim]`` and *result* is a dict
            containing ``prev_logprobs``, ``prev_values``, and
            ``forward_inputs`` for training replay.
        """
        device = next(self.parameters()).device

        # --- Build VLM inputs from env_obs ---
        images = env_obs["main_images"]  # [B, H, W, C] uint8 numpy
        states = env_obs["states"]  # [B, STATE_DIM] numpy
        task_descriptions = env_obs.get("task_descriptions", [""] * len(images))

        batch_size = len(images)

        # Convert to tensors
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Build VLM batch using processor
        vlm_batch = self._build_vlm_batch(images, task_descriptions, device)

        # Add state to batch for XR0 model
        vlm_batch["state"] = state_tensor.to(dtype=torch.bfloat16)

        # --- Run XR0 inference ---
        actions_pred = self.xr0_model.generate(vlm_batch)  # [B, action_len, action_dim]

        # Truncate/pad to num_action_chunks
        actions_np = actions_pred[:, : self.num_action_chunks, :].float().cpu().numpy()

        # --- Build forward_inputs for training replay ---
        forward_inputs = {
            "input_ids": vlm_batch.get("input_ids"),
            "attention_mask": vlm_batch.get("attention_mask"),
            "pixel_values": vlm_batch.get("pixel_values"),
            "image_grid_thw": vlm_batch.get("image_grid_thw"),
            "state": state_tensor.detach().cpu(),
            "action": torch.from_numpy(actions_np.reshape(batch_size, -1)),
        }

        # Remove None values
        forward_inputs = {k: v for k, v in forward_inputs.items() if v is not None}

        # --- Stub logprobs/values for v1 ---
        prev_logprobs = torch.zeros(batch_size, device="cpu")
        prev_values = torch.zeros(batch_size, device="cpu")

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions_np, result

    # ------------------------------------------------------------------
    # Training-time: default_forward
    # ------------------------------------------------------------------

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Run training-time forward for PPO terms (logprob/entropy/value).

        v1 stub: returns zeros for all RL terms.  The full implementation
        will replay the denoising chain from ``forward_inputs`` and compute
        logprobs via the flow-matching probability model.

        Args:
            forward_inputs: Cached rollout tensors from ``predict_action_batch``.
            compute_logprobs: Whether to compute action log-probabilities.
            compute_entropy: Whether to compute policy entropy.
            compute_values: Whether to compute value baseline.

        Returns:
            Dict with ``logprobs``, ``entropy``, and ``values``.
        """
        if forward_inputs is None:
            forward_inputs = {}

        # Determine batch size from available tensors
        batch_size = 1
        for v in forward_inputs.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                break

        device = next(self.parameters()).device

        return {
            "logprobs": torch.zeros(batch_size, device=device),
            "values": torch.zeros(batch_size, device=device),
            "entropy": torch.zeros(batch_size, device=device),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vlm_batch(
        self,
        images: np.ndarray,
        task_descriptions: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Build a VLM input batch from raw images and task descriptions.

        For v1, this creates a minimal batch with placeholder tokenization.
        The full implementation will use the Qwen3-VL processor.
        """
        batch_size = len(images)

        # v1 stub: create minimal tensors that the XR0 model expects
        # In the full implementation, these will come from the Qwen3-VL processor
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Enable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_enable"):
                self.xr0_model.vlm.gradient_checkpointing_enable(**kwargs)
            elif hasattr(self.xr0_model.vlm.model, "visual"):
                self.xr0_model.vlm.model.visual.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_disable"):
                self.xr0_model.vlm.gradient_checkpointing_disable()
