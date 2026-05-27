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
from PIL import Image
from transformers import AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

from .utils import ACTION_DIM, denormalize_action, resize_image


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
        action_mean: Per-timestep action mean for denormalization, shape
            ``(num_action_chunks, action_dim)`` or ``None``.
        action_std: Per-timestep action std for denormalization, shape
            ``(num_action_chunks, action_dim)`` or ``None``.
    """

    def __init__(
        self,
        xr0_model: nn.Module,
        action_dim: int = ACTION_DIM,
        num_action_chunks: int = 30,
        num_steps: int = 5,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.logger = get_logger()

        self.xr0_model = xr0_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_steps = int(num_steps)

        # Action normalization stats
        self.register_buffer(
            "action_mean",
            torch.from_numpy(action_mean).float() if action_mean is not None else None,
        )
        self.register_buffer(
            "action_std",
            torch.from_numpy(action_std).float() if action_std is not None else None,
        )

        # Qwen3-VL processor (lazy-loaded on first use to avoid HF download in tests)
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
            self._processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
            self._processor.tokenizer.padding_side = "right"
        return self._processor

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
            compute_values: Whether to compute value estimates (stub for now).

        Returns:
            Tuple of ``(actions, result)`` where *actions* is a numpy array of
            shape ``[B, num_action_chunks, action_dim]`` and *result* is a dict
            containing ``prev_logprobs``, ``prev_values``, and
            ``forward_inputs`` for training replay.
        """
        device = next(self.parameters()).device

        images = env_obs["main_images"]  # [B, H, W, C] uint8 numpy
        states = env_obs["states"]  # [B, STATE_DIM] numpy
        task_descriptions = env_obs.get("task_descriptions", [""] * len(images))
        batch_size = len(images)

        # State tensor
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Build VLM batch with real processor
        vlm_batch = self._build_vlm_batch(
            images, task_descriptions, state_tensor, device
        )

        # Run XR0 generate
        actions_pred = self.xr0_model.generate(vlm_batch)  # [B, action_len, action_dim]

        # Denormalize
        actions_np = actions_pred.float().cpu().numpy()
        if self.action_mean is not None and self.action_std is not None:
            mean = self.action_mean.cpu().numpy()
            std = self.action_std.cpu().numpy()
            actions_np = denormalize_action(actions_np, mean, std)

        actions_np = actions_np[:, : self.num_action_chunks, :]

        # Build forward_inputs for training replay
        forward_inputs: dict[str, Any] = {}
        for k, v in vlm_batch.items():
            if isinstance(v, torch.Tensor):
                forward_inputs[k] = v.detach().cpu()
        forward_inputs["action"] = torch.from_numpy(
            actions_np.reshape(batch_size, -1).astype(np.float32)
        )

        # Stub logprobs/values (real computation deferred to PR 3)
        prev_logprobs = torch.zeros(batch_size)
        prev_values = torch.zeros(batch_size)

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

        Stub: returns zeros.  Full implementation (PR 3) will replay the
        denoising chain and compute logprobs via the flow-matching model.
        """
        if forward_inputs is None:
            forward_inputs = {}

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
        state_tensor: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert env observations to Qwen3-VL processor format.

        Args:
            images: ``[B, H, W, C]`` uint8 numpy arrays.
            task_descriptions: Language instructions per sample.
            state_tensor: Proprioceptive state ``[B, STATE_DIM]``.
            device: Target device.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``pixel_values``,
            ``image_grid_thw``, ``state``.
        """
        # 1. Convert numpy images to PIL, resize
        pil_images = []
        for img_np in images:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            pil_img = resize_image(pil_img, factor=32, max_pixels=90000)
            pil_images.append(pil_img)

        # 2. Build Qwen3-VL chat messages
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

        # 3. Process with Qwen3-VL processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )

        # 4. Move to device, add state
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
