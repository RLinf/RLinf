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

"""
Value Policy class for VLA models with batch inference support.

This module provides a ValuePolicy class that predicts return values instead of actions.
Uses expert-based categorical value prediction via PI05ValueCritic.
"""

import logging
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
from rlinf.datasets.lerobot.transforms import (
    DataTransformFn,
    compose,
)

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ValuePolicy(BasePolicy):
    """Policy that predicts return values instead of actions.

    Uses PI05ValueCritic's expert model for direct categorical value prediction.
    Supports both single and batch inference.
    """

    def __init__(
        self,
        model,
        *,
        transforms: Sequence[DataTransformFn] = (),
        metadata: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        num_return_bins: int = 201,
        return_min: float = -1.0,
        return_max: float = 0.0,
        **kwargs,
    ):
        """Initialize value policy.

        Args:
            model: PI05ValueCritic for value prediction
            transforms: Input transforms
            metadata: Policy metadata
            device: Device to run inference on
            checkpoint_dir: Checkpoint directory
            num_return_bins: Number of return bins
            return_min: Minimum return value
            return_max: Maximum return value
        """
        self._model = model
        self._input_transform = compose(transforms)
        self._device = device
        self._checkpoint_dir = checkpoint_dir

        self.metadata = metadata or {}
        self.num_return_bins = num_return_bins
        self.return_min = return_min
        self.return_max = return_max

        self._model = self._model.to(device)
        self._model.eval()

        logger.info("ValuePolicy initialized:")
        logger.info("  Using expert inference (PI05ValueCritic.predict_value)")
        logger.info(f"  Return range: [{return_min}, {return_max}]")

    def infer(self, obs: dict, *, noise: Optional[np.ndarray] = None) -> dict:
        """Infer value from observations.

        Args:
            obs: Observation dictionary
            noise: Optional noise (unused for value prediction)

        Returns:
            Dictionary containing predicted value and timing information
        """
        inputs = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()
        }

        inputs = self._input_transform(inputs)
        observation = self._prepare_observation(inputs)

        start_time = time.monotonic()
        with torch.no_grad():
            values = self._model.predict_value(observation)
            value = float(values[0].item())
        model_time = time.monotonic() - start_time

        return {
            "value": value,
            "state": obs.get("state", np.array([])),
            "policy_timing": {"infer_ms": model_time * 1000},
        }

    def infer_batch(self, obs_list: list[dict], *, batch_size: int = 64) -> list[dict]:
        """Batch inference for multiple observations.

        Args:
            obs_list: List of observation dictionaries
            batch_size: Maximum batch size for single forward pass

        Returns:
            List of dictionaries containing predicted values
        """
        if len(obs_list) == 0:
            return []

        all_outputs = []

        for batch_start in range(0, len(obs_list), batch_size):
            batch_end = min(batch_start + batch_size, len(obs_list))
            batch_obs = obs_list[batch_start:batch_end]

            inputs_list = []
            for obs in batch_obs:
                inputs = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
                inputs = self._input_transform(inputs)
                inputs_list.append(inputs)

            observation = self._prepare_observation_batch(inputs_list)

            with torch.no_grad():
                values = self._model.predict_value(observation).cpu()

            for i, obs in enumerate(batch_obs):
                all_outputs.append(
                    {
                        "value": float(values[i]),
                        "state": obs.get("state", np.array([])),
                    }
                )

        return all_outputs

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        """Prepare batched observation dict for model.forward().

        Args:
            inputs_list: List of input dicts (after transforms)

        Returns:
            Batched observation dict with [B, ...] tensors
        """
        all_images = []
        all_image_masks = []
        all_tokens = []
        all_masks = []
        all_ar_masks = []
        all_loss_masks = []

        for inputs in inputs_list:
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_tokens.append(single_obs["tokenized_prompt"])
            all_masks.append(single_obs["tokenized_prompt_mask"])
            all_ar_masks.append(single_obs["token_ar_mask"])
            all_loss_masks.append(single_obs["token_loss_mask"])

        if isinstance(all_images[0], dict):
            batched_images = {
                k: torch.cat([img[k] for img in all_images], dim=0)
                for k in all_images[0].keys()
            }
            batched_masks = {
                k: torch.cat([m[k] for m in all_image_masks], dim=0)
                for k in all_image_masks[0].keys()
            }
        else:
            batched_images = torch.cat(all_images, dim=0)
            batched_masks = torch.cat(all_image_masks, dim=0)

        return {
            "images": batched_images,
            "image_masks": batched_masks,
            "tokenized_prompt": torch.cat(all_tokens, dim=0),
            "tokenized_prompt_mask": torch.cat(all_masks, dim=0),
            "token_ar_mask": torch.cat(all_ar_masks, dim=0),
            "token_loss_mask": torch.cat(all_loss_masks, dim=0),
        }

    def _prepare_observation(self, inputs: dict) -> dict:
        """Prepare observation dict for model.forward().

        For value prediction inference:
        - Input: "Task: {prompt}. Value: " (no response token)
        - Model predicts the value at the [CLS] position

        Output format matches PI05DataCollator:
        - images: [B, num_cameras, C, H, W]
        - image_masks: [B, num_cameras]
        - tokenized_prompt: [B, seq]
        - tokenized_prompt_mask: [B, seq]
        - token_ar_mask: [B, seq]
        - token_loss_mask: [B, seq]
        """
        processor = getattr(self._model, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Model processor not attached. Cannot prepare observation for inference."
            )

        # Get images (after transforms, may be in 'image' key as a dict)
        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key in inputs:
                if "image" in key.lower() and isinstance(
                    inputs[key], (np.ndarray, torch.Tensor)
                ):
                    img_key = key
                    for prefix in [
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ]:
                        img_key = img_key.replace(prefix, "")
                    images_dict[img_key] = inputs[key]

        # Get state
        state = inputs.get("state", inputs.get("observation/state"))
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # Get prompt
        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        # Prepare images - convert CHW to BHWC format for image_processor
        images_bhwc = {}

        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            if img.dim() == 3:
                img = img.unsqueeze(0).permute(0, 2, 3, 1)  # CHW -> BHWC
            elif img.dim() == 4:
                img = img.permute(0, 2, 3, 1)  # BCHW -> BHWC

            images_bhwc[cam_name] = img

        # Get image masks from inputs if available
        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc.keys():
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor([mask], dtype=torch.bool)
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        # Process images
        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

        # For inference: tokenize prefix only (no response token)
        # Unified template: Task: {prompt}. [State: {state}. ][{prefix} ]
        cleaned_prompt = processor._clean_text(prompt)
        cleaned_prompt = processor._strip_trailing_punctuation(cleaned_prompt)

        value_prefix = "Value:"

        if processor.discrete_state_input and state is not None:
            state_str = processor._discretize_state(state)
            prefix_text = f"Task: {cleaned_prompt}. State: {state_str}. {value_prefix} "
        else:
            prefix_text = f"Task: {cleaned_prompt}. {value_prefix} "

        tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=True)
        seq_len = len(tokens)

        # For inference: all tokens are bidirectional (ar_mask=0), no loss (loss_mask=False)
        max_length = processor.max_token_len
        if seq_len < max_length:
            padding_len = max_length - seq_len
            mask = [True] * seq_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
            ar_mask = [0] * max_length
            loss_mask = [False] * max_length
        else:
            tokens = tokens[:max_length]
            mask = [True] * max_length
            ar_mask = [0] * max_length
            loss_mask = [False] * max_length

        # Build observation dict - handle dict of tensors for images
        pixel_values = processed_img["pixel_values"]
        image_masks = processed_img["image_masks"]

        if isinstance(pixel_values, dict):
            images_on_device = {k: v.to(self._device) for k, v in pixel_values.items()}
        else:
            images_on_device = pixel_values.to(self._device)

        if isinstance(image_masks, dict):
            masks_on_device = {k: v.to(self._device) for k, v in image_masks.items()}
        else:
            masks_on_device = image_masks.to(self._device)

        observation = {
            "images": images_on_device,
            "image_masks": masks_on_device,
            "tokenized_prompt": torch.tensor(
                [tokens], dtype=torch.long, device=self._device
            ),
            "tokenized_prompt_mask": torch.tensor(
                [mask], dtype=torch.bool, device=self._device
            ),
            "token_ar_mask": torch.tensor(
                [ar_mask], dtype=torch.long, device=self._device
            ),
            "token_loss_mask": torch.tensor(
                [loss_mask], dtype=torch.bool, device=self._device
            ),
        }

        return observation

    def reset(self) -> None:
        """Reset the policy state."""
        pass
