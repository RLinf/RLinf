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
Extended from vla_lib to support batch inference for faster advantage computation.

Inference modes:
1. VLM-based (token prediction):
   - "continuous": Compute expectation over bin token probabilities (V = Σ p_i * v_i)
   - "discrete": Use the bin with highest probability (argmax)

2. Expert-based (direct value prediction):
   - "expert_mse": Use expert model for continuous scalar output
   - "expert_categorical": Use expert model with categorical output (expectation)
   - "expert_distributional": Use expert model with distributional output (expectation)
"""

import logging
import re
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from vla_lib.datasets.vla_datasets.lerobot_datasets.transforms import (
    DataTransformFn,
    compose,
)

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ValuePolicy(BasePolicy):
    """Policy that predicts return values instead of actions.

    This mirrors Policy class architecture but outputs value predictions:
    - Input transforms are applied to observations before inference
    - Model predicts value via VLM token distribution or expert direct output
    - Value is computed based on value_mode

    Extended to support batch inference via infer_batch() method.
    """

    # Expert modes that use PI05ValueCritic's predict_value() directly
    EXPERT_MODES = ("expert_mse", "expert_categorical", "expert_distributional")
    # VLM modes that use token logits prediction
    VLM_MODES = ("continuous", "discrete")

    def __init__(
        self,
        model,
        *,
        transforms: Sequence[DataTransformFn] = (),
        metadata: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        value_mode: str = "continuous",
        num_return_bins: int = 201,
        return_min: float = -1.0,
        return_max: float = 0.0,
        use_ar_generation: bool = False,
        **kwargs,
    ):
        """Initialize value policy.

        Args:
            model: PI05 model or PI05ValueCritic for value prediction
            transforms: Input transforms
            metadata: Policy metadata
            device: Device to run inference on
            checkpoint_dir: Checkpoint directory
            value_mode: Inference mode:
                - "continuous": VLM token logits, expectation over bins
                - "discrete": VLM token logits, argmax bin
                - "expert_mse": Expert model, continuous scalar output
                - "expert_categorical": Expert model, categorical output
                - "expert_distributional": Expert model, distributional output
            num_return_bins: Number of return bins (for VLM modes)
            return_min: Minimum return value
            return_max: Maximum return value
            use_ar_generation: If True and mode is "discrete", use AR generation
        """
        self._model = model
        self._input_transform = compose(transforms)
        self._device = device
        self._checkpoint_dir = checkpoint_dir

        self.metadata = metadata or {}
        self.value_mode = value_mode
        self.num_return_bins = num_return_bins
        self.return_min = return_min
        self.return_max = return_max
        self.use_ar_generation = use_ar_generation
        self._is_expert_mode = value_mode in self.EXPERT_MODES

        # Compute bin centers (used for VLM modes)
        bin_width = (return_max - return_min) / num_return_bins
        self.bin_centers = np.array(
            [return_min + (b + 0.5) * bin_width for b in range(num_return_bins)],
            dtype=np.float32,
        )

        self._model = self._model.to(device)
        self._model.eval()

        logger.info("ValuePolicy initialized:")
        logger.info(f"  Value mode: {value_mode}")
        if self._is_expert_mode:
            logger.info("  Using expert inference (PI05ValueCritic.predict_value)")
        else:
            logger.info("  Using VLM token inference")
            logger.info(f"  Return bins: {num_return_bins}")
        logger.info(f"  Return range: [{return_min}, {return_max}]")
        if use_ar_generation and not self._is_expert_mode:
            logger.info("  Using AR generation for discrete mode")

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

        # Apply input transforms
        inputs = self._input_transform(inputs)

        # Prepare observation for model (uses processor to tokenize, etc.)
        observation = self._prepare_observation(inputs)

        # Model inference
        start_time = time.monotonic()
        with torch.no_grad():
            value = self._predict_value(observation)
        model_time = time.monotonic() - start_time

        outputs = {
            "value": float(value),
            "state": obs.get("state", np.array([])),
            "policy_timing": {"infer_ms": model_time * 1000},
        }

        return outputs

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

        # Process in batches
        for batch_start in range(0, len(obs_list), batch_size):
            batch_end = min(batch_start + batch_size, len(obs_list))
            batch_obs = obs_list[batch_start:batch_end]

            # Apply input transforms to all observations
            inputs_list = []
            for obs in batch_obs:
                inputs = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
                inputs = self._input_transform(inputs)
                inputs_list.append(inputs)

            # Prepare batched observation
            observation = self._prepare_observation_batch(inputs_list)

            # Batch model inference
            with torch.no_grad():
                values = self._predict_value_batch(observation)

            # Build output list
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
        # Process each sample and collect
        all_images = []
        all_image_masks = []
        all_tokens = []
        all_masks = []
        all_ar_masks = []
        all_loss_masks = []

        for inputs in inputs_list:
            # Reuse existing single-sample logic, but only collect data
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_tokens.append(single_obs["tokenized_prompt"])
            all_masks.append(single_obs["tokenized_prompt_mask"])
            all_ar_masks.append(single_obs["token_ar_mask"])
            all_loss_masks.append(single_obs["token_loss_mask"])

        # Stack into batched tensors
        # Handle images (can be dict of tensors)
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

    def _predict_value_batch(self, observation: dict) -> torch.Tensor:
        """Predict values for batched observation.

        Args:
            observation: Batched observation dict with [B, ...] tensors

        Returns:
            Tensor of shape [B] with predicted values
        """
        batch_size = observation["tokenized_prompt"].shape[0]

        if self._is_expert_mode:
            # Expert mode: model.predict_value() already returns [B]
            values = self._model.predict_value(observation)
            return values.cpu()

        # VLM mode: handle batch logits
        processor = getattr(self._model, "processor", None)
        if processor is None:
            return torch.zeros(batch_size)

        value_token_ids = processor.get_value_token_ids()
        if value_token_ids is None:
            return torch.zeros(batch_size)

        model = self._model.model
        logits = model.predict_next_token_logits(observation)  # [B, vocab_size]

        bin_token_ids = torch.tensor(
            [value_token_ids[i] for i in range(self.num_return_bins)],
            dtype=torch.long,
            device=logits.device,
        )
        bin_logits = logits[:, bin_token_ids]  # [B, num_bins]
        bin_probs = F.softmax(bin_logits.float(), dim=1).cpu().numpy()

        if self.value_mode == "continuous":
            values = np.sum(bin_probs * self.bin_centers, axis=1)
        else:  # discrete
            values = self.bin_centers[np.argmax(bin_probs, axis=1)]

        return torch.from_numpy(values.astype(np.float32))

    def _predict_value(self, observation: dict) -> float:
        """Predict value from observation.

        Modes:
        1. Expert modes: Use PI05ValueCritic.predict_value() for direct output
        2. VLM logits-based: Use predict_next_token_logits for single-step
        3. VLM AR generation: Use full generate() for discrete mode

        Args:
            observation: Observation dict formatted for model.forward()

        Returns:
            Predicted value as a scalar
        """
        # Expert mode: use model's predict_value directly
        if self._is_expert_mode:
            return self._predict_value_expert(observation)

        # VLM mode: need processor for token IDs
        processor = getattr(self._model, "processor", None)
        if processor is None:
            logger.warning("Model has no processor, cannot predict value")
            return 0.0

        value_token_ids = processor.get_value_token_ids()
        if value_token_ids is None:
            logger.warning("Processor has no value token IDs")
            return 0.0

        # For discrete mode with AR generation, use full generate()
        if self.value_mode == "discrete" and self.use_ar_generation:
            return self._predict_value_ar(observation)

        # Default: use single-step logits prediction
        return self._predict_next_token_logits(observation, value_token_ids)

    def _predict_value_expert(self, observation: dict) -> float:
        """Predict value using expert model (PI05ValueCritic).

        Uses the model's predict_value() method which outputs continuous
        values directly (no token decoding needed).
        """
        values = self._model.predict_value(observation)  # Returns Tensor [B]
        return float(values[0].item())

    def _predict_next_token_logits(
        self, observation: dict, value_token_ids: dict[int, int]
    ) -> float:
        """Predict value using single-step logits (efficient).

        Uses predict_next_token_logits for a single forward pass to get logits at the
        last position, then computes value from the distribution.
        """
        model = self._model.model  # PI05FlowMatching
        logits = model.predict_next_token_logits(observation)  # [B, vocab_size]
        logits = logits[0]  # Take first batch element [vocab_size]

        # Get logits for value tokens
        bin_token_ids = torch.tensor(
            [value_token_ids[i] for i in range(self.num_return_bins)],
            dtype=torch.long,
            device=logits.device,
        )
        bin_logits = logits[bin_token_ids]  # [num_bins]

        # Compute probabilities
        bin_probs = F.softmax(bin_logits.float(), dim=0).cpu().numpy()

        if self.value_mode == "continuous":
            return float(np.sum(bin_probs * self.bin_centers))
        elif self.value_mode == "discrete":
            return float(self.bin_centers[np.argmax(bin_probs)])
        else:
            raise ValueError(f"Unknown value mode: {self.value_mode}")

    def _predict_value_ar(self, observation: dict) -> float:
        """Predict value using full AR generation (verifies generation pipeline).

        Uses model.generate() in VLM mode to autoregressively generate tokens,
        then extracts the value token from the generated sequence.
        Expected output format: "<vN><eos>" where N is the bin index.

        Raises:
            ValueError: If no value token is found in the generated sequence.
        """
        # Generate using VLM mode (only need 2 tokens: <vN> + <eos>)
        output = self._model.generate(
            observation,
            forward_mode="vlm",
            max_new_tokens=2,
            temperature=0.0,
        )
        output_tokens = output[0]  # [B, actual_len] - exact length, no padding

        # Decode generated tokens
        tokens = output_tokens[0].cpu().tolist()
        decoded = self._model.processor.tokenizer.decode(
            tokens, skip_special_tokens=False
        )

        # Extract bin index N from "<vN>" pattern
        match = re.search(r"<v(\d+)>", decoded)
        if match:
            bin_idx = int(match.group(1))
            if 0 <= bin_idx < self.num_return_bins:
                return float(self.bin_centers[bin_idx])
            raise ValueError(
                f"Bin index {bin_idx} out of range [0, {self.num_return_bins})"
            )

        raise ValueError(
            f"AR generation did not produce a valid value token. "
            f"Generated: '{decoded}', tokens: {tokens}"
        )

    def _prepare_observation(self, inputs: dict) -> dict:
        """Prepare observation dict for model.forward() in VLM inference mode.

        For value prediction inference:
        - Input: "Task: {prompt}. Value: " (no response token)
        - Model predicts the value token at the last position

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
        # Input format must EXACTLY match training tokenization in PI05Processor._tokenize_single
        #
        # Unified template: Task: {prompt}. [State: {state}. ][{prefix} ]
        cleaned_prompt = processor._clean_text(prompt)
        cleaned_prompt = processor._strip_trailing_punctuation(cleaned_prompt)

        value_prefix = "Value:"  # This matches config value.value_prefix

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

        # Move images to device (can be dict or tensor)
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
