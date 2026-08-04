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

"""Shared image + text processing for embodied value/critic models.

The ``recap`` and ``steam`` processors are ~90% identical. The shared
logic lives here; the two flavours differ only in:

* **Output pixel range** — ``"signed"`` (``[-1, 1]``, recap: the
  processor itself applies the SigLIP ``mean=std=0.5`` normalisation) vs
  ``"unit"`` (``[0, 1]``, steam: it emits raw pixels and defers the encoder's
  real ``image_mean``/``image_std`` normalisation to the backbone, so the
  vision encoder stays swappable).
* **Default image resolution** — 224x224 vs the vision encoder's native size
  (384x384 for ``siglip-so400m-patch14-384``).
* **Prompt template** — ``"Task: {prompt}."`` vs ``"Task: {prompt}\\nValue: "``.

Concrete processors in ``recap/processing.py`` and
``steam/processing.py`` subclass these and only set the differing defaults, so
existing class names — and therefore checkpoint ``from_pretrained`` paths —
keep working.
"""

import logging
import os
import string
from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchFeature, PreTrainedTokenizerBase
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType

logger = logging.getLogger(__name__)


def resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
    *,
    float_pad_value: float = 0.0,
    float_clamp_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Resize an image to target size without distortion by padding with black.

    ``uint8`` images are clamped to ``[0, 255]`` and padded with ``0``. Float
    images are clamped to ``float_clamp_range`` and padded with
    ``float_pad_value`` so callers can keep either a ``[0, 1]`` (pad ``0.0``)
    or a ``[-1, 1]`` (pad ``-1.0``) convention.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        float_pad_value: Padding value for float32 inputs.
        float_clamp_range: ``(min, max)`` clamp applied to float32 inputs.

    Returns:
        Resized and padded tensor with same shape format as input.
    """
    added_batch_dim = False

    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
            added_batch_dim = True
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)
            added_batch_dim = True

    batch_size, channels, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # ``F.interpolate(mode="bilinear")`` does not support uint8 tensors, so
    # interpolate in float and cast back to the original dtype below.
    interpolation_input = (
        images.to(torch.float32) if images.dtype == torch.uint8 else images
    )
    resized_images = F.interpolate(
        interpolation_input,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
        constant_value: float = 0
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(
            float_clamp_range[0], float_clamp_range[1]
        )
        constant_value = float_pad_value
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
        if added_batch_dim:
            padded_images = padded_images.squeeze(0)

    return padded_images


# Camera view names. Chosen so pre-existing LeRobot datasets (and the openpi
# policies that repack to these keys) plug in without modification.
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def resolve_image_size(image_processor: Any) -> tuple[int, int]:
    """Resolve ``(height, width)`` from a HuggingFace image processor."""
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return (224, 224)


def resolve_vision_image_size(
    vision_repo_id: str,
    revision: Optional[str] = None,
) -> tuple[int, int]:
    """Load a vision processor and return its native image size."""
    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(
        vision_repo_id,
        revision=revision,
        use_fast=True,
    )
    return resolve_image_size(image_processor)


def normalize_image_to_range(
    img: torch.Tensor,
    output_range: str,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Convert any image format to BCHW float in the requested range.

    Args:
        img: Input image tensor (CHW, HWC, BCHW, or BHWC; uint8 or float).
        output_range: ``"unit"`` for ``[0, 1]`` or ``"signed"`` for ``[-1, 1]``.
        device: Optional target device.
        dtype: Optional target dtype (e.g. ``torch.bfloat16``).

    Returns:
        Tensor in BCHW format, normalised to the requested range.
    """
    if device is not None:
        img = img.to(device)

    if img.dim() == 3:
        is_chw = img.shape[0] == 3
    elif img.dim() == 4:
        is_chw = img.shape[1] == 3
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.dim()}D")

    if img.dim() == 3:
        img = img[None, ...]

    img = img.float()

    if not is_chw:
        img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW

    if output_range == "signed":
        if img.max() > 1.0:
            img = img / 255.0 * 2.0 - 1.0
        elif img.min() >= 0.0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0
        # else: already in [-1, 1], leave as is
    else:  # "unit"
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(0.0, 1.0)

    if dtype is not None:
        img = img.to(dtype)
    return img


class BaseMultiViewImageProcessor(ImageProcessingMixin):
    """Multi-camera image processor with OpenPI-style augmentations.

    Resizes raw multi-camera input to ``image_size`` and outputs BCHW float
    tensors in the range selected by ``output_range`` (``"unit"`` → ``[0, 1]``,
    ``"signed"`` → ``[-1, 1]``). Augmentations always run internally in
    ``[0, 1]`` and convert at the boundaries.
    """

    model_input_names: ClassVar[list[str]] = ["pixel_values", "image_masks"]

    # Subclasses override these two class attributes.
    output_range: str = "unit"
    DEFAULT_IMAGE_SIZE: ClassVar[tuple[int, int]] = (224, 224)

    def __init__(
        self,
        image_size: Optional[tuple[int, int]] = None,
        do_resize: bool = True,
        do_augment: bool = True,
        image_keys: Sequence[str] = IMAGE_KEYS,
        output_range: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = (
            image_size if image_size is not None else self.DEFAULT_IMAGE_SIZE
        )
        self.do_resize = do_resize
        self.do_augment = do_augment
        self.image_keys = image_keys
        if output_range is not None:
            self.output_range = output_range
        if self.output_range not in ("unit", "signed"):
            raise ValueError(
                f"output_range must be 'unit' or 'signed', got {self.output_range!r}"
            )

    @property
    def _float_pad_value(self) -> float:
        return -1.0 if self.output_range == "signed" else 0.0

    @property
    def _float_clamp_range(self) -> tuple[float, float]:
        return (-1.0, 1.0) if self.output_range == "signed" else (0.0, 1.0)

    def _to_output_range(self, image: torch.Tensor) -> torch.Tensor:
        """Normalise a resized BHWC float image to ``self.output_range``."""
        image = image.float()
        if self.output_range == "signed":
            if image.max() > 1.0:
                image = image / 255.0 * 2.0 - 1.0
            elif image.min() >= 0.0 and image.max() <= 1.0:
                image = image * 2.0 - 1.0
            # else: already in [-1, 1], leave as is
        else:  # "unit"
            if image.max() > 1.0:
                image = image / 255.0
            image = image.clamp(0.0, 1.0)
        return image

    def apply_augmentations(
        self, image: torch.Tensor, is_wrist_camera: bool = False
    ) -> torch.Tensor:
        """Apply OpenPI-style augmentations.

        Operates internally in ``[0, 1]`` BHWC float and converts to/from the
        configured output range at the boundaries. Non-wrist cameras get
        geometric augmentations (crop+resize, small rotation); all cameras get
        brightness/contrast/saturation jitter.
        """
        if self.output_range == "signed":
            image = image / 2.0 + 0.5  # [-1, 1] -> [0, 1]

        if not is_wrist_camera:
            height, width = image.shape[1:3]

            crop_height = int(height * 0.95)
            crop_width = int(width * 0.95)

            max_h = height - crop_height
            max_w = width - crop_width
            if max_h > 0 and max_w > 0:
                start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                image = image[
                    :,
                    start_h : start_h + crop_height,
                    start_w : start_w + crop_width,
                    :,
                ]

            image = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            angle = torch.rand(1, device=image.device) * 10 - 5
            if torch.abs(angle) > 0.1:
                angle_rad = angle * torch.pi / 180.0
                cos_a = torch.cos(angle_rad)
                sin_a = torch.sin(angle_rad)

                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)

                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                grid_x_rot = grid_x * cos_a - grid_y * sin_a
                grid_y_rot = grid_x * sin_a + grid_y * cos_a
                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                image = F.grid_sample(
                    image.permute(0, 3, 1, 2),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

        # Color augmentations (apply to all cameras)
        brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
        image = image * brightness_factor

        contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
        gray = image.mean(dim=-1, keepdim=True)
        image = gray + (image - gray) * saturation_factor

        image = torch.clamp(image, 0.0, 1.0)

        if self.output_range == "signed":
            image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return image

    def process_images(
        self,
        images_dict: dict[str, torch.Tensor],
        image_masks_dict: Optional[dict[str, torch.Tensor]] = None,
        train: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Process a batch of multi-camera images.

        Output:
            (processed_images_dict, processed_masks_dict)
            Images are BCHW float in ``self.output_range``. Missing camera keys
            get zero placeholders with mask ``False``.
        """
        out_images = {}
        out_masks = {}

        batch_size = None
        template_device = None
        for key in images_dict:
            if images_dict[key] is not None:
                batch_size = images_dict[key].shape[0]
                template_device = images_dict[key].device
                break

        for key in self.image_keys:
            image = images_dict.get(key)

            # Missing keys get placeholder zero images with mask=False.
            if image is None:
                if batch_size is not None:
                    h, w = self.image_size
                    placeholder = torch.zeros(
                        batch_size, 3, h, w, device=template_device
                    )
                    out_images[key] = placeholder
                    out_masks[key] = torch.zeros(
                        batch_size, dtype=torch.bool, device=template_device
                    )
                continue

            is_wrist = "wrist" in key

            is_bchw = image.shape[1] == 3
            if is_bchw:
                image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC

            # tuple() needed because self.image_size may be a list after
            # deserialization from a saved processor config.
            if self.do_resize and tuple(image.shape[1:3]) != tuple(self.image_size):
                image = resize_with_pad(
                    image,
                    self.image_size[1],
                    self.image_size[0],
                    float_pad_value=self._float_pad_value,
                    float_clamp_range=self._float_clamp_range,
                )
                if image.dim() == 3:
                    image = image.unsqueeze(0)

            image = self._to_output_range(image)

            if train and self.do_augment:
                image = self.apply_augmentations(image, is_wrist_camera=is_wrist)

            image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW

            out_images[key] = image

            if image_masks_dict is not None and key in image_masks_dict:
                out_masks[key] = image_masks_dict[key]
            else:
                bsize = image.shape[0]
                out_masks[key] = torch.ones(
                    bsize, dtype=torch.bool, device=image.device
                )

        return out_images, out_masks

    def __call__(
        self,
        images: dict[str, torch.Tensor],
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_augment: Optional[bool] = None,
        train: bool = False,
        **kwargs,
    ) -> BatchFeature:
        apply_augmentations = train and (
            do_augment if do_augment is not None else self.do_augment
        )

        output_images, output_masks = self.process_images(
            images, image_masks, train=apply_augmentations
        )

        return {"pixel_values": output_images, "image_masks": output_masks}


class BaseValueTextProcessor(ProcessorMixin):
    """Combined image + text processor for embodied value/critic models.

    Subclasses set the prompt template and the default image-processor class.
    The prompt text is ``self.text_template.format(prompt=...)``.
    """

    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    _tokenize_log_count = 0

    # Subclasses override these.
    image_processor_class: Optional[str] = None
    _default_image_processor_cls: ClassVar[Optional[type]] = None
    text_template: str = "Task: {prompt}."

    def __init__(
        self,
        image_processor: Optional[BaseMultiViewImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 200,
        tokenizer_name_or_path: Optional[str] = None,
        image_keys: Optional[tuple] = None,
        do_augment: bool = True,
        **kwargs,
    ):
        if image_processor is None:
            cls = self._default_image_processor_cls
            if cls is None:
                raise ValueError(
                    f"{type(self).__name__} must set _default_image_processor_cls "
                    "or be constructed with an explicit image_processor."
                )
            image_processor = (
                cls(image_keys=image_keys, do_augment=do_augment)
                if image_keys
                else cls(do_augment=do_augment)
            )

        if tokenizer is None:
            tokenizer_path = tokenizer_name_or_path or os.environ.get(
                "VLA_TOKENIZER_PATH"
            )
            if not tokenizer_path or not os.path.exists(tokenizer_path):
                raise ValueError(
                    f"No tokenizer found. Provide tokenizer_name_or_path, "
                    f"set VLA_TOKENIZER_PATH env var, or place tokenizer files "
                    f"in the project pretrained_models directory. "
                    f"Tried: {tokenizer_path!r}"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, add_bos_token=True, local_files_only=True
            )

        self.image_processor = image_processor
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_token_len = max_token_len
        self.tokenizer_name_or_path = tokenizer_name_or_path
        # Required for save_pretrained compatibility with ProcessorMixin.
        self.chat_template = None
        self.audio_tokenizer = None

    def _clean_text(self, text: str) -> str:
        return text.lower().strip().replace("_", " ").replace("\n", " ")

    def _strip_trailing_punctuation(self, text: str) -> str:
        if text and text[-1] in string.punctuation and text[-1] not in "\"'":
            return text[:-1]
        return text

    def _build_prefix_text(self, prompt: str) -> str:
        """Build the prefix text from a task prompt."""
        cleaned = self._strip_trailing_punctuation(self._clean_text(prompt))
        return self.text_template.format(prompt=cleaned)

    def _tokenize_single(
        self,
        prompt: str,
        max_length: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_length is None:
            max_length = self.max_token_len

        prefix_text = self._build_prefix_text(prompt)
        tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)

        seq_len = len(tokens)
        if seq_len < max_length:
            pad = max_length - seq_len
            mask = [True] * seq_len + [False] * pad
            tokens = tokens + [0] * pad
        else:
            if seq_len > max_length:
                logger.warning(
                    "Token length (%d) exceeds max (%d), truncating.",
                    seq_len,
                    max_length,
                )
            tokens = tokens[:max_length]
            mask = [True] * max_length

        worker_info = torch.utils.data.get_worker_info()
        is_worker_0 = worker_info is None or worker_info.id == 0
        if (
            is_worker_0
            and int(os.environ.get("RANK", 0)) == 0
            and type(self)._tokenize_log_count < 2
        ):
            type(self)._tokenize_log_count += 1
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            logger.info(
                "[Tokenization Example #%d] prompt=%r → %r  (raw_len=%d, pad_to=%d)",
                type(self)._tokenize_log_count,
                prompt,
                decoded,
                seq_len,
                max_length,
            )

        return np.asarray(tokens), np.asarray(mask)

    def process_text(
        self,
        prompts: list[str],
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of prompts.

        Args:
            prompts: List of task prompts.
            max_length: Padding/truncation length (defaults to ``max_token_len``).
            return_tensors: ``"pt"`` returns ``torch.Tensor``; else ``np.ndarray``.
        """
        del kwargs  # accepted for forward/back-compat signature parity
        if max_length is None:
            max_length = self.max_token_len

        batch_tokens = []
        batch_masks = []
        for prompt in prompts:
            tokens, mask = self._tokenize_single(prompt=prompt, max_length=max_length)
            batch_tokens.append(tokens)
            batch_masks.append(mask)

        result = {
            "input_ids": np.stack(batch_tokens),
            "attention_mask": np.stack(batch_masks),
        }

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Union[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor] = None,
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        return_tensors: Optional[str] = "pt",
        train: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either text or images")

        result_data = {}

        if text is not None:
            is_batched = isinstance(text, list)
            texts = text if is_batched else [text]

            processed = self.process_text(
                prompts=texts,
                return_tensors=return_tensors,
            )
            result_data.update(processed)

            if not is_batched:
                for key in result_data:
                    if result_data[key].dim() > 0:
                        result_data[key] = result_data[key][0]

        if images is not None:
            image_inputs = self.image_processor(
                images,
                image_masks=image_masks,
                return_tensors=return_tensors,
                train=train,
            )
            result_data.update(image_inputs)

        return BatchFeature(data=result_data, tensor_type=return_tensors)

    def decode(self, token_ids: Union[list[int], torch.Tensor], **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = [t for t in token_ids if t != 0]
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(
        self, token_ids_batch: Union[list[list[int]], torch.Tensor], **kwargs
    ) -> list[str]:
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        return [self.decode(tokens, **kwargs) for tokens in token_ids_batch]

    @property
    def model_input_names(self):
        return [
            "pixel_values",
            "image_masks",
            "input_ids",
            "attention_mask",
        ]


__all__ = [
    "BaseMultiViewImageProcessor",
    "BaseValueTextProcessor",
    "normalize_image_to_range",
    "resize_with_pad",
    "resolve_image_size",
    "resolve_vision_image_size",
    "IMAGE_KEYS",
]
