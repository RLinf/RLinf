"""
Image and text processors for value model.

Contains:
- ValueImageProcessor: HuggingFace-style image processor handling resize, padding,
  augmentation, and multi-camera views.
- normalize_image_to_model_format: Utility to convert arbitrary image tensors
  to BCHW [-1, 1] float format.
- ValueProcessor: Processor combining image preprocessing and text tokenization.

Text template: ``Task: {prompt}.``

All tokens are bidirectional (ar_mask=0). The value model's expert head
predicts the value via a [CLS] token appended at the model level, not here.
"""

import logging
import os
import string
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

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
) -> torch.Tensor:
    """Resize an image to target size without distortion by padding with black.

    If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
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

    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_image_to_model_format(
    img: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Normalize a single image to model format (verified standard from policy._prepare_observation).

    Converts any image format to BCHW [-1, 1] float tensor.

    Args:
        img: Input image tensor (CHW, HWC, BCHW, or BHWC format; uint8 or float)
        device: Target device (optional)
        dtype: Target dtype (optional, e.g., torch.bfloat16)

    Returns:
        Tensor in BCHW format, normalized to [-1, 1], with optional dtype conversion
    """
    if device is not None:
        img = img.to(device)

    # Detect format: CHW/BCHW vs HWC/BHWC
    if img.dim() == 3:
        is_chw = img.shape[0] == 3
    elif img.dim() == 4:
        is_chw = img.shape[1] == 3
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.dim()}D")

    # Add batch dimension if needed
    if img.dim() == 3:
        img = img[None, ...]

    # Convert to float for normalization
    img = img.float()

    # Normalize to [-1, 1] and ensure BCHW format
    if is_chw:
        # Already BCHW, just normalize
        if img.max() > 1.0:
            img = img / 255.0 * 2.0 - 1.0
        elif img.min() >= 0.0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0
    else:
        # BHWC format, convert to BCHW
        img = img.permute(0, 3, 1, 2)
        if img.max() > 1.0:
            img = img / 255.0 * 2.0 - 1.0
        elif img.min() >= 0.0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0

    # Convert to target dtype if specified
    if dtype is not None:
        img = img.to(dtype)

    return img


# ---------------------------------------------------------------------------
# ValueImageProcessor
# ---------------------------------------------------------------------------

class ValueImageProcessor(ImageProcessingMixin):
    """
    Value model image processor that replicates OpenPI's preprocessing logic.

    Implements the exact image preprocessing pipeline from OpenPI:
    - Resize with padding to maintain aspect ratio
    - Training augmentations: crop, rotation, color jitter
    - Images kept in [-1, 1] range
    - Handles multiple camera views
    """

    model_input_names: ClassVar[List[str]] = ["pixel_values", "image_masks"]

    def __init__(
        self,
        image_size: Tuple[int, int] = IMAGE_RESOLUTION,
        do_resize: bool = True,
        do_augment: bool = True,
        image_keys: Sequence[str] = IMAGE_KEYS,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.do_resize = do_resize
        self.do_augment = do_augment
        self.image_keys = image_keys

    def apply_augmentations(
        self,
        image: torch.Tensor,
        is_wrist_camera: bool = False
    ) -> torch.Tensor:
        """
        Apply OpenPI-style augmentations to the image.

        Args:
            image: Input image tensor in BHWC format, range [-1, 1]
            is_wrist_camera: Whether this is a wrist camera (affects augmentation)

        Returns:
            Augmented image tensor in BHWC format, range [-1, 1]
        """
        # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
        image = image / 2.0 + 0.5

        if not is_wrist_camera:
            # Geometric augmentations for non-wrist cameras
            height, width = image.shape[1:3]

            # Random crop and resize (95% crop scale like OpenPI)
            crop_height = int(height * 0.95)
            crop_width = int(width * 0.95)

            # Random crop
            max_h = height - crop_height
            max_w = width - crop_width
            if max_h > 0 and max_w > 0:
                start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

            # Resize back to original size
            image = F.interpolate(
                image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Random rotation (small angles, -5 to 5 degrees like OpenPI)
            angle = torch.rand(1, device=image.device) * 10 - 5
            if torch.abs(angle) > 0.1:
                # Convert to radians
                angle_rad = angle * torch.pi / 180.0

                # Create rotation matrix
                cos_a = torch.cos(angle_rad)
                sin_a = torch.sin(angle_rad)

                # Apply rotation using grid_sample
                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)

                # Create meshgrid
                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                # Expand to batch dimension
                grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                # Apply rotation transformation
                grid_x_rot = grid_x * cos_a - grid_y * sin_a
                grid_y_rot = grid_x * sin_a + grid_y * cos_a

                # Stack and reshape for grid_sample
                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                image = F.grid_sample(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

        # Color augmentations for all cameras
        # Random brightness (0.7 to 1.3 like OpenPI)
        brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
        image = image * brightness_factor

        # Random contrast (0.6 to 1.4 like OpenPI)
        contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        # Random saturation (0.5 to 1.5 like OpenPI)
        saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
        gray = image.mean(dim=-1, keepdim=True)
        image = gray + (image - gray) * saturation_factor

        # Clamp values to [0, 1]
        image = torch.clamp(image, 0, 1)

        # Back to [-1, 1]
        image = image * 2.0 - 1.0

        return image

    def process_images(
        self,
        images_dict: Dict[str, torch.Tensor],
        image_masks_dict: Optional[Dict[str, torch.Tensor]] = None,
        train: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a batch of images efficiently.

        Matches policy._prepare_observation behavior (the verified standard):
        - Always outputs BCHW format
        - Normalizes to [-1, 1] float

        Args:
            images_dict: Dict with OpenPI camera keys, each tensor [B, C, H, W] or [B, H, W, C]
            image_masks_dict: Optional dict of image masks
            train: Whether to apply training augmentations

        Returns:
            Tuple of (processed_images_dict, processed_masks_dict)
            Images are returned in BCHW format, [-1, 1] float for model consumption.
        """
        out_images = {}
        out_masks = {}

        # Get batch size from any available image
        batch_size = None
        template_device = None
        for key in images_dict:
            if images_dict[key] is not None:
                batch_size = images_dict[key].shape[0]
                template_device = images_dict[key].device
                break

        for key in self.image_keys:
            image = images_dict.get(key)

            # Handle missing keys by creating placeholder zero images
            if image is None:
                if batch_size is not None:
                    h, w = self.image_size
                    placeholder = torch.zeros(batch_size, 3, h, w, device=template_device)
                    out_images[key] = placeholder
                    out_masks[key] = torch.zeros(batch_size, dtype=torch.bool, device=template_device)
                continue

            is_wrist = "wrist" in key

            # Detect input format: BCHW vs BHWC
            is_bchw = image.shape[1] == 3

            # Convert to BHWC for internal processing (resize, augmentations work on HWC)
            if is_bchw:
                image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC

            # Resize if needed (operates on BHWC)
            # Note: use tuple() for comparison since self.image_size may be a list after deserialization
            if self.do_resize and tuple(image.shape[1:3]) != tuple(self.image_size):
                image = resize_with_pad(image, self.image_size[1], self.image_size[0])
                # Ensure 4D output (resize_with_pad may squeeze batch dim)
                if image.dim() == 3:
                    image = image.unsqueeze(0)

            # Normalize to [-1, 1] (matching policy._prepare_observation)
            image = image.float()
            if image.max() > 1.0:
                # uint8 [0, 255] -> float32 [-1, 1]
                image = image / 255.0 * 2.0 - 1.0
            elif image.min() >= 0.0 and image.max() <= 1.0:
                # float [0, 1] -> float32 [-1, 1]
                image = image * 2.0 - 1.0
            # else: already in [-1, 1], leave as is

            # Apply augmentations if enabled (operates on [-1, 1] BHWC)
            if train and self.do_augment:
                image = self.apply_augmentations(image, is_wrist_camera=is_wrist)

            # Always output BCHW format (matching policy._prepare_observation)
            image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW

            out_images[key] = image

            # Handle masks
            if image_masks_dict is not None and key in image_masks_dict:
                out_masks[key] = image_masks_dict[key]
            else:
                # Default to True for all batch elements
                batch_size = image.shape[0]
                out_masks[key] = torch.ones(batch_size, dtype=torch.bool, device=image.device)

        return out_images, out_masks

    def __call__(
        self,
        images: Dict[str, torch.Tensor],
        image_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_augment: Optional[bool] = None,
        train: bool = False,
        **kwargs
    ) -> BatchFeature:
        """
        Process images for value model following OpenPI's preprocessing.

        Args:
            images: Dict of images with OpenPI camera keys or list/tensor of images
            image_masks: Optional dict of image masks
            return_tensors: Type of tensors to return
            do_augment: Whether to apply augmentations (overrides self.do_augment)
            train: Whether in training mode

        Returns:
            BatchFeature containing processed images and image masks
        """
        # Determine if we should apply augmentations
        apply_augmentations = train and (do_augment if do_augment is not None else self.do_augment)

        # Handle different input formats
        # Dict format with OpenPI camera keys - use batch processing
        output_images, output_masks = self.process_images(
            images, image_masks, train=apply_augmentations
        )

        return {
            "pixel_values": output_images,
            "image_masks": output_masks
        }


# ---------------------------------------------------------------------------
# ValueProcessor
# ---------------------------------------------------------------------------

class ValueProcessor(ProcessorMixin):
    """
    Value model processor combining image preprocessing and text tokenization.

    Text template: ``Task: {prompt}.``
    All tokens are bidirectional (ar_mask=0). The value model's expert head
    predicts the value via a [CLS] token appended at the model level.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ValueImageProcessor"
    tokenizer_class = "AutoTokenizer"
    _tokenize_log_count = 0

    @staticmethod
    def _default_tokenizer_path() -> Optional[str]:
        # Walk up to find the project root (where pyproject.toml lives)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                candidate = parent / "pretrained_models" / "paligemma-3b-mix-224"
                if candidate.exists():
                    return str(candidate)
                break
        return None

    def __init__(
        self,
        image_processor: Optional[ValueImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 200,
        tokenizer_name_or_path: Optional[str] = None,
        image_keys: Optional[tuple] = None,
        do_augment: bool = True,
        **kwargs
    ):
        if image_processor is None:
            # Use custom image_keys if provided, otherwise use defaults
            image_processor = (
                ValueImageProcessor(image_keys=image_keys, do_augment=do_augment)
                if image_keys
                else ValueImageProcessor(do_augment=do_augment)
            )

        if tokenizer is None:
            tokenizer_path = (
                tokenizer_name_or_path
                or os.environ.get("VLA_TOKENIZER_PATH")
                or ValueProcessor._default_tokenizer_path()
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
        # Required for save_pretrained compatibility with transformers ProcessorMixin
        self.chat_template = None
        self.audio_tokenizer = None

    def _clean_text(self, text: str) -> str:
        """Clean text by stripping and normalizing."""
        return text.lower().strip().replace("_", " ").replace("\n", " ")

    def _strip_trailing_punctuation(self, text: str) -> str:
        """Remove trailing punctuation from text, but preserve quotes."""
        if text and text[-1] in string.punctuation and text[-1] not in '"\'':
            return text[:-1]
        return text

    def _tokenize_single(
        self,
        prompt: str,
        max_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt into the value model's input format.

        Template: ``Task: {prompt}.``

        All tokens are bidirectional (ar_mask=0).  The model's forward() only
        uses (tokens, mask, ar_mask); loss_mask and kv_cache_mask are not
        consumed, so they are no longer produced here.

        Returns:
            (tokens, mask, ar_mask) as numpy arrays of shape (max_length,).
        """
        if max_length is None:
            max_length = self.max_token_len

        cleaned = self._strip_trailing_punctuation(self._clean_text(prompt))
        prefix_text = f"Task: {cleaned}."
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
                    seq_len, max_length,
                )
            tokens = tokens[:max_length]
            mask = [True] * max_length

        ar_mask = [0] * max_length  # all bidirectional

        # Debug logging (first 2 examples on rank 0)
        worker_info = torch.utils.data.get_worker_info()
        is_worker_0 = worker_info is None or worker_info.id == 0
        if (
            is_worker_0
            and int(os.environ.get("RANK", 0)) == 0
            and ValueProcessor._tokenize_log_count < 2
        ):
            ValueProcessor._tokenize_log_count += 1
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            logger.info(
                "[Tokenization Example #%d] prompt=%r → %r  (len=%d)",
                self._tokenize_log_count, prompt, decoded, seq_len,
            )

        return np.asarray(tokens), np.asarray(mask), np.asarray(ar_mask)

    def process_text(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of prompts for the value model.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``token_ar_mask``.
        """
        if max_length is None:
            max_length = self.max_token_len

        batch_tokens = []
        batch_masks = []
        batch_ar_masks = []

        for prompt in prompts:
            tokens, mask, ar_mask = self._tokenize_single(
                prompt=prompt,
                max_length=max_length,
            )
            batch_tokens.append(tokens)
            batch_masks.append(mask)
            batch_ar_masks.append(ar_mask)

        result = {
            "input_ids": np.stack(batch_tokens),
            "attention_mask": np.stack(batch_masks),
            "token_ar_mask": np.stack(batch_ar_masks),
        }

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor] = None,
        image_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_tensors: Optional[str] = 'pt',
        train: bool = False,
        **kwargs
    ) -> BatchFeature:
        """
        Process text and images for value model.

        Args:
            text: Input text (prompt)
            images: Image dict with camera keys
            image_masks: Optional image masks
            return_tensors: Output tensor format
            train: Whether in training mode
        """
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
                train=train
            )
            result_data.update(image_inputs)

        return BatchFeature(data=result_data, tensor_type=return_tensors)

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode tokens to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = [t for t in token_ids if t != 0]
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids_batch: Union[List[List[int]], torch.Tensor], **kwargs) -> List[str]:
        """Decode batch of tokens to text."""
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
            "token_ar_mask",
        ]


__all__ = [
    "ValueImageProcessor",
    "ValueProcessor",
    "normalize_image_to_model_format",
]
