"""
processing_pi0.py

HuggingFace-style processor for PI0 VLA model, adapted from OpenPI preprocessing logic.
Handles image preprocessing and text tokenization for PI0 models.
"""

from collections.abc import Sequence
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import TensorType
from transformers import AutoImageProcessor, AutoProcessor

from rlinf.utils.image_utils import resize_with_pad
from rlinf.utils.dist_utils import get_logger

logger = get_logger(__name__)

# OpenPI constants
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb", 
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


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


class PI0ImageProcessor(ImageProcessingMixin):
    """
    PI0 Image Processor that replicates OpenPI's preprocessing logic.
    
    Implements the exact image preprocessing pipeline from OpenPI:
    - Resize with padding to maintain aspect ratio 
    - Training augmentations: crop, rotation, color jitter
    - Images kept in [-1, 1] range as expected by PI0 models
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
        Process images for PI0 model following OpenPI's preprocessing.
        
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
        
        # Convert to list format for consistent output
        # output_images = list(processed_images_dict.values())
        # output_masks = list(processed_masks_dict.values())
        
        # Return as list for multi-camera compatibility
        # pixel_values = torch.stack(output_images)
        # image_masks = torch.stack(output_masks)
        # logger.info(f"Output images: {[(img.shape, img.dtype, img.min(), img.max()) for img in output_images.values()]}")
        # logger.info(f"Output masks: {[(mask.shape, mask.dtype, mask.min(), mask.max()) for mask in output_masks.values()]}")
        return {
            "pixel_values": output_images,
            "image_masks": output_masks
        }


class PI0Processor(ProcessorMixin):
    """
    PI0 Processor that combines image preprocessing and text tokenization.
    
    This processor provides both OpenPI-compatible and standard transformers formats:
    - Image preprocessing matching OpenPI's behavior exactly
    - Text tokenization using HuggingFace tokenizer
    - Flexible output format for different use cases
    """
    
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "PI0ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    @staticmethod
    def _default_tokenizer_path() -> Optional[str]:
        """Return local tokenizer path if bundled in repo."""
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / "pretrained_models" / "paligemma-3b-mix-224"
        return str(candidate) if candidate.exists() else None

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 48,
        tokenizer_name_or_path: Optional[str] = None,
        **kwargs
    ):
        # Initialize image processor if not provided
        if image_processor is None:
            image_processor = PI0ImageProcessor()
            
        # Initialize tokenizer if not provided
        if tokenizer is None:
            tokenizer_path = (
                tokenizer_name_or_path
                or os.environ.get("VLA_TOKENIZER_PATH")
                or PI0Processor._default_tokenizer_path()
            )
            tokenizer_kwargs = {"add_bos_token": True}
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer_kwargs["local_files_only"] = True
                tokenizer_source = tokenizer_path
            else:
                tokenizer_source = tokenizer_path or "google/paligemma-3b-pt-224"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
            
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.chat_template = kwargs.pop("chat_template", None)
        self.audio_tokenizer = kwargs.pop("audio_tokenizer", None)

    def _tokenize_text(
        self, 
        text: str, 
        max_length: int = None, 
        padding: bool = True, 
        truncation: bool = True, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize text following OpenPI logic."""
        if max_length is None:
            max_length = self.max_token_len

        # Clean text like OpenPI
        cleaned_text = text.strip().replace("_", " ").replace("\n", " ")

        # Tokenize with special tokens + newline (like OpenPI)
        tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=True) \
            + self.tokenizer.encode("\n", add_special_tokens=False)
        tokens_len = len(tokens)

        if padding and tokens_len < max_length:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
            pad_token = [pad_token_id] * (max_length - tokens_len)
            tokens = tokens + pad_token
            mask = [True] * tokens_len + [False] * (max_length - tokens_len)
        elif truncation and tokens_len > max_length:
            tokens = tokens[:max_length]
            mask = [True] * max_length
        else:
            mask = [True] * tokens_len

        return np.asarray(tokens), np.asarray(mask)

    def process_text(
        self, 
        text: Union[str, List[str]], 
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = 'pt',
        return_attention_mask: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text in standard transformers format.
        
        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_token_len

        is_batched = isinstance(text, list)
        if not is_batched:
            text = [text]

        batch_input_ids = []
        batch_attention_mask = []
        for txt in text:
            tokens, mask = self._tokenize_text(txt, max_length, padding, truncation)
            batch_input_ids.append(tokens.tolist())
            batch_attention_mask.append(mask.astype(int).tolist())

        result = {
            "input_ids": batch_input_ids,
        }
        
        if return_attention_mask:
            result["attention_mask"] = batch_attention_mask
        
        # Convert to BatchEncoding and handle tensor conversion
        encoding = BatchEncoding(result, tensor_type=return_tensors)
        
        # If input was not batched, remove the batch dimension
        if not is_batched:
            for key in encoding.keys():
                encoding[key] = encoding[key][0]
        
        return encoding

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor] = None,
        image_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = 'pt',
        train: bool = False,
        **kwargs
    ) -> BatchFeature:
        """
        Process text and images for PI0 model.
        
        Args:
            text: Text inputs (language instructions)
            images: Dict of images with OpenPI camera keys or list/tensor of images
            image_masks: Optional dict of image masks
            return_tensors: Format of returned tensors
            train: Whether in training mode (affects augmentations)
            
        Returns:
            BatchFeature with processed inputs
        """
        if text is None and images is None:
            raise ValueError("You must provide either text or images")
        
        result_data = {}
        
        # Process text if provided
        if text is not None:
            text_inputs = self.process_text(
                text, 
                return_tensors=return_tensors, 
                **kwargs
            )
            result_data.update(text_inputs)
        
        # Process images if provided  
        if images is not None:
            image_inputs = self.image_processor(
                images, 
                image_masks=image_masks,
                return_tensors=return_tensors,
                train=train
            )
            result_data.update(image_inputs)
        
        return BatchFeature(data=result_data, tensor_type=return_tensors)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode a single token sequence to text."""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, token_ids_batch: List[List[int]], **kwargs) -> List[str]:
        """Decode multiple token sequences to text."""
        return self.tokenizer.batch_decode(token_ids_batch, **kwargs)

    @property
    def model_input_names(self):
        """Return list of expected model input names."""
        return [
            "pixel_values", 
            "image_masks", 
            "input_ids",
            "attention_mask",
        ]


# Register our custom classes with HuggingFace AutoClass system
try:
    # Import PI0Config for registration
    from .configuration_pi0 import PI0Config
    
    # Register our custom classes
    AutoImageProcessor.register(PI0Config, PI0ImageProcessor)
    AutoProcessor.register(PI0Config, PI0Processor)
    
except ImportError:
    # If PI0Config is not available during import, registration will be handled elsewhere
    logger.warning("PI0Config not found, skipping registration")

# Export classes for easy import
__all__ = ["PI0Processor", "PI0ImageProcessor"]
