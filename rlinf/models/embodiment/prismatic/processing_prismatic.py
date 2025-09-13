# Copyright 2025 The RLinf Authors.
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

from typing import ClassVar, Dict, List, Optional, Union

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor as PrismaticImageProcessorOrginal,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticProcessor as PrismaticProcessorOriginal,
)
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class PrismaticImageProcessor(PrismaticImageProcessorOrginal):
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])
        img: [B, num_images, C, H, W]
        """
        if self.tvf_do_letterbox:
            raise NotImplementedError("Letterbox padding is not yet supported!")

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        batch_size = img.shape[0]
        img = img.reshape(-1, *img.shape[2:])

        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])

            if isinstance(img_idx, Image.Image):
                img_idx = TVF.to_tensor(img_idx)

            img_idx = img_idx / 255.0
            img_idx = TVF.normalize(img_idx, **self.tvf_normalize_params[idx])

            imgs_t.append(img_idx)

        # [Contract] `imgs_t` is a list of Tensors of shape [B, num_images, C, H, W]; stack along dim C
        img_t = torch.cat(imgs_t, dim=1)
        img_t = img_t.reshape(
            batch_size, -1, *img_t.shape[1:]
        )  # [B, num_images, C * n, H, W]

        return img_t

    def preprocess(
        self,
        images: torch.Tensor,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.
        @param images: [B, C, H, W]
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = self.apply_transform(images)

        # Dict[str, torch.Tensor]
        return BatchFeature(
            data={"pixel_values": pixel_values}, tensor_type=return_tensors
        )

    def __call__(self, images: torch.Tensor, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class PrismaticProcessor(PrismaticProcessorOriginal):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        images: torch.Tensor,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: torch.Tensor [B, C, H, W].
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        assert self.tokenizer.padding_side == "left", (
            "Required: Init tokenizer with padding_side='left'"
        )

        pixel_values = self.image_processor(images, return_tensors=return_tensors)[
            "pixel_values"
        ]
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        input_ids = text_inputs["input_ids"]  # [B, L]
        attention_mask = text_inputs["attention_mask"]  # [B, L]

        first_nonzero_indices = torch.argmax(attention_mask, dim=1).unsqueeze(
            1
        )  # [B, 1]
        # assert first token is BOS token
        assert torch.all(
            input_ids.gather(1, first_nonzero_indices) == self.tokenizer.bos_token_id
        )
        # assert left padding
        assert torch.all(input_ids[:, -1] != self.tokenizer.pad_token_id)

        input_ids.scatter_(1, first_nonzero_indices, self.tokenizer.pad_token_id)
        attention_mask.scatter_(1, first_nonzero_indices, 0)

        input_ids[:, 0] = self.tokenizer.bos_token_id
        attention_mask[:, 0] = 1

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            print(pixel_values.shape, text_inputs.input_ids.shape)
            raise ValueError(
                "Batch is malformed; expected same number of images and text inputs!"
            )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})


class MultiInputPrismaticProcessor(PrismaticProcessorOriginal):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        images: Dict[str, torch.Tensor],
        proprio_states: torch.Tensor,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: torch.Tensor [B, C, H, W].
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        assert self.tokenizer.padding_side == "left", (
            "Required: Init tokenizer with padding_side='left'"
        )

        all_pixel_values = []
        for image_key in images:
            all_pixel_values.append(
                self.image_processor(images[image_key], return_tensors=return_tensors)[
                    "pixel_values"
                ]
            )

        input_pixel_values = torch.cat(all_pixel_values, dim=1)

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        input_ids = text_inputs["input_ids"]  # [B, L]
        attention_mask = text_inputs["attention_mask"]  # [B, L]

        first_nonzero_indices = torch.argmax(attention_mask, dim=1).unsqueeze(
            1
        )  # [B, 1]
        # assert first token is BOS token
        assert torch.all(
            input_ids.gather(1, first_nonzero_indices) == self.tokenizer.bos_token_id
        )
        # assert left padding
        assert torch.all(input_ids[:, -1] != self.tokenizer.pad_token_id)

        input_ids.scatter_(1, first_nonzero_indices, self.tokenizer.pad_token_id)
        attention_mask.scatter_(1, first_nonzero_indices, 0)

        input_ids[:, 0] = self.tokenizer.bos_token_id
        attention_mask[:, 0] = 1

        # [Validate] Need same number of images and text inputs!
        if input_pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            print(input_pixel_values.shape, text_inputs.input_ids.shape)
            raise ValueError(
                "Batch is malformed; expected same number of images and text inputs!"
            )

        return BatchFeature(data={**text_inputs, "pixel_values": input_pixel_values})
