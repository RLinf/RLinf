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

"""Image and text processors for the value model.

Thin subclasses of the shared base in
:mod:`rlinf.models.embodiment.value_model.value_common.image_text_processing`. The value model
emits BCHW images in the SigLIP-normalised ``[-1, 1]`` range at a fixed
224x224 resolution and uses the plain ``Task: {prompt}.`` prompt template.

All prefix tokens use bidirectional attention. The value model's expert head
predicts the value via a [CLS] token appended at the model level, not here.
"""

from typing import ClassVar, Optional

import torch

from ..value_common.image_text_processing import (
    BaseMultiViewImageProcessor,
    BaseValueTextProcessor,
    normalize_image_to_range,
)

IMAGE_RESOLUTION = (224, 224)


def normalize_image_to_model_format(
    img: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Convert any image format to BCHW ``[-1, 1]`` float (value model format)."""
    return normalize_image_to_range(img, "signed", device=device, dtype=dtype)


class ValueImageProcessor(BaseMultiViewImageProcessor):
    """Value model image processor: BCHW ``[-1, 1]`` float at 224x224."""

    output_range: str = "signed"
    DEFAULT_IMAGE_SIZE: ClassVar[tuple[int, int]] = IMAGE_RESOLUTION


class ValueProcessor(BaseValueTextProcessor):
    """Value model processor. Text template: ``Task: {prompt}.``"""

    image_processor_class = "ValueImageProcessor"
    _default_image_processor_cls: ClassVar[Optional[type]] = ValueImageProcessor
    text_template: str = "Task: {prompt}."


__all__ = [
    "ValueImageProcessor",
    "ValueProcessor",
    "normalize_image_to_model_format",
]
