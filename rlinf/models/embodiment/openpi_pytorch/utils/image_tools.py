# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained PIL-based image resize-with-pad (vendored from openpi_client).

The old BEHAVIOR eval path resizes camera images to 224x224 with this exact
PIL-bilinear, aspect-preserving, zero-padded routine before the model. Vendoring
it keeps image preprocessing byte-identical to the old path while removing the
``openpi_client`` dependency.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Resize a batch of ``[..., H, W, C]`` images to ``height x width`` with padding.

    Validates the external contract: ``images`` must be a ``numpy`` array of rank
    >= 3 with a PIL-convertible trailing channel axis (1, 3, or 4), and the target
    ``height``/``width`` must be positive.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError(
            f"resize_with_pad expects a numpy array, got {type(images).__name__}."
        )
    if images.ndim < 3:
        raise ValueError(
            f"resize_with_pad expects images of rank >= 3 ([..., H, W, C]); "
            f"got shape {images.shape}."
        )
    if (
        not (isinstance(height, int) and isinstance(width, int))
        or height <= 0
        or width <= 0
    ):
        raise ValueError(
            f"resize_with_pad target (height, width) must be positive ints; "
            f"got ({height!r}, {width!r})."
        )
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"resize_with_pad expects 1/3/4 image channels in the last axis; "
            f"got shape {images.shape}."
        )
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> np.ndarray:
    """Resize one PIL image to ``height x width`` without distortion (zero-padded)."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.asarray(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.asarray(zero_image)
