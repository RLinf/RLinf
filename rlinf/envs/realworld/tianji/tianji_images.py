# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared Stage1/Stage2 RGB resize and padding."""

import numpy as np


def fit_frame(image: np.ndarray, role=None, image_hw=(224, 224)) -> np.ndarray:
    """Preserve aspect ratio and match the Stage1 head downsampling."""
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("camera frame must be RGB uint8 HxWx3")
    source = Image.fromarray(array)
    if role == "head" and source.size == (1280, 720):
        source = source.resize((640, 360), Image.Resampling.BILINEAR)
    h, w = image_hw
    ratio = min(w / source.width, h / source.height)
    nw, nh = max(1, int(source.width * ratio)), max(1, int(source.height * ratio))
    resized = np.asarray(source.resize((nw, nh), Image.Resampling.BILINEAR))
    output = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    output[y : y + nh, x : x + nw] = resized
    return output
