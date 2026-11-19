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

"""Lossless transport of single-sample VLM processor outputs."""

import base64
import io
from typing import Any

import numpy as np
import torch

VIDEO_INPUT_KEY = "rlinf_video_inputs_v1"
_VIDEO_FIELDS = {
    "input_ids",
    "attention_mask",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
    "video_second_per_grid",
    "mm_token_type_ids",
}


def encode_video_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    """Encode one unpadded processor sample without changing pixels or tokens."""
    unexpected = set(inputs) - _VIDEO_FIELDS
    if unexpected:
        raise ValueError(f"Unsupported video processor fields: {sorted(unexpected)}")
    arrays = {
        key: value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
        for key, value in inputs.items()
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return {VIDEO_INPUT_KEY: base64.b64encode(stream.getvalue()).decode("ascii")}


def decode_video_inputs(payload: dict[str, str]) -> dict[str, torch.Tensor]:
    """Decode and validate one sample; object arrays and pickle are disallowed."""
    raw = base64.b64decode(payload[VIDEO_INPUT_KEY], validate=True)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) - _VIDEO_FIELDS:
            raise ValueError("Unsupported video processor fields")
        tensors = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    for name in ("input_ids", "pixel_values_videos", "video_grid_thw"):
        if name not in tensors:
            raise ValueError(f"Missing video processor field: {name}")
    ids = tensors["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64:
        raise ValueError("Video requests require one int64 input_ids sequence")
    mask = tensors.get("attention_mask")
    if mask is not None and (mask.shape != ids.shape or not bool(mask.all())):
        raise ValueError("Video requests must contain one unpadded sample")
    grid = tensors["video_grid_thw"]
    pixels = tensors["pixel_values_videos"]
    if (
        grid.ndim != 2
        or grid.shape[1] != 3
        or grid.dtype not in (torch.int32, torch.int64)
        or not bool((grid > 0).all())
    ):
        raise ValueError("Invalid video_grid_thw")
    if (
        pixels.ndim != 2
        or not pixels.is_floating_point()
        or int(grid.prod(dim=-1).sum()) != pixels.shape[0]
    ):
        raise ValueError("Video pixels do not match video_grid_thw")
    return tensors
