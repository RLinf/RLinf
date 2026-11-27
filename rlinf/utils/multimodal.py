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

"""Aligned multimodal preprocessing, in-memory image cache, and batch collation."""

import hashlib
import inspect
import io
import math
import os
from collections.abc import Sequence
from typing import Any

import torch

_DEFAULT_IMAGE_CACHE_MAX_ITEMS = 256


def _smart_resize_dims(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Match Qwen-VL / SGLang ``smart_resize`` dimension rounding."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Qwen-VL images must have positive dimensions and aspect ratio <= 200."
        )
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = math.floor(height / scale / factor) * factor
        w = math.floor(width / scale / factor) * factor
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    if min(h, w) <= 0:
        raise ValueError(
            "Qwen-VL pixel budget is too small for this image's aspect ratio."
        )
    return h, w


def _read_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("true", "1", "yes", "y"):
        return True
    if val in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}.")


def qwen_image_processor_policy(
    image_processor: Any,
) -> tuple[int, int, int, bool, Any]:
    """Return ``(factor, min_pixels, max_pixels, limit_pixels, resample)`` matching SGLang."""
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size
    size = getattr(image_processor, "size", None) or {}
    min_pixels = int(
        size["shortest_edge"]
        if "shortest_edge" in size
        else getattr(image_processor, "min_pixels", 56 * 56)
    )
    max_pixels = int(
        size["longest_edge"]
        if "longest_edge" in size
        else getattr(image_processor, "max_pixels", 28 * 28 * 1280)
    )
    compress_images = _read_env_bool("SGLANG_VLM_QWEN_LIMIT_PIXEL", default=False)
    resample = None
    if compress_images:
        from PIL import Image as PILImage

        if max_pixels >= (32 * factor) ** 2:
            max_pixels //= 16
        resample_name = os.environ.get("SGLANG_RESIZE_RESAMPLE", "")
        resample = getattr(PILImage, resample_name, None) if resample_name else None
    if factor <= 0 or not 0 < min_pixels <= max_pixels:
        raise ValueError(
            f"Invalid Qwen-VL image policy: factor={factor}, min={min_pixels}, max={max_pixels}."
        )
    return factor, min_pixels, max_pixels, compress_images, resample


def process_single_image(
    image_bytes: bytes,
    processor: Any,
    *,
    dtype: torch.dtype | None = None,
    cache: dict[tuple[str, torch.dtype | None], tuple[torch.Tensor, torch.Tensor, int]]
    | None = None,
    cache_max_items: int = _DEFAULT_IMAGE_CACHE_MAX_ITEMS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Preprocess one raw image into ``(pixel_values, image_grid_thw, num_tokens)``."""
    digest = hashlib.sha256(image_bytes).hexdigest() if cache is not None else ""
    cache_key = (digest, dtype)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    from PIL import Image

    image_processor = getattr(processor, "image_processor", processor)
    factor, min_pixels, max_pixels, compress_images, resample = (
        qwen_image_processor_policy(image_processor)
    )
    with Image.open(io.BytesIO(image_bytes)) as opened:
        rgb = opened.convert("RGB")
    try:
        if compress_images:
            height, width = _smart_resize_dims(
                rgb.height, rgb.width, factor, min_pixels, max_pixels
            )
            rgb = rgb.resize((width, height), resample=resample)
        kwargs: dict[str, Any] = {
            "images": [rgb],
            "return_tensors": "pt",
            "size": {"shortest_edge": min_pixels, "longest_edge": max_pixels},
        }
        preprocess_fn = getattr(image_processor, "preprocess", image_processor)
        if "device" in inspect.signature(preprocess_fn).parameters:
            kwargs["device"] = "cpu"
        features = image_processor(**kwargs)
    finally:
        rgb.close()

    pixel_values = features["pixel_values"]
    if not torch.is_tensor(pixel_values):
        pixel_values = torch.as_tensor(pixel_values)
    if dtype is not None and pixel_values.dtype != dtype:
        pixel_values = pixel_values.to(dtype)
    image_grid_thw = features["image_grid_thw"]
    if not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.long)
    image_grid_thw = image_grid_thw.reshape(-1, 3).to(dtype=torch.long)

    merge_size = int(getattr(image_processor, "merge_size", 2))
    num_tokens = int(image_grid_thw.prod(dim=1).sum().item()) // (merge_size**2)
    result = (pixel_values.contiguous(), image_grid_thw.contiguous(), num_tokens)
    if cache is not None and cache_max_items > 0:
        while len(cache) >= cache_max_items:
            cache.pop(next(iter(cache)))
        cache[cache_key] = result
    return result


def build_mm_token_type_ids(
    seq_len: int, image_spans: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Build 1D uint8 ``mm_token_type_ids`` (0=text, 1=image) for real spans."""
    token_types = torch.zeros(seq_len, dtype=torch.uint8)
    for start, end in image_spans:
        if not (0 < start < end < seq_len):
            raise ValueError(
                f"Image span [{start}, {end}) must lie strictly inside sequence length {seq_len}."
            )
        token_types[start:end] = 1
    return token_types


def _normalize_tensor_list(
    value: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    return list(value)


def collate_fsdp_multimodal_inputs(
    payloads: list[dict | None],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Collate non-empty per-row multimodal tensors for FSDP forward passes."""
    non_empty = [inputs for inputs in payloads if inputs]
    if not non_empty:
        return {}
    if any("mm_token_type_ids" in inputs for inputs in non_empty):
        raise ValueError(
            "Aligned multi-turn multimodal inputs with mm_token_type_ids require "
            "Megatron-Bridge, not FSDP."
        )
    collated = {}
    for key in non_empty[0]:
        parts: list[torch.Tensor] = []
        for inputs in non_empty:
            parts.extend(_normalize_tensor_list(inputs[key]))
        collated[key] = torch.cat(parts, dim=0).to(device)
    return collated


def materialize_multimodal_inputs(
    payloads: list[dict | None],
    device: str | torch.device,
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    aligned_mode: bool = False,
) -> dict[str, torch.Tensor]:
    """Build one microbatch's pixels and optional ``mm_token_type_ids`` in row order."""
    if aligned_mode and (input_ids is None or attention_mask is None):
        raise ValueError(
            "aligned_mode=True requires both input_ids and attention_mask."
        )
    pixels: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    token_types = (
        torch.zeros(input_ids.shape, dtype=torch.uint8) if aligned_mode else None
    )
    if aligned_mode and not payloads:
        payloads = [None] * input_ids.shape[0]
    if aligned_mode and len(payloads) != input_ids.shape[0]:
        raise ValueError(
            "Aligned multi_modal_inputs must align with the training batch: "
            f"expected {input_ids.shape[0]} rows, got {len(payloads)}."
        )

    for row, entry in enumerate(payloads):
        if aligned_mode:
            if not entry:
                continue
            if "mm_token_type_ids" not in entry:
                raise ValueError(
                    "Aligned multimodal training rows must include mm_token_type_ids."
                )
            row_mask = attention_mask[row].bool().cpu()
            valid_len = int(row_mask.sum().item())
            row_types = torch.as_tensor(
                entry["mm_token_type_ids"], dtype=torch.uint8
            ).cpu()
            if row_types.ndim != 1 or len(row_types) > valid_len:
                raise ValueError(
                    f"Row {row} mm_token_type_ids length ({len(row_types)}) exceeds "
                    f"active sequence length ({valid_len})."
                )
            if len(row_types) < valid_len:
                padded_types = torch.zeros(valid_len, dtype=torch.uint8)
                padded_types[: len(row_types)] = row_types
                row_types = padded_types
            token_types[row, row_mask] = row_types
        elif entry and "mm_token_type_ids" in entry:
            raise ValueError(
                "Aligned multimodal rows with mm_token_type_ids require "
                "rollout.sglang.align_multimodal_inputs=True."
            )
        if not entry:
            continue
        pixels.extend(_normalize_tensor_list(entry["pixel_values"]))
        grids.extend(_normalize_tensor_list(entry["image_grid_thw"]))

    result: dict[str, torch.Tensor] = (
        {"mm_token_type_ids": token_types.to(device)} if token_types is not None else {}
    )
    if not pixels:
        return result
    return {
        **result,
        "pixel_values": torch.cat(pixels, dim=0).to(device),
        "image_grid_thw": torch.cat(grids, dim=0).to(device),
    }


def dense_token_inputs(batch: dict) -> dict:
    """Pad ragged microbatch token inputs before Bridge embeds its inputs."""
    if torch.is_tensor(batch["input_ids"]):
        return batch
    from torch.nn.utils.rnn import pad_sequence

    tokens = pad_sequence(batch["input_ids"], batch_first=True)
    if "attention_mask" in batch:
        mask = pad_sequence(batch["attention_mask"], batch_first=True)
    else:
        lengths = torch.tensor(
            [len(row) for row in batch["input_ids"]], device=tokens.device
        )
        mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            < lengths[:, None]
        )
    return {
        **batch,
        "input_ids": tokens,
        "attention_mask": mask,
    }
