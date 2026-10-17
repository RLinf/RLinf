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

"""NPU-safe patch packing for Qwen-VL fast image processors.

Transformers (tracked at 4.57.x) packs Qwen-VL patches through a 10-dim
view/permute/reshape chain. The view and permute are metadata, but the
reshape after the permute copies a non-contiguous tensor with more than 8
dims, and CANN's copy kernel rejects it (``AclNN_Parameter_Error EZ1001``).
The replacement here performs the same rearrangement in <=8-dim stages and
installs itself on ``Qwen2VLImageProcessorFast._preprocess`` (the base class
the Qwen2.5-VL and Qwen3-VL fast processors reuse), so device tensors stay on
the NPU through the whole image leg. Values are identical to the upstream
chain; only the op decomposition changes.
"""

from __future__ import annotations

from typing import Any

import torch


def _is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _patchify_qwen_vl(
    patches: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Pack ``(B, T, C, H, W)`` into ``(B, grid_t*grid_h*grid_w, C*t_p*p*p)``.

    Stage-wise equivalent of the upstream 10-dim view/permute/reshape with
    every copy at 8 dims or fewer. Requires ``H`` and ``W`` to be multiples
    of ``patch_size * merge_size`` and ``T`` to be a multiple of
    ``temporal_patch_size`` — the same constraints the upstream view enforces.
    """
    batch, frames, channel, height, width = patches.shape
    grid_t = frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    # (B, T, C, H, W) -> (B, T, grid_h, grid_w, C, p, p)
    x = (
        patches.view(batch, frames, channel, grid_h, patch_size, grid_w, patch_size)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
    )
    # -> (B, grid_t, grid_h, grid_w, C, t_p, p, p)
    x = (
        x.view(
            batch,
            grid_t,
            temporal_patch_size,
            grid_h,
            grid_w,
            channel,
            patch_size,
            patch_size,
        )
        .permute(0, 1, 3, 4, 5, 2, 6, 7)
        .contiguous()
    )
    # -> (B, grid_t, grid_h//m, grid_w//m, m, m, C*t_p*p*p)
    x = (
        x.view(
            batch,
            grid_t,
            grid_h // merge_size,
            merge_size,
            grid_w // merge_size,
            merge_size,
            -1,
        )
        .permute(0, 1, 2, 4, 3, 5, 6)
        .contiguous()
    )
    return x.view(batch, grid_t * grid_h * grid_w, -1)


def _qwen_vl_fast_preprocess_npu(
    self: Any,
    images: list[torch.Tensor],
    do_resize: bool,
    size: Any,
    interpolation: Any | None,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float] | None,
    image_std: float | list[float] | None,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    disable_grouping: bool | None,
    return_tensors: str | Any | None,
    **kwargs: Any,
) -> Any:
    """``Qwen2VLImageProcessorFast._preprocess`` with staged patch packing.

    Mirrors transformers 4.57.x down to the patch packing, which is replaced
    by :func:`_patchify_qwen_vl`. The upstream file is the source of truth;
    re-sync this body when the pinned venvs move to a new transformers.
    """
    from transformers.image_processing_utils import BatchFeature
    from transformers.image_processing_utils_fast import (
        SizeDict,
        group_images_by_shape,
        reorder_images,
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    # Group images by size for batched resizing
    grouped_images, grouped_images_index = group_images_by_shape(
        images, disable_grouping=disable_grouping
    )
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        height, width = stacked_images.shape[-2:]
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            stacked_images = self.resize(
                image=stacked_images,
                size=SizeDict(height=resized_height, width=resized_width),
                interpolation=interpolation,
            )
        resized_images_grouped[shape] = stacked_images
    resized_images = reorder_images(resized_images_grouped, grouped_images_index)

    # Group images by size for further processing
    # Needed in case do_resize is False, or resize returns images with different sizes
    grouped_images, grouped_images_index = group_images_by_shape(
        resized_images, disable_grouping=disable_grouping
    )
    processed_images_grouped = {}
    processed_grids = {}
    for shape, stacked_images in grouped_images.items():
        resized_height, resized_width = stacked_images.shape[-2:]
        # Fused rescale and normalize
        patches = self.rescale_and_normalize(
            stacked_images,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
        )
        if patches.ndim == 4:
            # add a temporal dimension if we have images
            patches = patches.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)
        batch_size, grid_t, _ = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        flatten_patches = _patchify_qwen_vl(
            patches, patch_size, temporal_patch_size, merge_size
        )

        processed_images_grouped[shape] = flatten_patches
        processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

    processed_images = reorder_images(processed_images_grouped, grouped_images_index)
    processed_grids = reorder_images(processed_grids, grouped_images_index)
    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids)

    return BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        tensor_type=return_tensors,
    )


_PATCH_INSTALLED = False


def ensure_qwen_vl_npu_patchify_patch() -> bool:
    """Install the staged patch packing on Qwen-VL fast image processors.

    No-op off Ascend NPU or when already installed. Returns whether the
    replacement is active; callers should keep their host fallback when it
    reports ``False``.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED or not _is_npu():
        return _PATCH_INSTALLED
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
            Qwen2VLImageProcessorFast,
        )
    except ImportError:
        return False
    Qwen2VLImageProcessorFast._preprocess = _qwen_vl_fast_preprocess_npu
    _PATCH_INSTALLED = True
    return True
