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

"""Convert RLinf ``env_obs`` into Evo-1's inference inputs.

Evo-1 expects, per batch element, an ordered list of RGB views plus an
``image_mask`` marking valid views, a language prompt, and a proprio state.
Its LIBERO recipe uses 3 view slots ``[agentview, wrist, dummy]`` with mask
``[1, 1, 0]`` at 448x448 (see ``Evo-1/scripts/Evo1_server.py`` and
``LIBERO_evaluation/libero_client_4tasks.py``).

Views are emitted as CHW float32 tensors in ``[0, 1]`` (the ``torchvision``
``ToTensor`` convention) rather than PIL images. This matters: Evo-1's embedder
(`InternVL3._preprocess_images`) treats an already-``image_size`` CHW tensor as a
single tile, exactly mirroring the reference server, whereas a PIL image would
trigger dynamic tiling and change the visual-token count.

The RLinf LIBERO env already rotates the agentview/wrist frames 180 degrees to
match training (see ``rlinf/envs/libero/utils.py``), so no extra flip is applied
here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a single image to HxWxC uint8 (RGB)."""
    arr = np.asarray(arr)
    # CHW -> HWC when the leading dim looks like channels.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        # Float images may be in [0, 1] or [0, 255].
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_chw_float(arr_hwc_uint8: np.ndarray, size: int) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0, 1], resized to ``size`` if needed.

    Mirrors ``torchvision.transforms.ToTensor`` applied to a resized RGB frame,
    which is how Evo-1's reference server feeds images to the embedder.
    """
    t = torch.from_numpy(np.ascontiguousarray(arr_hwc_uint8)).permute(2, 0, 1)
    t = t.float().div_(255.0)
    if t.shape[-2] != size or t.shape[-1] != size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return t.contiguous()


def build_evo1_inputs(
    env_obs: dict[str, Any],
    *,
    image_size: int = 448,
    num_view_slots: int = 3,
    view_keys: tuple[str, ...] = ("main_images", "wrist_images", "extra_view_images"),
):
    """Return ``(images, image_mask, prompts, states)`` for a batch.

    Args:
        env_obs: RLinf env observation dict. Recognized keys: ``main_images``
            ([B,H,W,C] uint8 for LIBERO), ``wrist_images``, ``extra_view_images``,
            ``states`` ([B, state_dim]), ``task_descriptions`` (list[str]).
        image_size: target square resolution for every view (Evo-1 LIBERO: 448).
        num_view_slots: fixed number of view slots Evo-1 was trained with.
        view_keys: env_obs keys, in priority order, to fill the view slots.

    Returns:
        images: list (len B) of lists (len num_view_slots) of CHW float32 tensors
            in [0, 1].
        image_mask: LongTensor [B, num_view_slots], 1 for real views else 0.
        prompts: list[str] of length B.
        states: FloatTensor [B, state_dim] or None.
    """
    main = _to_numpy(env_obs["main_images"])
    batch_size = main.shape[0]

    # Collect available view stacks in priority order.
    view_stacks: list[np.ndarray] = []
    for key in view_keys:
        val = env_obs.get(key)
        if val is None:
            continue
        view_stacks.append(_to_numpy(val))

    images: list[list[torch.Tensor]] = []
    masks: list[list[int]] = []
    for i in range(batch_size):
        views: list[torch.Tensor] = []
        for stack in view_stacks:
            sample = stack[i]
            if sample.ndim == 4:  # [N, H, W, C]: multiple sub-views
                for v in range(sample.shape[0]):
                    views.append(_to_chw_float(_to_hwc_uint8(sample[v]), image_size))
            else:
                views.append(_to_chw_float(_to_hwc_uint8(sample), image_size))

        mask = [1] * len(views)
        # Pad with zero (dummy) views up to the fixed slot count.
        while len(views) < num_view_slots:
            views.append(torch.zeros(3, image_size, image_size, dtype=torch.float32))
            mask.append(0)
        views = views[:num_view_slots]
        mask = mask[:num_view_slots]
        images.append(views)
        masks.append(mask)

    image_mask = torch.tensor(masks, dtype=torch.long)

    task_desc = env_obs.get("task_descriptions")
    if task_desc is None:
        prompts = [""] * batch_size
    else:
        prompts = [str(task_desc[i]) for i in range(batch_size)]

    states = env_obs.get("states")
    if states is not None:
        states = states if torch.is_tensor(states) else torch.as_tensor(states)
        states = states.float()

    return images, image_mask, prompts, states
