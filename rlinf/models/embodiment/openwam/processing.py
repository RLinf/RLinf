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

"""Observation images, prompts, and robot states for OpenWAM inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def _infer_batch_size(obs: dict[str, Any]) -> int:
    for key in ("states", "state", "proprio", "main_images", "image", "images"):
        if key in obs:
            value = obs[key]
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return int(value.shape[0]) if value.ndim > 1 else 1
            if isinstance(value, (list, tuple)):
                return len(value)
    return 1


def _batch_value(
    obs: dict[str, Any], keys: tuple[str, ...], index: int, default: Any = None
) -> Any:
    value = next((obs[key] for key in keys if key in obs), default)
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 1:
        return value[index]
    if isinstance(value, (list, tuple)) and value and len(value) > index:
        return value[index]
    return value


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        array = np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype(
            np.uint8
        )
    return Image.fromarray(array)


def _axis_angle_to_rotation_6d(value: Any) -> np.ndarray:
    """Convert a LIBERO axis-angle state to OpenWAM's 6-D representation."""
    from scipy.spatial.transform import Rotation

    from rlinf.utils.rot6d import matrix_to_rot6d

    matrix = Rotation.from_rotvec(
        np.asarray(value, dtype=np.float64).reshape(3)
    ).as_matrix()
    return matrix_to_rot6d(matrix)


def _libero_state_to_eef10(value: Any) -> np.ndarray | None:
    """Convert RLinf's 8-D LIBERO state to OpenWAM's achieved EEF10 state."""
    if value is None:
        return None
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape[0] == 10:
        return state
    if state.shape[0] != 8:
        raise ValueError(
            "OpenWAM LIBERO rollout expects an 8-D state "
            f"(xyz, axis-angle, gripper qpos), got {state.shape[0]}"
        )
    gripper_width = float(state[6] - state[7])
    gripper_open_scale = np.clip(2.0 * gripper_width / 0.08 - 1.0, -1.0, 1.0)
    return np.concatenate(
        [state[:3], _axis_angle_to_rotation_6d(state[3:6]), [gripper_open_scale]]
    ).astype(np.float32)


ROBOTWIN_PROMPT_PREFIX = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: "
)
_TEMPLATED_DATASET_TYPES = ("robotwin", "robodojo", "ebench")


def _prompt_template_for_dataset(dataset_type: Any) -> str | None:
    """Return the training-time instruction prefix of an OpenWAM dataset reader."""
    if dataset_type is None or str(dataset_type) not in _TEMPLATED_DATASET_TYPES:
        return None
    try:
        from openwam.dataloader.transforms.multiview import (
            format_prompt_for_inference,
        )

        return format_prompt_for_inference("")
    except ImportError:
        return ROBOTWIN_PROMPT_PREFIX


def _wrist_frames(value: Any) -> list[Image.Image]:
    """Split a wrist observation into per-camera PIL frames, in layout order.

    RLinf environments hand back either one wrist image (``[H, W, 3]``) or a
    stack of them (``[n, H, W, 3]``; RoboTwin stacks left then right).
    """
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        return [_to_pil(item) for item in value if item is not None]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        return [_to_pil(frame) for frame in array]
    return [_to_pil(array)]


# ``assemble_multiview_layout`` default: the head camera takes the top two
# thirds of the canvas, the wrist cameras split the bottom third.
_MULTIVIEW_TOP_HEIGHT_RATIO = 2.0 / 3.0


def _compose_observation_image(
    main_image: Image.Image,
    wrist_image: Any,
    *,
    multiview: bool,
    camera_layout: list[str],
    height: int,
    width: int,
) -> Image.Image:
    """Match OpenWAM's single-view or three-camera training layout.

    Single-view checkpoints see the head camera resized to the training
    canvas without cropping, as the native dataset readers do. Multiview
    checkpoints get the head camera in slot 0 and the wrist cameras in the
    following slots of ``camera_layout``; missing cameras stay black, exactly
    like the dataset reader pads them.
    """
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    if not multiview:
        if main_image.size == (width, height):
            return main_image
        # Native single-view readers decode directly to (width, height); do not
        # center-crop a square environment frame and discard its vertical view.
        return main_image.resize((width, height), Image.Resampling.LANCZOS)

    # The training readers decode the head camera at the top-slot size and the
    # wrist cameras at the bottom-slot size with LANCZOS before composition, so
    # the BILINEAR stretch inside ``assemble_multiview_layout`` is a no-op there.
    # Pre-resizing the environment frames the same way keeps eval pixels on the
    # training distribution instead of bilinear-stretching raw frames.
    top_h = int(round(height * _MULTIVIEW_TOP_HEIGHT_RATIO))
    head_size = (width, top_h)
    wrist_size = (width // 2, height - top_h)
    frames = {camera_layout[0]: _resize_for_slot(main_image, head_size)}
    for slot, frame in zip(camera_layout[1:], _wrist_frames(wrist_image)):
        frames[slot] = _resize_for_slot(frame, wrist_size)
    return assemble_multiview_layout(
        frames,
        camera_layout=camera_layout,
        out_h=height,
        out_w=width,
        top_height_ratio=_MULTIVIEW_TOP_HEIGHT_RATIO,
    )


def _resize_for_slot(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """LANCZOS-resize ``image`` to ``size`` ``(width, height)`` like the readers."""
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _observation_proprio(env_obs: dict[str, Any], index: int) -> np.ndarray | None:
    """Use explicit native physical units, or convert LIBERO's achieved pose."""
    value = _batch_value(env_obs, ("native_proprio",), index, None)
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(
                "native_proprio must be a finite vector in checkpoint physical units"
            )
        return value
    return _libero_state_to_eef10(
        _batch_value(env_obs, ("states", "state", "proprio"), index, None)
    )
