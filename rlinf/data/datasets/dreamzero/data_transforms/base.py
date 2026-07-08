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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform


@dataclass(frozen=True)
class RolloutObsLayout:
    """Maps standard RLinf rollout ``env_obs`` fields to model modality keys."""

    video_fields: tuple[tuple[str, str], ...]
    state_fields: tuple[tuple[str, str | tuple[str, ...]], ...]
    language_env_key: str = "task_descriptions"
    binarize_gripper: bool = False
    fill_missing_video_keys: bool = False


@runtime_checkable
class DreamZeroEmbodimentTransform(Protocol):
    """Static interface implemented by each embodiment module."""

    TAG: str
    DEFAULT_TAG_MAPPING: dict[str, int]
    DEFAULT_ACTION_HORIZON: int
    ROLLOUT_OBS_LAYOUT: RolloutObsLayout

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]: ...

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
        transform_on_gpu: bool = False,
    ) -> ComposedModalityTransform: ...

    @staticmethod
    def format_training_prompt(instruction: str) -> str: ...

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """Concat multi-view frames ``(v, t, c, h, w)`` to ``(1, t, c, H, W)``."""
        ...


def _rollout_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rollout_to_bthwc_uint8(images: Any, batch_size: int) -> np.ndarray:
    """Normalize images to ``[B, T, H, W, C]`` uint8 (``T=1`` for single-frame rollout)."""
    arr = _rollout_to_numpy(images)
    if arr.ndim == 5:
        if arr.shape[2] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 1, 3, 4, 2))
    elif arr.ndim == 4:
        if arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 2, 3, 1))
        arr = arr[:, None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected 3D–5D array.")

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    if arr.shape[0] != batch_size:
        raise ValueError(
            f"Image batch size {arr.shape[0]} does not match env batch {batch_size}."
        )
    return arr


def _rollout_split_droid_state(
    states: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split flat ``states`` into joint (7) and gripper (1) components."""
    s = states.astype(np.float32, copy=False)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim > 2:
        s = s.reshape(batch_size, -1)
    if s.shape[-1] < 8:
        joint = np.zeros((batch_size, 7), dtype=np.float32)
        gripper = np.zeros((batch_size, 1), dtype=np.float32)
        joint[:, : s.shape[-1]] = s
        return joint[:, None, :], gripper[:, None, :]
    joint = s[..., :7]
    gripper = s[..., 7:8]
    return joint[:, None, :], gripper[:, None, :]


def _rollout_assign_video(
    out: dict[str, Any],
    env_obs: dict[str, Any],
    env_key: str,
    model_key: str,
    batch_size: int,
) -> None:
    if env_key not in env_obs or env_obs[env_key] is None:
        return
    raw = env_obs[env_key]
    if env_key == "extra_view_images":
        arr = _rollout_to_numpy(raw)
        if arr.ndim == 5:
            arr = arr[:, 0]
        out[model_key] = _rollout_to_bthwc_uint8(arr, batch_size)
        return
    out[model_key] = _rollout_to_bthwc_uint8(raw, batch_size)


def convert_rollout_env_obs_with_layout(
    env_obs: dict[str, Any],
    layout: RolloutObsLayout,
    language_model_key: str,
) -> dict[str, Any]:
    """Convert a rollout ``env_obs`` batch using a :class:`RolloutObsLayout`."""
    if "main_images" not in env_obs:
        raise KeyError("env_obs must contain 'main_images'.")

    batch_size = int(_rollout_to_numpy(env_obs["main_images"]).shape[0])
    converted: dict[str, Any] = {}

    for env_key, model_key in layout.video_fields:
        _rollout_assign_video(converted, env_obs, env_key, model_key, batch_size)

    for env_key, model_targets in layout.state_fields:
        if env_key not in env_obs or env_obs[env_key] is None:
            continue
        states = _rollout_to_numpy(env_obs[env_key]).astype(np.float32)
        if isinstance(model_targets, str):
            if states.ndim == 1:
                states = states[None, :]
            elif states.ndim > 2:
                states = states.reshape(batch_size, -1)
            converted[model_targets] = states[:, None, :]
        else:
            joint_key, gripper_key = model_targets
            joint, gripper = _rollout_split_droid_state(states, batch_size)
            converted[joint_key] = joint
            converted[gripper_key] = gripper

    prompts = env_obs.get(layout.language_env_key)
    if prompts is None:
        prompts = [""] * batch_size
    if isinstance(prompts, str):
        prompts = [prompts] * batch_size
    converted[language_model_key] = list(prompts)

    if layout.fill_missing_video_keys:
        ref_key = next((k for k in converted if k.startswith("video.")), None)
        if ref_key is not None:
            blank = np.zeros_like(converted[ref_key])
            for _, model_key in layout.video_fields:
                converted.setdefault(model_key, blank)

    return converted
