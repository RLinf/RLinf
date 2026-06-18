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

from typing import Any

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform

_VIDEO_KEYS = [
    "video.image",
    "video.wrist_image",
]
_STATE_KEYS = ["state.state"]
_ACTION_KEYS = ["action.actions"]

_VIDEO_BACKEND = "torchvision"

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a robot "
_MULTIVIEW_LAYOUT = (
    " The video is split into two horizontal views: the left view shows the "
    "exterior camera and the right view shows the wrist camera. The robot "
)


class LiberoSimDataTransform:
    """Provides modality config and composed transform for libero_sim."""

    TAG = "libero_sim"
    DEFAULT_TAG_MAPPING = {"libero_sim": 21}
    DEFAULT_ACTION_HORIZON = 16
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.image"),
            ("wrist_images", "video.wrist_image"),
        ),
        state_fields=(("states", "state.state"),),
        binarize_gripper=True,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """Build multi-view layout prompt for LIBERO (matches Groot collate template)."""
        return _TRAINING_PROMPT_PREFIX + instruction + _MULTIVIEW_LAYOUT + instruction

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """Horizontal concat: exterior (left) | wrist (right)."""
        v, t, c, h, w = images.shape
        if v < 2:
            raise ValueError(
                f"libero_sim expects at least 2 video views, got v={v} with shape {images.shape}"
            )
        concat_images = np.zeros((1, t, c, h, 2 * w), dtype=images.dtype)
        concat_images[0, :, :, :, :w] = images[0]
        concat_images[0, :, :, :, w:] = images[1]
        return concat_images

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config dict for libero_sim (25 video delta, 24 action delta)."""
        return {
            "video": ModalityConfig(
                delta_indices=list(range(25)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(_STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(24)),
                modality_keys=list(_ACTION_KEYS),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
            "lapa_action": ModalityConfig(
                delta_indices=[0],
                modality_keys=["lapa_action"],
            ),
        }

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
        transform_on_gpu: bool = False,
    ) -> ComposedModalityTransform:
        """Build the full ``ComposedModalityTransform`` chain for libero_sim."""
        return LiberoSimDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", LiberoSimDataTransform.DEFAULT_ACTION_HORIZON)
            ),
            max_state_dim=int(cfg.get("max_state_dim", 64)),
            max_action_dim=int(cfg.get("max_action_dim", 32)),
            max_length=int(cfg.get("max_seq_len", 512)),
            default_instruction=str(
                cfg.get("default_instruction", "Perform the default behavior.")
            ),
            language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
            always_use_default_instruction=bool(
                cfg.get("always_use_default_instruction", False)
            ),
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
            transform_on_gpu=transform_on_gpu,
        )

    @staticmethod
    def _build_composed_transform(
        tokenizer_path: str,
        state_horizon: int,
        action_horizon: int,
        max_state_dim: int,
        max_action_dim: int,
        max_length: int,
        default_instruction: str,
        language_dropout_prob: float,
        always_use_default_instruction: bool,
        embodiment_tag_mapping: dict[str, int],
        video_height: int = 256,
        video_width: int = 256,
        transform_on_gpu: bool = False,
    ) -> ComposedModalityTransform:
        vk = list(_VIDEO_KEYS)
        state_k = list(_STATE_KEYS)
        action_k = list(_ACTION_KEYS)

        transforms: list[Any] = [
            VideoToTensor(
                apply_to=vk, backend=_VIDEO_BACKEND, output_on_cuda=transform_on_gpu
            ),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=video_height,
                width=video_width,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=vk, backend=_VIDEO_BACKEND),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={"state.state": "q99"},
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={"action.actions": "q99"},
            ),
            ConcatTransform(
                apply_to=[],
                video_concat_order=vk,
                state_concat_order=state_k,
                action_concat_order=action_k,
            ),
            DreamTransform(
                default_instruction=default_instruction,
                language_dropout_prob=language_dropout_prob,
                always_use_default_instruction=always_use_default_instruction,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                max_length=max_length,
                state_horizon=state_horizon,
                action_horizon=action_horizon,
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=embodiment_tag_mapping,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
