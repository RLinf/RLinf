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
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
]
_STATE_KEYS = ["state.joint_position", "state.gripper_position"]
_ACTION_KEYS = ["action.joint_position", "action.gripper_position"]

_VIDEO_BACKEND = "torchvision"

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a robot "
_MULTIVIEW_LAYOUT = (
    " The video is split into three views: The top view shows the camera view "
    "from the robot's wrist, the bottom-left view shows the camera view from the "
    "left exterior camera, and the bottom-right view shows the camera view from "
    "the right exterior camera. During training, one of the two bottom exterior "
    "views may be a black screen (dropped view). The robot "
)


class OxeDroidDataTransform:
    """Provides modality config and composed transform for oxe_droid."""

    TAG = "oxe_droid"
    DEFAULT_TAG_MAPPING = {"oxe_droid": 17}
    DEFAULT_ACTION_HORIZON = 24
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.exterior_image_1_left"),
            ("extra_view_images", "video.exterior_image_2_left"),
            ("wrist_images", "video.wrist_image_left"),
        ),
        state_fields=(("states", ("state.joint_position", "state.gripper_position")),),
        binarize_gripper=True,
        fill_missing_video_keys=True,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """Build multi-view layout prompt for DROID (matches Groot collate template)."""
        return _TRAINING_PROMPT_PREFIX + instruction + _MULTIVIEW_LAYOUT + instruction

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """2x2 grid: wrist spans top row; exterior views on bottom row."""
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(
                f"oxe_droid expects at least 3 video views, got v={v} with shape {images.shape}"
            )
        left_exterior = images[0]
        right_exterior = images[1]
        wrist_image = images[2]
        concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        wrist_wide = np.repeat(wrist_image, 2, axis=-1)
        concat_images[0, :, :, :h, :] = wrist_wide
        concat_images[0, :, :, h:, :w] = left_exterior
        concat_images[0, :, :, h:, w:] = right_exterior
        return concat_images

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config dict for oxe_droid (25 video delta, 24 action delta)."""
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
                modality_keys=[
                    "annotation.language.language_instruction",
                    "annotation.language.language_instruction_2",
                    "annotation.language.language_instruction_3",
                ],
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
    ) -> ComposedModalityTransform:
        """Build the full ``ComposedModalityTransform`` chain for oxe_droid."""
        return OxeDroidDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", OxeDroidDataTransform.DEFAULT_ACTION_HORIZON)
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
    ) -> ComposedModalityTransform:
        vk = list(_VIDEO_KEYS)
        state_k = list(_STATE_KEYS)
        action_k = list(_ACTION_KEYS)

        transforms: list[Any] = [
            VideoToTensor(apply_to=vk, backend=_VIDEO_BACKEND),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=176,
                width=320,
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
                normalization_modes={
                    "state.joint_position": "q99",
                    "state.gripper_position": "q99",
                },
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={
                    "action.joint_position": "q99",
                    "action.gripper_position": "q99",
                },
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
