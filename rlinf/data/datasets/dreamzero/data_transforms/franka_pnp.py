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

"""DreamZero transforms for Franka pick-and-place (dual-view, state/actions columns)."""

from typing import Any

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

from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform
from rlinf.data.datasets.dreamzero.data_transforms.libero_sim import (
    _ACTION_KEYS,
    _STATE_KEYS,
    _VIDEO_BACKEND,
    _VIDEO_KEYS,
    LiberoSimDataTransform,
)

_DEFAULT_VIDEO_HEIGHT = 176
_DEFAULT_VIDEO_WIDTH = 320
# Match ``8 * max_chunk_size + 1``; use 17 when ``max_chunk_size=2`` (short ~75f episodes).
_NUM_VIDEO_FRAMES = 17


class FrankaPnpDataTransform(LiberoSimDataTransform):
    """Dual-view Franka PnP: same modality keys as ``libero_sim``, Franka training recipe."""

    TAG = "franka_pnp"
    DEFAULT_TAG_MAPPING = {"franka_pnp": 21}
    DEFAULT_ACTION_HORIZON = 24

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config (17 video frames @ max_chunk_size=2, 24-step action)."""
        return {
            "video": ModalityConfig(
                delta_indices=list(range(_NUM_VIDEO_FRAMES)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(_STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(
                    range(FrankaPnpDataTransform.DEFAULT_ACTION_HORIZON)
                ),
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
    ) -> ComposedModalityTransform:
        """Build the full ``ComposedModalityTransform`` chain for ``franka_pnp``."""
        return FrankaPnpDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", FrankaPnpDataTransform.DEFAULT_ACTION_HORIZON)
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
            video_height=int(cfg.get("target_video_height", _DEFAULT_VIDEO_HEIGHT)),
            video_width=int(cfg.get("target_video_width", _DEFAULT_VIDEO_WIDTH)),
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
        video_height: int,
        video_width: int,
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
