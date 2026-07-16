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

"""RLinf :class:`DreamTransform` subclass with embodiment-specific multi-view concat."""

from __future__ import annotations

from typing import Any

import numpy as np
from einops import rearrange
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)


def resolve_registry_tag(embodiment_tag: Any) -> str:
    """Map Groot ``EmbodimentTag`` or registry tag string to ``_EMBODIMENT_REGISTRY`` key."""
    if hasattr(embodiment_tag, "value"):
        return str(embodiment_tag.value)
    return str(embodiment_tag)


def concat_generic_grid_views(images: np.ndarray) -> np.ndarray:
    """Fallback 2x2 grid for unknown embodiments."""
    v, t, c, h, w = images.shape
    concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
    if v > 0:
        concat_images[0, :, :, :h, :w] = images[0]
    if v > 1:
        concat_images[0, :, :, h:, :w] = images[1]
    if v > 2:
        concat_images[0, :, :, :h, w:] = images[2]
    return concat_images


def concat_multiview_video(embodiment_tag: Any, images: Any) -> np.ndarray:
    """Concat multi-view frames using the embodiment registered in ``_EMBODIMENT_REGISTRY``."""
    from rlinf.data.datasets.dreamzero.data_transforms import (
        _EMBODIMENT_REGISTRY,
        _require_embodiment,
    )

    arr = np.asarray(images)
    tag = resolve_registry_tag(embodiment_tag)
    if tag in _EMBODIMENT_REGISTRY:
        return _require_embodiment(tag).concat_multiview_video(arr)
    return concat_generic_grid_views(arr)


class DreamTransform(DreamTransformBase):
    """DreamTransform that delegates multi-view layout to ``data_transforms`` registry."""

    def apply_single(self, data: dict) -> dict:
        """Apply transform for one sample.

        Groot's ``apply_single`` compares against ``EmbodimentTag`` members that are not
        part of RLinf's patched enum (e.g. ``GR1_UNIFIED_SEGMENTATION``). RLinf only
        registers standard manipulation datasets, so skip those branches here.
        """
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language, is_lapa_instance, is_dream_instance, is_cotrain_instance = (
            self._prepare_language(data)
        )
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

            # default for lapa instance
            transformed_data["lapa_action"] = np.zeros_like(transformed_data["action"])
            transformed_data["lapa_action_mask"] = np.zeros_like(
                transformed_data["action_mask"]
            )

        transformed_data["text_negative"] = (
            "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
            "style, artwork, painting, image, still, grayscale, dull, worst quality, "
            "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
            "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
            "image, cluttered background, three legs, many people in the background, "
            "walking backwards."
        )

        for key, value in vlm_outputs.items():
            assert key not in transformed_data, (
                f"Key {key} already exists in transformed_data."
            )
            transformed_data[key] = value

        transformed_data["embodiment_id"] = self.get_embodiment_tag()
        transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
        transformed_data["is_cotrain_instance"] = np.array(
            is_cotrain_instance, dtype=bool
        )

        if is_dream_instance:
            assert "dream_actions" in data
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["dream"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
            dream_actions = data["dream_actions"]
            assert dream_actions.size == actions_shape[0] * actions_shape[1], (
                f"dream_actions size {dream_actions.size} does not match action shape "
                f"{actions_shape}"
            )
            transformed_data["action"] = dream_actions.reshape(actions_shape)

        if is_lapa_instance:
            assert "lapa_action" in data
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["lapa"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape
            lapa_actions = data["lapa_action"]
            assert lapa_actions.size == actions_shape[0] * actions_shape[1], (
                f"Cannot reshape lapa_actions of size {lapa_actions.size} to "
                f"{actions_shape}"
            )
            reshaped_lapa_actions = lapa_actions.reshape(actions_shape)
            assert np.all(reshaped_lapa_actions >= -1) and np.all(
                reshaped_lapa_actions <= 1
            ), "LAPA action values should be between -1 and 1"
            transformed_data["action"] = reshaped_lapa_actions
            transformed_data["action_mask"] = np.ones(actions_shape, dtype=bool)

        if self.training:
            action_and_mask_keys = [
                "action",
                "action_mask",
                "lapa_action",
                "lapa_action_mask",
            ]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), (
                "Shape mismatch: "
                f"{[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"
            )

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        """Collate with RLinf prompt wrapping (supports all registered embodiments)."""
        import tree

        from rlinf.data.datasets.dreamzero.dataloader import DreamZeroCollator

        data.pop("lapa_action", None)
        data.pop("dream_actions", None)
        data_split = [
            tree.map_structure(lambda x: x[i], data) for i in range(batch_size)
        ]
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return DreamZeroCollator.collate_batch(
            data_split_processed,
            self.tokenizer,
            self.embodiment_tag_mapping,
        )

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            return concat_multiview_video(self.embodiment_tag, images)
        return images
