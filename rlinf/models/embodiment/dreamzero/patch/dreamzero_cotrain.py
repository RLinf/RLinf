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

"""Libero_sim: multi-view video concat + collate text templates for DreamTransform."""

import ast

import numpy as np
import torch
from einops import rearrange
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)


def collate(
    features: list[dict],
    tokenizer,
    num_views=3,
    embodiment_tag_mapping=None,
) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                item = elem[key]
                try:
                    parsed_item = ast.literal_eval(item)
                    # Handle different return types from ast.literal_eval
                    if isinstance(parsed_item, (list, tuple)):
                        processed_item = str(parsed_item[0])
                    else:
                        # If it's already a scalar (string, float, int, etc.), convert to string
                        processed_item = str(parsed_item)

                    if (
                        elem["embodiment_id"]
                        == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]
                    ):
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + processed_item.lower()
                        )
                    elif (
                        elem["embodiment_id"]
                        == embodiment_tag_mapping[EmbodimentTag.LIBERO_SIM.value]
                    ):
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into two horizontal views: the left view shows the exterior camera and the right view shows the wrist camera. The robot "
                            + processed_item.lower()
                        )
                    else:
                        raise ValueError(
                            f"Embodiment ID {elem['embodiment_id']} not supported."
                        )
                    output_values.append(processed_item)
                except (ValueError, SyntaxError, TypeError):
                    # If parsing fails or item is already a string, use it directly
                    if (
                        elem["embodiment_id"]
                        == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]
                    ):
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + str(item).lower()
                        )
                    elif (
                        elem["embodiment_id"]
                        == embodiment_tag_mapping[EmbodimentTag.LIBERO_SIM.value]
                    ):
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into two horizontal views: the left view shows the exterior camera and the right view shows the wrist camera. The robot "
                            + str(item).lower()
                        )
                    else:
                        raise ValueError(
                            f"Embodiment ID {elem['embodiment_id']} not supported."
                        )
                    output_values.append(item)
            ids, mask = tokenizer(
                output_values, return_mask=True, add_special_tokens=True
            )
            batch[key] = ids
            batch["text_attention_mask"] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask_negative"] = mask
        else:
            values = [elem[key] for elem in features]
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except ValueError as e:
                shapes = [np.asarray(v).shape for v in values]
                raise ValueError(
                    f"Shape mismatch in collate for key='{key}': shapes={shapes}"
                ) from e
    return batch


class DreamTransform(DreamTransformBase):
    """Adds LIBERO_SIM horizontal two-view concat (exterior | wrist)."""

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            v, t, c, h, w = images.shape

            # For DROID embodiment: 2x2 grid where the wrist view spans the full top row,
            # and the two exterior views occupy the bottom row.
            if self.embodiment_tag == EmbodimentTag.OXE_DROID and v >= 3:
                left_exterior = images[0]  # (t, c, h, w)
                right_exterior = images[1]  # (t, c, h, w)
                wrist_image = images[2]  # (t, c, h, w)

                concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

                wrist_wide = np.repeat(wrist_image, 2, axis=-1)  # (t, c, h, 2w)
                concat_images[0, :, :, :h, :] = wrist_wide

                concat_images[0, :, :, h:, :w] = left_exterior
                concat_images[0, :, :, h:, w:] = right_exterior

                return concat_images

            if self.embodiment_tag == EmbodimentTag.LIBERO_SIM and v >= 2:
                concat_images = np.zeros((1, t, c, h, 2 * w), dtype=images.dtype)
                concat_images[0, :, :, :, :w] = images[0]
                concat_images[0, :, :, :, w:] = images[1]
                return concat_images

            # For other embodiments: use 2x2 grid layout
            concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

            if v > 0:
                concat_images[0, :, :, :h, :w] = images[0]

            if v > 1:
                concat_images[0, :, :, h:, :w] = images[1]

            if v > 2:
                concat_images[0, :, :, :h, w:] = images[2]

            return concat_images

        return images
