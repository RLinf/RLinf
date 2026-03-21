# Copyright 2025 The RLinf Authors.
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
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from PIL import Image


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_wrist_images(
    data: dict, base_image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split wrist-camera observations into left/right views when available."""
    wrist_image = _parse_image(data["observation/wrist_image"])
    extra_view_image = (
        _parse_image(data["observation/extra_view_image"])
        if "observation/extra_view_image" in data
        else None
    )

    if wrist_image.ndim == 4:
        left_wrist_image = wrist_image[0]
        right_wrist_image = wrist_image[1] if wrist_image.shape[0] > 1 else None
    else:
        left_wrist_image = wrist_image
        right_wrist_image = None

    if right_wrist_image is None and extra_view_image is not None:
        right_wrist_image = (
            extra_view_image[0] if extra_view_image.ndim == 4 else extra_view_image
        )

    has_right_wrist = right_wrist_image is not None
    if right_wrist_image is None:
        right_wrist_image = np.zeros_like(base_image)

    return left_wrist_image, right_wrist_image, has_right_wrist


def _resize_image_exact(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image batch to the exact target size without aspect-ratio padding."""
    image = np.asarray(image)
    if image.shape[-3:-1] == (height, width):
        return image

    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), resample=Image.BILINEAR)
            )
            for frame in image
        ],
        axis=0,
    )
    return resized.reshape(*original_shape[:-3], height, width, original_shape[-1])


@dataclasses.dataclass(frozen=True)
class ResizeImagesNoPad(transforms.DataTransformFn):
    """Resize images exactly like the original LeRobot PI0.5 helper path."""

    height: int
    width: int

    def __call__(self, data: dict) -> dict:
        data["image"] = {
            key: _resize_image_exact(value, self.height, self.width)
            for key, value in data["image"].items()
        }
        return data


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right).
        # If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and
        # replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        left_wrist_image, right_wrist_image, has_right_wrist = _parse_wrist_images(
            data, base_image
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if has_right_wrist or self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # Number of action dimensions to keep from the model's max_action_dim output.
    # Default 7 for single-arm Libero; set to 14 for bimanual (e.g. YAM).
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
