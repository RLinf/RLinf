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
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

TURTLE_ACTION_DIM = 6


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TurtleInputs(transforms.DataTransformFn):
    """Converts turtle env observations into pi0/pi0.5 model input format.

    Turtle env provides 3 cameras (224x224) and 6-dim state (xyz + euler,
    single arm).  The main image maps to base_0_rgb; the two extra-view
    images map to left_wrist_0_rgb and right_wrist_0_rgb.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        image_dict = {"base_0_rgb": base_image}
        mask_dict = {"base_0_rgb": np.True_}

        extra = data.get("observation/extra_view_image")
        if extra is not None:
            extra = np.asarray(extra)
            if extra.ndim == 4:
                imgs = [_parse_image(extra[:, i]) for i in range(extra.shape[1])]
            elif extra.ndim == 3:
                imgs = [_parse_image(extra)]
            else:
                imgs = []

            wrist_keys = ["left_wrist_0_rgb", "right_wrist_0_rgb"]
            for i, key in enumerate(wrist_keys):
                if i < len(imgs):
                    image_dict[key] = imgs[i]
                    mask_dict[key] = np.True_
                else:
                    image_dict[key] = np.zeros_like(base_image)
                    mask_dict[key] = np.False_
        else:
            image_dict["left_wrist_0_rgb"] = np.zeros_like(base_image)
            image_dict["right_wrist_0_rgb"] = np.zeros_like(base_image)
            mask_dict["left_wrist_0_rgb"] = np.False_
            mask_dict["right_wrist_0_rgb"] = np.False_

        inputs = {
            "state": data["observation/state"],
            "image": image_dict,
            "image_mask": mask_dict,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TurtleOutputs(transforms.DataTransformFn):
    """Extracts the first TURTLE_ACTION_DIM actions from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :TURTLE_ACTION_DIM])}
