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


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexJoCoInputs(transforms.DataTransformFn):
    """Map DexJoCo observations and absolute actions to OpenPI inputs."""

    model_type: _model.ModelType
    dual_arm: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        if self.dual_arm:
            wrist_images = data.get("observation/wrist_image")
            if wrist_images is None:
                wrist_images = (
                    data["observation/wrist_image-0"],
                    data["observation/wrist_image-1"],
                )
            left_wrist_image = _parse_image(wrist_images[0])
            right_wrist_image = _parse_image(wrist_images[1])
        else:
            left_wrist_image = _parse_image(data["observation/wrist_image"])
            right_wrist_image = np.zeros_like(base_image)

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
                if self.dual_arm or self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DexJoCoOutputs(transforms.DataTransformFn):
    """Trim model actions to the 22- or 44-dim DexJoCo rotation-vector action."""

    dual_arm: bool = False

    def __call__(self, data: dict) -> dict:
        action_dim = 44 if self.dual_arm else 22
        return {"actions": np.asarray(data["actions"][:, :action_dim])}
