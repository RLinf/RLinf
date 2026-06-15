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


def make_rlt_joint_example() -> dict:
    """Creates a random input example for joint-space RLT datasets.

    Raw camera resolution is dataset-specific. The current realworld dataset uses
    128x128 images, while some simulation pipelines may emit larger frames. OpenPI
    resizes images later in the model transform stack, so this policy transform does
    not assume a fixed input resolution here.
    """
    return {
        "observation/state": np.random.rand(34),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "prompt": "insert the peg in the hole",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RLTJointInputs(transforms.DataTransformFn):
    """Converts joint-space RLT samples to the OpenPI input format.

    This layer only remaps keys and normalizes image layout to HWC uint8. It does
    not enforce a fixed camera resolution; OpenPI's downstream model transforms
    perform the actual resize step (pi0/pi05 default: 224x224).
    """

    model_type: _model.ModelType
    use_wrist_image: bool = False
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(
            data.get(
                "observation/wrist_image",
                data.get("observation/extra_view_image", np.zeros_like(base_image)),
            )
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image
                if self.use_wrist_image
                else np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.use_wrist_image else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        elif self.default_prompt is not None:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RLTJointOutputs(transforms.DataTransformFn):
    """Converts OpenPI outputs back to the joint-space RLT action format."""

    output_action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
