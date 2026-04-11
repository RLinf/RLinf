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


def make_calvin_example() -> dict:
    """Creates a random input example for the Calvin policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(200, 200, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(84, 84, 3), dtype=np.uint8
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


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state_ee_pos = data["observation/state_ee_pos"]
        state_ee_rot = data["observation/state_ee_rot"]
        state_gripper = data["observation/state_gripper"]
        state = np.concatenate([state_ee_pos, state_ee_rot, state_gripper])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions/delta_ee_pos" in data:
            delta_ee_pos = data["actions/delta_ee_pos"]
            delta_ee_rot = data["actions/delta_ee_rot"]
            gripper = data["actions/gripper"]
            actions = np.concatenate([delta_ee_pos, delta_ee_rot, gripper], axis=1)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
