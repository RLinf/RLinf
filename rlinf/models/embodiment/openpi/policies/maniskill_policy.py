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


def make_maniskill_example() -> dict:
    """Creates a random input example for the ManiSkill policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
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
class ManiSkillInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """

        print(f"data={data.keys()}") # data=dict_keys(['observation/image', 'observation/state', 'prompt'])

        Return values:
        inputs: dict, keys=['state', 'image', 'image_mask', 'prompt']
            state: torch.shape=8, float32
            image:dict[str, torch.Tensor]
                base_0_rgb: torch.shape=(480, 640, 3), uint8
                ...
            ...
        """

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

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
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

        # print(f"inputs={inputs.keys()}")

        # print(f"inputs/image/base_0_rgb={inputs['image']['base_0_rgb'].shape}, {inputs['image']['base_0_rgb'].dtype}")
        # print(f"inputs/state={inputs['state'].shape}, {inputs['state'].dtype}")
        return inputs


@dataclasses.dataclass(frozen=True)
class ManiSkillOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (sfince the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
