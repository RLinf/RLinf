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


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        "observation/state": np.random.rand(
            8
        ),  # eef_pos (3) + eef_quat (4) + gripper (1)
        "observation/image": np.random.randint(
            256, size=(3, 224, 224), dtype=np.uint8
        ),  # base view
        "observation/wrist_image": np.random.randint(
            256, size=(3, 224, 224), dtype=np.uint8
        ),  # wrist camera
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
class RobocasaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For robocasa, we use two 128x128 camera views: base (left agentview) and wrist.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # During inference, images come from robocasa_env as (C,H,W) float or uint8
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,  # 128x128 base view
                "left_wrist_0_rgb": wrist_image,  # 128x128 wrist view
                # Pad right wrist with zeros since we only have one wrist camera
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
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

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For robocasa, different robots have different action dimensions:
    - Panda arm only: 7D actions (7 joint angles)
    - PandaOmron (with mobile base): 12D actions (7 arm + 5 base)

    Args:
        action_dim: Target action dimension. If None, will be auto-detected from robot configuration.
                    Common values: 7 (Panda), 12 (PandaOmron)
    """

    action_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])

        # If action_dim is specified, use it
        if self.action_dim is not None:
            return {"actions": actions[:, : self.action_dim]}

        # Auto-detect: if actions have exactly 7 or 12 dims, keep them
        # Otherwise, default to first 12 dims for PandaOmron
        if actions.shape[-1] in [7, 12]:
            return {"actions": actions}
        else:
            # Default to 12D for PandaOmron
            return {"actions": actions[:, :12]}
