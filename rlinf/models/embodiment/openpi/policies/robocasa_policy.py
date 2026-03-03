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
import logging

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from typing_extensions import Dict, List, Union

from rlinf.envs.robocasa.utils import (
    DEFAULT_ROBOCASA_IMAGE_SIZE,
    OPENPI_IMAGES,
    _check_action_space,
    _check_image_space,
    _check_state_space,
    get_action_ids,
    get_action_space,
    get_image_space,
    get_state_ids,
    get_state_space,
)


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        # "": np.random.rand(12),
        # NOTE: TODO: it seems that pi0 ask state and action be the same meaning, but informed that not in need
        "observation/state": np.random.rand(25),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/extra_view_image": np.random.randint(
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


def extract_state_dict(data: Dict, state_space: Union[str, List[str]]) -> Dict:
    state_space = get_state_space(state_space)
    state_check_ret = _check_state_space(state_space)
    if not state_check_ret:
        state_content = ", ".join(list(state_space))
        logging.warning(
            f"List-format state_space got invalid content: {state_content}, Use default '25d' state_space instead."
        )
        state_space = get_state_space("25d")

    state_dict = {}
    all_state_ids = get_state_ids(state_space)
    state_dict["state"] = data["observation/state"][all_state_ids]

    return state_dict


def extract_image_dict(data: Dict, image_space: Union[str, Dict]) -> Dict:
    image_space = get_image_space(image_space)
    img_check_ret = _check_image_space(image_space)
    if not img_check_ret:
        img_kv_pairs = ", ".join([f"{k}={v}" for k, v in image_space.items()])
        logging.warning(
            f"Dict-format image_space got invalid content: {img_kv_pairs}, Use default '2views' image_space instead."
        )
        image_space = get_image_space("2views")

    image_dict = {
        "image": {
            openpi_img_name: np.zeros(DEFAULT_ROBOCASA_IMAGE_SIZE, dtype=np.uint8)
            for openpi_img_name in OPENPI_IMAGES
        },
        "image_mask": dict.fromkeys(OPENPI_IMAGES, np.False_),
    }
    for openpi_img_name, robocasa_img_name in image_space.items():
        parsed_img = _parse_image(data[robocasa_img_name])
        image_dict["image"].update({openpi_img_name: parsed_img})
        image_dict["image_mask"].update({openpi_img_name: np.True_})

    return image_dict


def extract_action_ids(action_space: Union[str, List[str]]) -> Dict:
    action_space = get_action_space(action_space)
    action_check_ret = _check_action_space(action_space)
    if not action_check_ret:
        action_content = ", ".join(list(action_space))
        logging.warning(
            f"List-format action_space got invalid content: {action_content}. Use default '12d' action_space instead."
        )
        action_space = get_action_space("12d")

    all_action_ids = get_action_ids(action_space)

    return all_action_ids


def extract_action_dict(data: Dict, action_space: Union[str, List[str]]) -> Dict:
    action_dict = {}
    all_action_ids = extract_action_ids(action_space)

    action_dict["actions"] = data["actions"][:, all_action_ids]

    return action_dict


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """
    This class is used to convert the lerobot-formatted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    state_space: Union[str, List[str]]
    image_space: Union[str, Dict]
    action_space: Union[str, List[str]]
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.

        inputs = {}
        # STATE #####

        state_dict = extract_state_dict(data, self.state_space)

        inputs.update(state_dict)

        # IMAGE #####

        image_dict = extract_image_dict(data, self.image_space)

        inputs.update(image_dict)

        # ACTIONS #####

        # Pad actions to the model action dimension.
        # Actions are only available during training.
        if "actions" in data:
            action_dict = extract_action_dict(data, self.action_space)

            inputs.update(action_dict)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_space: Union[str, List[str]]

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.

        action_dim = len(get_action_ids(get_action_space(self.action_space)))

        return {"actions": np.asarray(data["actions"][:, :action_dim])}
