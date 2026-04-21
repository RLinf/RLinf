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
import torch
from openpi import transforms
from openpi.models import model as _model


def make_franka_dagger_example() -> dict:
    """Creates a random input example for the Franka dagger policy."""
    return {
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/extra_view_image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(19),
        "prompt": "pick up the object and put it into another bin",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaDaggerOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset action format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class FrankaDaggerInputs(transforms.DataTransformFn):
    """Converts Franka dagger dataset samples to OpenPI inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (19,), (
            f"Expected state shape (19,), got {data['observation/state'].shape}"
        )

        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(
                data["observation/state"]
            ).float()

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        extra_view_image = _parse_image(
            data.get("observation/extra_view_image", np.zeros_like(base_image))
        )

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (
                base_image,
                extra_view_image,
                np.zeros_like(base_image),
            )
            image_masks = (np.True_, np.True_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (
                base_image,
                extra_view_image,
                np.zeros_like(base_image),
            )
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 7, (
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
