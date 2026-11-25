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

"""OpenPI input/output transforms for RoboUniverse-10K policies."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboUniverseInputs(transforms.DataTransformFn):
    """Map RoboUniverse-10K observations and joint-position actions to OpenPI inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_raw = data.get("observation/wrist_image")
        has_wrist = wrist_raw is not None
        left_wrist_image = (
            _parse_image(wrist_raw) if has_wrist else np.zeros_like(base_image)
        )
        right_wrist_image = np.zeros_like(base_image)

        is_pi0_fast = self.model_type == _model.ModelType.PI0_FAST
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist or is_pi0_fast else np.False_,
                "right_wrist_0_rgb": np.True_ if is_pi0_fast else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboUniverseOutputs(transforms.DataTransformFn):
    """Slice OpenPI output actions to the RoboUniverse robot's joint dimension."""

    action_dim: int = 9

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
