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
"""
RLBench policy transforms for OpenPi.

Action normalization aligned with Metaworld/Libero: same openpi pipeline
(input Normalize + output Unnormalize with norm_stats; no extra scaling in env).
RLBench 18-tasks: image, overhead_image, wrist_image, state (7D), actions (7D:
delta_pos 3, delta_euler 3, delta_gripper 1). Env only converts 7D -> 8D for RLBench API.
"""

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
class RLBenchInputs(transforms.DataTransformFn):
    """
    Convert RLBench dataset inputs to model format.
    num_views: 1=front only, 2=front+wrist, 3=front+wrist+overhead.
    image_mask controls which views the model attends to.
    """

    model_type: _model.ModelType
    num_views: int = 3

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        overhead_image = _parse_image(data["observation/overhead_image"])

        n = max(1, min(3, self.num_views))
        mask_base = True
        mask_wrist = n >= 2
        mask_overhead = n >= 3

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": overhead_image,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(mask_base),
                "left_wrist_0_rgb": np.bool_(mask_wrist),
                "right_wrist_0_rgb": np.bool_(mask_overhead),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    """Convert model outputs to RLBench format. 7D actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
