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

from __future__ import annotations

import dataclasses
from typing import Any

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 HWC format for OpenPI transforms."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class StreamingVLALiberoInputs(transforms.DataTransformFn):
    """Map a LIBERO LeRobot sample into OpenPI's observation schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Transform one LIBERO sample while retaining ``action_states``."""
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        pi0_fast = getattr(_model.ModelType, "PI0_FAST", None)
        use_right_wrist = self.model_type == pi0_fast if pi0_fast is not None else False
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if use_right_wrist else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "observation/action_states" in data:
            inputs["action_states"] = data["observation/action_states"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class StreamingVLALiberoOutputs(transforms.DataTransformFn):
    """Trim padded actions to the environment action dimension."""

    action_env_dim: int = 7

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return actions in the environment dimension."""
        return {"actions": np.asarray(data["actions"])[..., : self.action_env_dim]}


@dataclasses.dataclass(frozen=True)
class PadStreamingVLAStatesActions(transforms.DataTransformFn):
    """Pad states, action deltas, and initial action states consistently."""

    model_action_dim: int

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pad the numeric fields to ``model_action_dim``."""
        data["state"] = transforms.pad_to_dim(
            data["state"], self.model_action_dim, axis=-1
        )
        if "actions" in data:
            data["actions"] = transforms.pad_to_dim(
                data["actions"], self.model_action_dim, axis=-1
            )
        if "action_states" in data and data["action_states"] is not None:
            data["action_states"] = transforms.pad_to_dim(
                data["action_states"], self.model_action_dim, axis=-1
            )
        return data


class StreamingVLANormalize(transforms.Normalize):
    """StreamingVLA's zero-centred linear quantile normalization."""

    def _normalize_quantile(
        self, x: np.ndarray, stats: transforms.NormStats
    ) -> np.ndarray:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("StreamingVLA quantile stats require q01 and q99.")
        q01 = stats.q01
        q99 = stats.q99
        scale = np.maximum(np.abs(q01), np.abs(q99))
        if (dim := scale.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [x[..., :dim] / (scale + 1e-6), x[..., dim:]], axis=-1
            )
        return x / (scale + 1e-6)


class StreamingVLAUnnormalize(transforms.Unnormalize):
    """Inverse of :class:`StreamingVLANormalize`."""

    def _unnormalize_quantile(
        self, x: np.ndarray, stats: transforms.NormStats
    ) -> np.ndarray:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("StreamingVLA quantile stats require q01 and q99.")
        scale = np.maximum(np.abs(stats.q01), np.abs(stats.q99))
        if (dim := scale.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [x[..., :dim] * (scale + 1e-6), x[..., dim:]], axis=-1
            )
        return x * (scale + 1e-6)
