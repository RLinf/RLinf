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

"""Four-view, absolute-qpos transforms matching Bench2Dex's pi05 policy.

Based on Bench2Dex fd90dcc (MIT); see third_party/Bench2Dex.LICENSE.
"""

import dataclasses

import numpy as np
from openpi import transforms

IMAGE_KEYS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_QUANTILE_RANGE_EPS = 1e-4


class Bench2DexNormalize(transforms.Normalize):
    """Use the upstream clipped quantiles and centered constant joints."""

    def _normalize_quantile(self, x, stats: transforms.NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        value_range = q99 - q01
        near_constant = value_range < _QUANTILE_RANGE_EPS
        center = (q99 + q01) / 2.0
        safe_range = np.where(near_constant, 1.0, value_range)
        scaled = (x - q01) / (safe_range + 1e-6) * 2.0 - 1.0
        normalized = np.where(near_constant, x - center, scaled)
        return np.clip(normalized, -1.0, 1.0)


class Bench2DexUnnormalize(transforms.Unnormalize):
    """Restore upstream quantiles while preserving padded coordinates."""

    def _unnormalize_quantile(self, x, stats: transforms.NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        dim = min(stats.q01.shape[-1], x.shape[-1])
        normalized = x[..., :dim]
        q01, q99 = stats.q01[..., :dim], stats.q99[..., :dim]
        value_range = q99 - q01
        near_constant = value_range < _QUANTILE_RANGE_EPS
        center = (q99 + q01) / 2.0
        scaled = (normalized + 1.0) / 2.0 * (value_range + 1e-6) + q01
        unnormalized = np.where(near_constant, normalized + center, scaled)
        if dim < x.shape[-1]:
            return np.concatenate([unnormalized, x[..., dim:]], axis=-1)
        return unnormalized


@dataclasses.dataclass(frozen=True)
class Bench2DexInputs(transforms.DataTransformFn):
    """Pack stereo and wrist views, preserving absolute joint coordinates."""

    action_dim: int
    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            # Native Bench2Dex samples use these four upstream camera names.
            frames = tuple(
                data["images"][key]
                for key in (
                    "cam_stereo_left",
                    "cam_stereo_right",
                    "cam_wrist_left",
                    "cam_wrist_right",
                )
            )
            state = np.asarray(data["state"], dtype=np.float32)
        else:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            extra = np.asarray(data["observation/extra_view_image"])
            wrists = np.asarray(data["observation/wrist_image"])
            if (
                extra.ndim != 4
                or extra.shape[0] != 1
                or wrists.ndim != 4
                or wrists.shape[0] != 2
            ):
                raise ValueError(
                    "Bench2Dex pi05 expects one extra stereo view and two wrist views"
                )
            frames = (data["observation/image"], extra[0], wrists[0], wrists[1])
        if state.shape != (self.action_dim,):
            raise ValueError(
                f"Expected Bench2Dex state ({self.action_dim},), got {state.shape}"
            )
        images = {}
        for key, frame in zip(IMAGE_KEYS, frames, strict=True):
            frame = np.asarray(frame)
            if frame.ndim != 3:
                raise ValueError(f"Expected RGB image, got {key}: {frame.shape}")
            if frame.shape[0] == 3 and frame.shape[-1] != 3:
                frame = frame.transpose(1, 2, 0)
            if np.issubdtype(frame.dtype, np.floating):
                frame = (255 * frame).astype(np.uint8)
            images[key] = frame
        result = {
            "image": images,
            "image_mask": dict.fromkeys(IMAGE_KEYS, np.True_),
            "state": transforms.pad_to_dim(state, self.model_action_dim),
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected action dim {self.action_dim}, got {actions.shape}"
                )
            result["actions"] = transforms.pad_to_dim(actions, self.model_action_dim)
        return result


@dataclasses.dataclass(frozen=True)
class Bench2DexOutputs(transforms.DataTransformFn):
    """Remove model padding after unnormalization, without gripper conversion."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"Model output width {actions.shape[-1]} is below environment width {self.action_dim}"
            )
        return {"actions": actions[..., : self.action_dim]}
