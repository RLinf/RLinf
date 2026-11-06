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

"""The observations an env mode sends to the policy."""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class PolicyObsSpec:
    """Conforms env observations to the keys and image size of one policy batch.

    Rollout concatenates the observations of every env in a mode into one
    batch, so envs from different simulators must send the same keys with the
    same shapes. ``keys`` keeps only the observations the policy reads, and
    ``image_size`` resizes every image view to one ``[height, width]``. Only the
    policy's copy is conformed; reward models, video, and data collection see
    what the env produced.
    """

    CFG_KEY: ClassVar[str] = "policy_obs"
    # Keys of ``EnvOutput.prepare_observations``; images are ``[..., H, W, C]``.
    KEYS: ClassVar[tuple[str, ...]] = (
        "main_images",
        "wrist_images",
        "extra_view_images",
        "states",
        "task_descriptions",
    )
    IMAGE_KEYS: ClassVar[tuple[str, ...]] = KEYS[:3]

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        image_size: Sequence[int] | None = None,
    ):
        """Describe the policy observations.

        Args:
            keys: Observations the policy reads; the rest are sent as ``None``.
                ``None`` keeps every key.
            image_size: ``[height, width]`` every image view is resized to.
                ``None`` keeps each env's resolution.
        """
        if keys is not None:
            unknown = [key for key in keys if key not in self.KEYS]
            if unknown:
                raise ValueError(
                    f"{self.CFG_KEY}.keys must be drawn from {list(self.KEYS)}, "
                    f"got unknown keys {unknown}."
                )
        if image_size is not None and len(image_size) != 2:
            raise ValueError(
                f"{self.CFG_KEY}.image_size must be [height, width], "
                f"got {list(image_size)}."
            )
        self._keys = None if keys is None else frozenset(keys)
        self._image_size = None if image_size is None else tuple(image_size)

    @classmethod
    def from_cfg(cls, mode_cfg: DictConfig | None) -> "PolicyObsSpec":
        """Build the spec ``mode_cfg.policy_obs`` describes; absent means no change."""
        spec_cfg = None if mode_cfg is None else mode_cfg.get(cls.CFG_KEY)
        if spec_cfg is None:
            return cls()
        return cls(keys=spec_cfg.get("keys"), image_size=spec_cfg.get("image_size"))

    def apply(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return the policy's copy of ``obs``; ``obs`` itself is not modified."""
        if self._keys is None and self._image_size is None:
            return obs
        conformed = {}
        for key, value in obs.items():
            if self._keys is not None and key not in self._keys:
                value = None
            elif (
                key in self.IMAGE_KEYS
                and value is not None
                and self._image_size is not None
            ):
                value = self._resize(value)
            conformed[key] = value
        return conformed

    def _resize(self, images: Any) -> torch.Tensor:
        images = torch.as_tensor(images)
        height, width = self._image_size
        if tuple(images.shape[-3:-1]) == (height, width):
            return images
        channels = images.shape[-1]
        flat = images.reshape(-1, *images.shape[-3:]).permute(0, 3, 1, 2)
        resized = F.interpolate(
            flat.float(), size=(height, width), mode="bilinear", antialias=True
        )
        if not images.dtype.is_floating_point:
            resized = resized.round().clamp(
                torch.iinfo(images.dtype).min, torch.iinfo(images.dtype).max
            )
        return (
            resized.to(images.dtype)
            .permute(0, 2, 3, 1)
            .reshape(*images.shape[:-3], height, width, channels)
        )
