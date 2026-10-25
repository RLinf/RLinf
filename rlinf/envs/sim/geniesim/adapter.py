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

"""Observation adapter for GenieSim vector-env outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlinf.envs.utils import to_tensor

_CAM_MAP = {
    "main_images": "main",
    "wrist_images": "wrist_left",
    "wrist_images_right": "wrist_right",
}


class GenieSimAdapter:
    """Translate GenieSim observations to official RLinf embodied obs keys."""

    def __init__(
        self,
        *,
        main_image_key: str = "main",
        extra_image_keys: list[str] | None = None,
    ) -> None:
        self.main_image_key = str(main_image_key)
        self.extra_image_keys = (
            list(extra_image_keys)
            if extra_image_keys is not None
            else ["wrist_left", "wrist_right"]
        )

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _task_descriptions(raw_obs: dict[str, Any], num_envs: int) -> list[str]:
        raw = raw_obs.get("task_descriptions")
        if isinstance(raw, str) or not isinstance(raw, list | tuple):
            raise ValueError("GenieSim task_descriptions must be a list of strings")
        if len(raw) != num_envs:
            raise ValueError(
                "GenieSim task_descriptions must have length "
                f"{num_envs}, got {len(raw)}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in raw):
            raise ValueError(
                "GenieSim task_descriptions must contain non-empty strings"
            )
        return [value.strip() for value in raw]

    def _frame_batch(self, value: Any) -> np.ndarray:
        frames = self._np(value)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected an NHWC image batch, got shape {frames.shape}")
        return np.ascontiguousarray(frames.astype(np.uint8, copy=False))

    def wrap_batch(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert one vector observation returned by ``GenieSimVecEnv``."""
        states = self._np(raw_obs["states"]).astype(np.float32, copy=False)
        if states.ndim != 2:
            raise ValueError(f"Expected a 2-D GenieSim state batch, got {states.shape}")
        num_envs = states.shape[0]
        if num_envs == 0:
            raise ValueError("wrap_batch received an empty observation batch")

        descriptions = self._task_descriptions(raw_obs, num_envs)
        frames = {
            dst: self._frame_batch(raw_obs[src])
            for src, dst in _CAM_MAP.items()
            if src in raw_obs
        }
        if self.main_image_key not in frames:
            raise KeyError(
                f"GenieSim observation is missing main camera {self.main_image_key!r}"
            )
        if any(frame.shape[0] != num_envs for frame in frames.values()):
            raise ValueError("GenieSim image batches must match the state batch size")

        out: dict[str, Any] = {
            "states": states,
            "main_images": frames[self.main_image_key],
        }
        extra_keys = self.extra_image_keys or sorted(
            key for key in frames if key != self.main_image_key
        )
        if extra_keys:
            missing = [key for key in extra_keys if key not in frames]
            if missing:
                raise KeyError(f"GenieSim observation is missing cameras: {missing}")
            out["extra_view_images"] = np.stack(
                [frames[key] for key in extra_keys], axis=1
            )
        tensor_out = to_tensor(out)
        tensor_out["task_descriptions"] = descriptions
        return tensor_out
