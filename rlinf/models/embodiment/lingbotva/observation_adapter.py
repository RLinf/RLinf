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

"""Observation conversion utilities for LingBot-VA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class LingbotVAObservationAdapter:
    """Convert RLinf RoboTwin observations into LingBot-VA official format."""

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.to(torch.float32)
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _select_image_tensor(image_tensor: Any, env_idx: int) -> np.ndarray:
        if image_tensor is None:
            raise ValueError("LingBot-VA requires image observations, but got None.")
        if isinstance(image_tensor, torch.Tensor):
            return image_tensor[env_idx].detach().cpu().numpy()
        return np.asarray(image_tensor[env_idx])

    @staticmethod
    def _select_state_tensor(state_tensor: Any, env_idx: int) -> np.ndarray:
        if state_tensor is None:
            raise ValueError("LingBot-VA requires proprio/state observations.")
        if isinstance(state_tensor, torch.Tensor):
            if state_tensor.dtype == torch.bfloat16:
                state_tensor = state_tensor.to(torch.float32)
            return state_tensor[env_idx].detach().cpu().numpy()
        return np.asarray(state_tensor[env_idx])

    @staticmethod
    def _resolve_raw_image(raw_obs: dict[str, Any], key: str, camera_key: str) -> Any:
        value = raw_obs.get(key)
        if value is not None:
            return value
        return raw_obs.get("observation", {}).get(camera_key, {}).get("rgb")

    @staticmethod
    def _resolve_raw_state(raw_obs: dict[str, Any]) -> Any:
        value = raw_obs.get("state")
        if value is not None:
            return value
        return raw_obs.get("joint_action", {}).get("vector")

    @classmethod
    def format_raw_step_observation(
        cls,
        raw_obs: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Convert one raw RoboTwin step observation to LingBot-VA official format."""
        left_wrist = cls._resolve_raw_image(raw_obs, "left_wrist_image", "left_camera")
        right_wrist = cls._resolve_raw_image(
            raw_obs, "right_wrist_image", "right_camera"
        )
        if left_wrist is None or right_wrist is None:
            raise ValueError(
                "LingBot-VA raw RoboTwin step observation expects both left and right wrist images."
            )

        state = cls._resolve_raw_state(raw_obs)
        if state is None:
            raise ValueError("LingBot-VA raw RoboTwin step observation expects state.")

        main_image = cls._resolve_raw_image(raw_obs, "full_image", "head_camera")
        if main_image is None:
            raise ValueError(
                "LingBot-VA raw RoboTwin step observation expects full_image."
            )

        return {
            "observation.images.cam_high": cls._to_numpy(main_image),
            "observation.images.cam_left_wrist": cls._to_numpy(left_wrist),
            "observation.images.cam_right_wrist": cls._to_numpy(right_wrist),
            "observation.state": cls._to_numpy(state),
            "task": prompt,
        }

    @classmethod
    def format_observation(
        cls,
        env_obs: dict[str, Any],
        env_idx: int,
        prompt: str,
    ) -> dict[str, Any]:
        """Convert one batch element to the official LingBot-VA RoboTwin format."""
        wrist_images = env_obs.get("wrist_images")
        left_wrist = None
        right_wrist = None
        if wrist_images is not None:
            if isinstance(wrist_images, torch.Tensor):
                wrist_np = wrist_images[env_idx].detach().cpu().numpy()
            else:
                wrist_np = np.asarray(wrist_images[env_idx])
            if wrist_np.ndim == 4 and wrist_np.shape[0] >= 2:
                left_wrist = wrist_np[0]
                right_wrist = wrist_np[1]

        if left_wrist is None or right_wrist is None:
            raise ValueError(
                "LingBot-VA RoboTwin adapter expects both left and right wrist images."
            )
        main_image = cls._select_image_tensor(env_obs.get("main_images"), env_idx)

        return {
            "observation.images.cam_high": main_image,
            "observation.images.cam_left_wrist": left_wrist,
            "observation.images.cam_right_wrist": right_wrist,
            "observation.state": cls._select_state_tensor(
                env_obs.get("states"), env_idx
            ),
            "task": prompt,
        }
