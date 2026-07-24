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

"""Self-contained port of Evo-1's action/state normalizer.

Mirrors ``Evo_1/scripts/Evo1_server.py::Normalizer`` and the
``NormalizationType`` enum from ``Evo_1/dataset/lerobot_dataset_pretrain_mp.py``,
but without pulling in the Evo-1 inference server's heavy dependencies
(websockets, cv2, fvcore). Keeping it here also decouples RLinf from Evo-1's
module layout.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Union

import torch


class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


class Normalizer:
    """Normalize proprio state and denormalize predicted actions.

    Stats are keyed by ``arm_key`` and (optionally) ``dataset_key`` and stored
    in the checkpoint's ``norm_stats.json``. All vectors are padded to
    ``target_dim`` (24) to match Evo-1's multi-embodiment action space.
    """

    def __init__(
        self,
        stats_or_path: Union[str, dict],
        normalization_type: Union[str, NormalizationType] = NormalizationType.BOUNDS,
        target_dim: int = 24,
    ):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                self.stats_map = json.load(f)
        else:
            self.stats_map = stats_or_path

        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)
        self.normalization_type = normalization_type
        self.target_dim = target_dim
        self._cache_stats: dict = {}

    def _pad_vector(self, values, name):
        tensor = torch.tensor(values, dtype=torch.float32)
        length = tensor.shape[0]
        if length < self.target_dim:
            pad = torch.zeros(self.target_dim - length, dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif length > self.target_dim:
            raise ValueError(
                f"{name} length {length} exceeds expected {self.target_dim}"
            )
        return tensor

    def _prepare_stats(self, stats_dict, stats_name):
        return {
            key: self._pad_vector(values, f"{stats_name}.{key}")
            for key, values in stats_dict.items()
        }

    @staticmethod
    def _stat_to_device(stats_dict, key, device, dtype):
        tensor = stats_dict.get(key)
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype)

    def _get_stats_for(self, arm_key, dataset_key, stats_type):
        cache_key = (arm_key, dataset_key, stats_type)
        if cache_key in self._cache_stats:
            return self._cache_stats[cache_key]

        if arm_key not in self.stats_map:
            raise ValueError(f"Arm key '{arm_key}' not found in normalization stats.")

        arm_stats = self.stats_map[arm_key]
        if "observation.state" in arm_stats or "action" in arm_stats:
            raw_stats = arm_stats
        else:
            if dataset_key not in arm_stats:
                raise ValueError(
                    f"Dataset key '{dataset_key}' not found in normalization stats "
                    f"for arm '{arm_key}'."
                )
            raw_stats = arm_stats[dataset_key]

        dict_key = "observation.state" if stats_type == "state" else "action"
        if dict_key not in raw_stats:
            raise ValueError(
                f"Key '{dict_key}' not found in stats for {arm_key}/{dataset_key}"
            )

        prepared = self._prepare_stats(raw_stats[dict_key], dict_key)
        self._cache_stats[cache_key] = prepared
        return prepared

    def _normalize_tensor(self, tensor, stats_dict, clamp):
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type
        current_dim = tensor.shape[-1]

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError(
                    "Normal normalization selected but mean/std are missing."
                )
            return (tensor - mean[..., :current_dim]) / (std[..., :current_dim] + eps)

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)
        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)
        if low is None or high is None:
            raise ValueError(
                "Bounds normalization selected but min/max stats are missing."
            )

        low = low[..., :current_dim]
        high = high[..., :current_dim]
        normalized = 2 * (tensor - low) / (high - low + eps) - 1
        if clamp:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def _denormalize_tensor(self, tensor, stats_dict):
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError(
                    "Normal denormalization requested but mean/std stats are missing."
                )
            return tensor * (std + eps) + mean

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)
        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)
        if low is None or high is None:
            raise ValueError(
                "Bounds denormalization requested but min/max stats are missing."
            )

        current_dim = tensor.shape[-1]
        if low.shape[-1] > current_dim:
            low = low[..., :current_dim]
            high = high[..., :current_dim]
        return (tensor + 1.0) / 2.0 * (high - low + eps) + low

    def normalize_state(self, state, arm_key, dataset_key):
        stats = self._get_stats_for(arm_key, dataset_key, "state")
        norm_state = self._normalize_tensor(state, stats, clamp=True)
        if norm_state.shape[-1] < self.target_dim:
            pad = torch.zeros(
                (*norm_state.shape[:-1], self.target_dim - norm_state.shape[-1]),
                dtype=norm_state.dtype,
                device=norm_state.device,
            )
            norm_state = torch.cat([norm_state, pad], dim=-1)
        return norm_state

    def denormalize_action(self, action, arm_key, dataset_key):
        if action.ndim == 1:
            action = action.view(1, -1)
        stats = self._get_stats_for(arm_key, dataset_key, "action")
        denorm = self._denormalize_tensor(action, stats)
        if denorm.shape[-1] < self.target_dim:
            pad = torch.zeros(
                (*denorm.shape[:-1], self.target_dim - denorm.shape[-1]),
                dtype=denorm.dtype,
                device=denorm.device,
            )
            denorm = torch.cat([denorm, pad], dim=-1)
        return denorm
