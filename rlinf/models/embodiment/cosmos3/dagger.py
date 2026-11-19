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

"""Cosmos3 policy integration for online DAgger training."""

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.dagger import DAggerPolicyAdapter
from rlinf.utils.nested_dict_process import put_tensor_device


class Cosmos3DAggerPolicyAdapter(DAggerPolicyAdapter):
    """Prepare Cosmos SFT windows and track the optimizer clock for EMA."""

    @staticmethod
    def set_optimizer_step(
        model: torch.nn.Module, step: int, *, after_update: bool
    ) -> None:
        """Only a successful optimizer update may advance the EMA weights."""
        if after_update:
            model.set_global_step(step)
        else:
            model.set_global_step(step, update_ema=False)

    @staticmethod
    def lerobot_dataset_options(
        model_config: DictConfig,
        *,
        training_action_horizon: int,
        window_size: int | None,
        require_all_intervene: bool,
    ) -> dict[str, int | bool]:
        """Request complete future video windows for the joint SFT objective."""
        joint_vision = model_config.get("sft_objective") == "joint"
        return {
            "vision_horizon": training_action_horizon if joint_vision else 0,
            "require_full_window": joint_vision,
            "keep_uint8_images": (
                joint_vision
                and not require_all_intervene
                and window_size is not None
                and int(window_size) > 0
            ),
        }

    @staticmethod
    def stage_lerobot_batch(
        batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Keep pixels on the host until Cosmos builds its native video batch."""
        prepared = {
            key: batch[key]
            for key in (
                "task",
                "image",
                "wrist_image",
                "vision_images",
                "vision_wrist_images",
                "vision_valid",
            )
            if key in batch
        }
        prepared.update(
            put_tensor_device(
                {
                    key: batch[key]
                    for key in ("actions", "actions_is_pad")
                    if key in batch
                },
                device=device,
            )
        )
        return prepared
