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

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.utils.nested_dict_process import put_tensor_device


class Cosmos3DAggerAdapter:
    """Cosmos input and training-step conventions for the DAgger actor."""

    @staticmethod
    def lerobot_window_options(
        model_cfg: DictConfig, online_cfg: DictConfig, require_all_intervene: bool
    ) -> dict:
        horizon = online_cfg.get("training_action_horizon", model_cfg.num_action_chunks)
        joint = model_cfg.get("sft_objective") == "joint"
        window_size = online_cfg.get("rolling_lerobot_window_size")
        return {
            "chunk_size": horizon,
            "vision_horizon": horizon if joint else 0,
            "require_full_window": joint,
            "keep_uint8_images": (
                joint
                and not require_all_intervene
                and window_size is not None
                and int(window_size) > 0
            ),
        }

    @staticmethod
    def stage_lerobot_batch(
        batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        # Leave pixels on the host until Cosmos builds its video batch.
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

    @staticmethod
    def loss_and_metrics(
        output: torch.Tensor | dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Keep the native loss graph and detach Cosmos reporting metrics."""
        if not isinstance(output, dict):
            return output, {}
        metrics = {
            f"dagger/{key}": value.detach().float().item()
            for key, value in output.items()
            if key != "loss" and torch.is_tensor(value) and value.numel() == 1
        }
        return output["loss"], metrics

    @staticmethod
    def after_optimizer_step(model: torch.nn.Module, step: int) -> None:
        model.set_global_step(step)

    @staticmethod
    def restore_optimizer_step(model: torch.nn.Module, step: int) -> None:
        """Restore the iteration without an EMA update on resume or version changes."""
        model.set_global_step(step, update_ema=False)
