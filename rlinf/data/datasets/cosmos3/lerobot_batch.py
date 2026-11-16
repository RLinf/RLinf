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

"""Online LeRobot windows to Cosmos grouped action and video batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import torch


def prepare_lerobot_batch(
    batch: Mapping[str, Any],
    *,
    horizon: int,
    max_action_dim: int,
    normalizer: Any,
    prepare_observation_batch: Callable[
        ..., tuple[dict[str, Any], dict[str, torch.Tensor]]
    ],
    video_key: str = "video",
    include_vision: bool = False,
) -> dict[str, Any]:
    """Convert physical LIBERO labels and observations to Cosmos SFT inputs.

    Args:
        batch: LeRobot observations, task strings, raw actions [B,H,7], and a
            boolean ``actions_is_pad`` episode-tail mask [B,H].
        horizon: Number of action targets in each training window.
        max_action_dim: Padded Cosmos action width, at least 10.
        normalizer: Fixed LIBERO rot6d action normalizer, shared with inference.
        prepare_observation_batch: Builds grouped Cosmos observation metadata and
            batched prompt tokens with their validity mask.
        video_key: Native Cosmos model's input video field.
        include_vision: Train on real future observations where a complete
            ``vision_images`` clip is available, as indicated by ``vision_valid``.

    Returns:
        Native grouped inputs, with normalized rot6d actions and ragged tokens.

    Padded steps are clean zero conditioning slots, so native flow loss ignores
    them. They do not change the channel validity mask or raw action width.
    """
    raw7 = torch.as_tensor(batch["actions"]).detach().to(dtype=torch.float32)
    pad = torch.as_tensor(batch["actions_is_pad"])
    pad_cpu = pad.cpu()
    pad = pad.to(device=raw7.device)
    # Episode padding can contain repeated actions or NaNs; neither is a label.
    raw7 = raw7.masked_fill(pad.unsqueeze(-1), 0)

    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

    raw10 = torch.cat(
        [
            raw7[..., :3],
            convert_rotation(raw7[..., 3:6], "axisangle", "rot6d"),
            raw7[..., 6:7],
        ],
        dim=-1,
    )
    normalized = normalizer.normalize_action(raw10).detach().to(dtype=torch.float32)
    actions = torch.nn.functional.pad(normalized, (0, max_action_dim - 10))
    actions = actions.masked_fill(pad.unsqueeze(-1), 0)

    observations = {
        "main_images": batch["image"],
        "task_descriptions": batch["task"],
        "raw_actions": raw10,
        "actions_is_pad": pad_cpu,
    }
    if batch.get("wrist_image") is not None:
        observations["wrist_images"] = batch["wrist_image"]
    vision_kwargs = {}
    if include_vision:
        vision_valid = torch.as_tensor(batch["vision_valid"], device="cpu")
        vision_kwargs = {
            "vision_images": batch["vision_images"],
            "vision_wrist_images": batch.get("vision_wrist_images"),
            "vision_valid": vision_valid,
        }

    prepared, forward_inputs = prepare_observation_batch(observations, **vision_kwargs)
    data = dict(prepared)

    tokens = forward_inputs["tokenized_prompt"]
    token_mask = forward_inputs["tokenized_prompt_mask"]

    data["action"] = [[sample.contiguous()] for sample in actions]
    data["sequence_plan"] = [
        replace(
            plan,
            condition_frame_indexes_action=torch.where(sample_pad)[0].tolist(),
        )
        for plan, sample_pad in zip(data["sequence_plan"], pad_cpu, strict=True)
    ]
    # training_step accepts grouped ragged token IDs, not fixed-width replay IDs.
    data["text_token_ids"] = [
        [sample_tokens[sample_mask].to(dtype=torch.long).contiguous()]
        for sample_tokens, sample_mask in zip(tokens, token_mask, strict=True)
    ]
    return data
