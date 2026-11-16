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
    if horizon <= 0 or max_action_dim < 10:
        raise ValueError(
            "Expected horizon > 0 and max_action_dim >= 10; "
            f"got horizon={horizon}, max_action_dim={max_action_dim}."
        )
    missing = {"image", "actions", "task", "actions_is_pad"} - batch.keys()
    if missing:
        raise ValueError(f"LeRobot Cosmos batch is missing fields: {sorted(missing)}.")
    raw7 = torch.as_tensor(batch["actions"]).detach().to(dtype=torch.float32)
    if raw7.ndim != 3 or raw7.shape[0] == 0 or raw7.shape[1:] != (horizon, 7):
        raise ValueError(
            f"Expected raw LIBERO actions [B,{horizon},7], got {tuple(raw7.shape)}."
        )
    pad = torch.as_tensor(batch["actions_is_pad"])
    if pad.dtype != torch.bool or pad.shape != raw7.shape[:2]:
        raise ValueError(
            f"Expected actions_is_pad bool {tuple(raw7.shape[:2])}, "
            f"got shape={tuple(pad.shape)} dtype={pad.dtype}."
        )
    pad_cpu = pad.cpu()
    if pad_cpu[:, 0].any() or (pad_cpu[:, :-1] & ~pad_cpu[:, 1:]).any():
        raise ValueError("LeRobot action windows must have a non-empty valid prefix.")
    pad = pad.to(device=raw7.device)
    if not torch.isfinite(raw7[~pad]).all():
        raise ValueError("Non-padded expert actions must be finite.")
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
    if normalized.shape != raw10.shape or not torch.isfinite(normalized).all():
        raise ValueError("LIBERO normalization must return finite [B,H,10] actions.")
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
    batch_size = raw7.shape[0]
    modes = ["policy"] * batch_size
    vision_kwargs = {}
    if include_vision:
        missing = {"vision_images", "vision_valid"} - batch.keys()
        if missing:
            raise ValueError(
                f"Joint Cosmos SFT batch is missing fields: {sorted(missing)}."
            )
        vision_valid = torch.as_tensor(batch["vision_valid"], device="cpu")
        if vision_valid.shape != (batch_size,) or vision_valid.dtype != torch.bool:
            raise ValueError(f"Expected vision_valid bool [{batch_size}].")
        if (vision_valid & pad_cpu.any(dim=1)).any():
            raise ValueError(
                "A complete vision clip cannot contain padded action steps."
            )
        modes = ["wam" if valid else "policy" for valid in vision_valid.tolist()]
        vision_kwargs = {
            "vision_images": batch["vision_images"],
            "vision_wrist_images": batch.get("vision_wrist_images"),
            "vision_valid": vision_valid,
        }

    prepared, forward_inputs = prepare_observation_batch(observations, **vision_kwargs)
    data = dict(prepared)
    if data.get("mode") != modes or any(
        len(data.get(key, [])) != batch_size
        for key in ("sequence_plan", video_key, "raw_action_dim")
    ):
        raise ValueError("Expected a Cosmos batch with matching sample modes and size.")
    for videos, plan, raw_dim, mode in zip(
        data[video_key],
        data["sequence_plan"],
        data["raw_action_dim"],
        modes,
        strict=True,
    ):
        num_frames = horizon + 1 if mode == "wam" else 1
        if (
            len(videos) != 1
            or not isinstance(videos[0], torch.Tensor)
            or videos[0].ndim != 4
            or videos[0].shape[1] != num_frames
            or plan.condition_frame_indexes_vision != [0]
        ):
            raise ValueError(
                f"Cosmos {mode} training requires {num_frames} frames with "
                "the first frame conditioned."
            )
        if plan.condition_frame_indexes_action:
            raise ValueError("LeRobot action windows must not contain action history.")
        if raw_dim is None or int(raw_dim) != 10:
            raise ValueError("Cosmos LIBERO policy requires raw_action_dim=10.")

    tokens = forward_inputs["tokenized_prompt"]
    token_mask = forward_inputs["tokenized_prompt_mask"]
    if (
        tokens.ndim != 2
        or tokens.shape[0] != batch_size
        or tokens.dtype not in (torch.int32, torch.int64)
        or token_mask.dtype != torch.bool
        or token_mask.shape != tokens.shape
    ):
        raise ValueError(
            "Expected integer prompt tokens and a matching [B,L] bool mask."
        )
    if not token_mask.any(dim=1).all() or (tokens[token_mask] < 0).any():
        raise ValueError("Each Cosmos prompt needs non-empty, non-negative token IDs.")

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
