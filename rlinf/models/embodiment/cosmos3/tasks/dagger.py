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

import json
import math
import random
from collections.abc import Mapping
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


class Cosmos3DAggerInputs:
    """Prepare LeRobot training windows for the native Cosmos SFT loss."""

    def prepare_lerobot_sft_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare expert actions and optional observed video for online SFT."""
        from rlinf.models.embodiment.cosmos3.transforms.lerobot import (
            prepare_lerobot_batch,
        )

        return prepare_lerobot_batch(
            batch,
            horizon=int(self._rollout_config.get("num_frames", 17)) - 1,
            max_action_dim=int(self.omni.config.max_action_dim),
            normalizer=self._get_action_normalizer(),
            prepare_observation_batch=self._prepare_sft_observation_batch,
            video_key=self.omni.input_video_key,
            include_vision=self._rollout_config.get("sft_objective") == "joint",
        )

    def _prepare_sft_observation_batch(
        self,
        observations: dict[str, Any],
        *,
        vision_images: torch.Tensor | None = None,
        vision_wrist_images: torch.Tensor | None = None,
        vision_valid: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Build only the native training inputs, batching image transforms by mode."""
        from cosmos_framework.data.generator.action.utils.domain_utils import (
            get_domain_id,
        )
        from cosmos_framework.data.generator.action.utils.json_formatter import (
            ActionPromptJsonFormatter,
        )
        from cosmos_framework.data.generator.action.utils.transforms import (
            build_sequence_plan_from_mode,
            find_closest_target_size,
            reflection_pad_to_target,
        )
        from cosmos_framework.data.generator.augmentors.text_tokenizer import (
            _MAX_NUM_TOKENS,
        )
        from cosmos_framework.utils.generator.data_utils import (
            get_vision_data_resolution,
        )

        cfg = self._rollout_config
        horizon = int(cfg.get("num_frames", 17)) - 1
        image_size = int(cfg.get("image_size", 256))
        fps = int(cfg.get("fps", 20))

        cameras = [(observations["main_images"], vision_images)]
        wrist_images = observations.get("wrist_images")
        if wrist_images is not None:
            cameras.append((wrist_images, vision_wrist_images))
        batch_size = len(cameras[0][0])
        tasks = observations["task_descriptions"]
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size

        valid = (
            vision_valid.tolist() if vision_valid is not None else [False] * batch_size
        )
        modes = ["wam" if has_clip else "policy" for has_clip in valid]
        viewpoint = "concat_view" if len(cameras) == 2 else "third_person_view"
        formatter = ActionPromptJsonFormatter()
        fps_tensor = torch.tensor(fps, dtype=torch.long)
        domain_id = torch.tensor(get_domain_id("libero"), dtype=torch.long)
        raw_actions = observations.get("raw_actions")
        action_padding = observations.get("actions_is_pad")
        idle_frames = [0] * batch_size
        if raw_actions is not None:
            from cosmos_framework.data.generator.action.utils.action_spec import (
                Gripper,
                Pos,
                Rot,
                build_action_spec,
            )
            from cosmos_framework.data.generator.action.utils.pose_utils import (
                compute_idle_frames,
            )

            raw_actions = torch.as_tensor(raw_actions).detach().cpu()
            spec = build_action_spec(Pos(), Rot("rot6d"), Gripper())
            # Match LIBEROLeRobotDataset._compute_idle_frames on physical actions.
            for index, action in enumerate(raw_actions):
                if action_padding is not None:
                    action = action[~action_padding[index]]
                idle_frames[index] = compute_idle_frames(
                    action,
                    spec,
                    eps_t=5e-3 / fps,
                    eps_r=math.radians(1.5) / fps,
                    eps_g=1e-2,
                    joint_threshold=5e-3 / fps,
                    min_streak=3,
                )
        dropout_rate = float(cfg.get("sft_cfg_dropout_rate", 0.1))
        prepared = {}
        for mode in ("wam", "policy"):
            indices = [
                index for index, item_mode in enumerate(modes) if item_mode == mode
            ]
            if not indices:
                continue
            frame_count = horizon + 1 if mode == "wam" else 1
            views = []
            for current, clip in cameras:
                if mode == "wam":
                    pixels = torch.as_tensor(clip)
                    pixels = pixels.to(self.device, non_blocking=True)
                    if len(indices) != batch_size:
                        pixels = pixels[indices]
                    frames = self._rollout_images(
                        pixels.flatten(0, 1), device=self.device
                    )
                    del pixels
                else:
                    frames = self._rollout_images(current, device=self.device)[indices]
                views.append(self._resize_camera_frames(frames, image_size))

            video = torch.cat(views, dim=-1) if len(views) > 1 else views[0]
            del views, frames
            height, width = video.shape[-2:]
            resolution = cfg.get("resolution")
            if resolution is None:
                resolution = get_vision_data_resolution((height, width))
            target_w, target_h = find_closest_target_size(height, width, resolution)
            resized = reflection_pad_to_target(
                {"video": video}, ["video"], True, target_w, target_h
            )
            video = (
                resized["video"]
                .unflatten(0, (len(indices), frame_count))
                .transpose(1, 2)
                .contiguous()
            )
            for index, sample in zip(indices, video, strict=True):
                caption_data = {
                    "ai_caption": random.choice(
                        [
                            part.strip()
                            for part in tasks[index].split(" | ")
                            if part.strip()
                        ]
                        or [tasks[index]]
                    ),
                    "viewpoint": viewpoint,
                    "video": sample,
                    "video_num_frames": horizon + 1,
                    "image_size": resized["image_size"],
                    "conditioning_fps": fps_tensor,
                    "idle_frames": idle_frames[index],
                    "idle_frames_total": horizon,
                    "mode": mode,
                }
                if len(cameras) == 2:
                    caption_data["additional_view_description"] = (
                        "The left half shows the third-person view; the right half "
                        "shows the wrist-mounted camera."
                    )
                caption = json.dumps(formatter(caption_data)["ai_caption"])
                if dropout_rate > 0 and random.random() < dropout_rate:
                    caption = ""
                plan = build_sequence_plan_from_mode(
                    mode=mode, video_length=frame_count, action_length=horizon
                )
                prepared[index] = (sample, resized["image_size"], plan, caption)

        videos, sizes, plans, captions = zip(
            *(prepared[i] for i in range(batch_size)), strict=True
        )
        tokens = self.omni.tokenize_text(
            list(captions),
            pad_token_id=-1,
            device="cpu",
            is_video=False,
            use_system_prompt=False,
        )
        tokens = tokens[:, :_MAX_NUM_TOKENS]
        return {
            self.omni.input_video_key: [[video] for video in videos],
            "image_size": torch.stack(sizes),
            "sequence_plan": list(plans),
            "mode": modes,
            "raw_action_dim": [torch.tensor(10, dtype=torch.long)] * batch_size,
            "conditioning_fps": [fps_tensor] * batch_size,
            "domain_id": [domain_id] * batch_size,
        }, {"tokenized_prompt": tokens, "tokenized_prompt_mask": tokens != -1}
