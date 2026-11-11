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

"""LIBERO observation and action transforms shared by WAM PPO and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class Cosmos3EnvIO:
    """Translate tensor-only replay observations to native Cosmos WAM batches."""

    _task_bytes_limit = 4096

    @staticmethod
    def _images(images: Any) -> torch.Tensor:
        if isinstance(images, (list, tuple)):
            images = torch.stack([torch.as_tensor(image) for image in images])
        images = torch.as_tensor(images).detach().cpu()
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError(
                f"Expected nonempty RGB [B,H,W,3] or [B,3,H,W], got {images.shape}."
            )
        if images.shape[-1] != 3 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        if images.shape[-1] != 3 or not torch.isfinite(images).all():
            raise ValueError("Expected finite three-channel RGB images.")
        if images.min() < 0 or images.max() > 255:
            raise ValueError("Expected RGB pixels in [0,255].")
        if images.is_floating_point() and images.max() <= 1:
            images = images * 255
        return images.to(torch.uint8).contiguous()

    def replay_observation(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Store exact task strings and cameras in batch-first replay tensors."""
        image = env_obs.get("main_images", env_obs.get("images"))
        if image is None:
            raise ValueError("WAM PPO requires env_obs['main_images'].")
        image = self._images(image)
        batch_size = image.shape[0]
        tasks = env_obs.get("task_descriptions", env_obs.get("task"))
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        if tasks is None or len(tasks) != batch_size:
            raise ValueError(f"Expected {batch_size} task descriptions.")
        task_bytes = torch.zeros(batch_size, self._task_bytes_limit, dtype=torch.uint8)
        task_lengths = torch.empty(batch_size, dtype=torch.long)
        for index, task in enumerate(tasks):
            if not isinstance(task, str) or not task.strip():
                raise ValueError("WAM PPO requires nonempty task descriptions.")
            encoded = task.encode("utf-8")
            if len(encoded) > self._task_bytes_limit:
                raise ValueError(f"Task exceeds {self._task_bytes_limit} UTF-8 bytes.")
            task_bytes[index, : len(encoded)] = torch.tensor(
                list(encoded), dtype=torch.uint8
            )
            task_lengths[index] = len(encoded)
        result = {
            "observation/image": image,
            "task_bytes": task_bytes,
            "task_lengths": task_lengths,
        }
        if env_obs.get("wrist_images") is not None:
            wrist = self._images(env_obs["wrist_images"])
            if wrist.shape[0] != batch_size:
                raise ValueError(
                    "Main and wrist cameras must have matching batch sizes."
                )
            result["observation/wrist_image"] = wrist
        return result

    def native_batch(self, replay: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Build the same WAM prompt and image conditioning for rollout and update."""
        from cosmos_framework.data.generator.action.utils.json_formatter import (
            ActionPromptJsonFormatter,
        )
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode

        cfg = self.model_cfg
        cameras = [replay["observation/image"].detach().cpu()]
        if "observation/wrist_image" in replay:
            cameras.append(replay["observation/wrist_image"].detach().cpu())
        image_size = int(cfg.get("image_size", 256))
        images = torch.cat(
            [
                F.interpolate(
                    image.permute(0, 3, 1, 2).float(),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).to(torch.uint8)
                for image in cameras
            ],
            dim=-1,
        )
        encoded = replay["task_bytes"].detach().cpu()
        lengths = replay["task_lengths"].detach().cpu()
        tasks = [
            bytes(row[: int(length)].tolist()).decode("utf-8")
            for row, length in zip(encoded, lengths)
        ]
        viewpoint = "concat_view" if len(cameras) == 2 else "third_person_view"
        horizon = int(cfg.get("num_frames", 17)) - 1
        formatter = ActionPromptJsonFormatter()
        items = []
        for image, task in zip(images, tasks):
            item = build_action_batch(
                video=image.unsqueeze(1),
                action=torch.zeros(horizon, int(self.omni.config.max_action_dim)),
                raw_action_dim=10,
                prompt=task,
                view_point=viewpoint,
                domain_name="libero",
                model_mode=ModelMode.WAM,
                action_chunk_size=horizon,
                fps=int(cfg.get("fps", 20)),
                resolution=cfg.get("resolution"),
                input_video_key=self.omni.input_video_key,
                device=self.device,
            )
            caption_data = {
                "ai_caption": task,
                "viewpoint": viewpoint,
                "video": item[self.omni.input_video_key][0][0],
                "image_size": item["image_size"][0],
                "conditioning_fps": item["conditioning_fps"][0],
                "action": item["action"][0][0],
                "idle_frames": torch.tensor(0),
                "mode": "wam",
            }
            if len(cameras) == 2:
                caption_data["additional_view_description"] = (
                    "The left half shows the third-person view; the right half "
                    "shows the wrist-mounted camera."
                )
            caption = json.dumps(formatter(caption_data)["ai_caption"])
            item[self.omni.input_caption_key] = [caption]
            item["prompt"] = [caption]
            item[self.omni.input_video_key][0][0] = item[self.omni.input_video_key][0][
                0
            ].to(self.device)
            items.append(item)
        batch = {
            key: [item[key][0] for item in items]
            for key in items[0]
            if key != "image_size"
        }
        batch["image_size"] = torch.cat([item["image_size"] for item in items])
        return batch

    def decode_actions(self, normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize LIBERO rot6d actions and execute only the configured prefix."""
        from cosmos_framework.data.generator.action.utils import action_processing
        from cosmos_framework.data.generator.action.utils.pose_utils import (
            convert_rotation,
        )

        if self._action_normalizer is None:
            stats = self.model_cfg.get("action_stats_path") or (
                Path(action_processing.__file__).parents[1]
                / "normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
            )
            self._action_normalizer = action_processing.load_action_normalizer(
                "quantile_rot",
                stats_path=stats,
                stats_key="global_raw",
                apply_forward_clamp=False,
                expected_dim=10,
            )
        raw = self._action_normalizer.denormalize_action(normalized[..., :10])
        rotation = convert_rotation(
            raw[..., 3:9], input_format="rot6d", output_format="axisangle"
        )
        actions = torch.cat([raw[..., :3], rotation, raw[..., 9:10]], dim=-1)
        return actions[:, : int(self.model_cfg.num_action_chunks)].contiguous()
