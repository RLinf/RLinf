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

"""Environment observation packing, native sampling and action decoding for Cosmos3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.cosmos3.transforms import (
    CONCAT_VIEW_DESCRIPTION,
    as_uint8_bhwc,
    resize_camera_frames,
)


class Cosmos3EnvIO:
    """Handle native rollout I/O and provide the policy's LIBERO action normalizer."""

    def _get_action_normalizer(self):
        """Resolve the shared, unclamped LIBERO rot6d training/inference stats."""
        from cosmos_framework.data.generator.action.utils import action_processing

        cfg = self._rollout_config
        if self._action_normalizer is None:
            default_stats = (
                Path(action_processing.__file__).parents[1]
                / "normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
            )
            self._action_normalizer = action_processing.load_action_normalizer(
                "quantile_rot",
                stats_path=cfg.get("action_stats_path") or default_stats,
                stats_key="global_raw",
                apply_forward_clamp=False,
                expected_dim=10,
            )
        return self._action_normalizer

    def _prepare_rollout_batch(
        self, env_obs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        from cosmos_framework.data.generator.action.utils.json_formatter import (
            ActionPromptJsonFormatter,
        )
        from cosmos_framework.data.generator.action.utils.transforms import (
            build_sequence_plan_from_mode,
        )
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode

        main = env_obs.get("main_images")
        if main is None:
            main = env_obs.get("images")
        cameras = [as_uint8_bhwc(main)]
        if env_obs.get("wrist_images") is not None:
            cameras.append(as_uint8_bhwc(env_obs["wrist_images"]))
        batch_size = cameras[0].shape[0]
        tasks = env_obs.get("task_descriptions")
        if tasks is None:
            tasks = env_obs.get("task")
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size

        cfg = self._rollout_config
        action_model_mode = cfg.get("action_model_mode", "wam")
        horizon = int(cfg.get("num_frames", 17)) - 1
        image_size = int(cfg.get("image_size", 256))
        fps = int(cfg.get("fps", 20))
        action_dim = int(self.omni.config.max_action_dim)
        images = torch.cat(
            [resize_camera_frames(camera, image_size) for camera in cameras],
            dim=-1,
        )
        viewpoint = "concat_view" if len(cameras) == 2 else "third_person_view"
        formatter = ActionPromptJsonFormatter()
        items = []
        for index, task in enumerate(tasks):
            item = build_action_batch(
                video=images[index].unsqueeze(1),
                action=torch.zeros(horizon, action_dim),
                raw_action_dim=10,
                prompt=task,
                view_point=viewpoint,
                domain_name="libero",
                model_mode=ModelMode.WAM,
                action_chunk_size=horizon,
                fps=fps,
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
                "mode": action_model_mode,
            }
            if len(cameras) == 2:
                caption_data["additional_view_description"] = CONCAT_VIEW_DESCRIPTION
            caption = json.dumps(formatter(caption_data)["ai_caption"])
            item[self.omni.input_caption_key] = [caption]
            item["prompt"] = [caption]
            if action_model_mode == "policy":
                # Keep horizon metadata in the prompt, but only encode observed pixels.
                item[self.omni.input_video_key] = [
                    [item[self.omni.input_video_key][0][0][:, :1].contiguous()]
                ]
                item["sequence_plan"] = [
                    build_sequence_plan_from_mode(
                        mode="policy", video_length=1, action_length=horizon
                    )
                ]
                item["mode"] = ["policy"]
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

        # Fixed-width text keeps replay tensors concatenable across rollout calls.
        tokens = self.omni.tokenize_text(
            batch[self.omni.input_caption_key], pad_token_id=-1, device=self.device
        )
        max_length = int(cfg.get("max_prompt_length", 1024))
        token_mask = tokens != -1
        forward_inputs = {
            "observation/image": cameras[0].clone(),
            "tokenized_prompt": F.pad(
                tokens.clamp_min(0), (0, max_length - tokens.shape[1])
            ),
            "tokenized_prompt_mask": F.pad(
                token_mask, (0, max_length - tokens.shape[1])
            ),
        }
        if len(cameras) == 2:
            forward_inputs["observation/wrist_image"] = cameras[1].clone()
        if env_obs.get("states") is not None:
            states = torch.as_tensor(env_obs["states"])
            forward_inputs["observation/state"] = states.detach().clone()
        return batch, forward_inputs

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        *,
        seed: int | list[int] | None = None,
        num_steps: int | None = None,
        guidance: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate LIBERO actions and replay inputs using the native Cosmos sampler.

        Args:
            env_obs: Batched RGB cameras, task descriptions and optional states.
            mode: Both train and eval use the same no-grad policy sampler.
            seed: One seed per sample, or an integer shared by the batch.
            num_steps: Override the configured diffusion integration steps.
            guidance: Override the configured classifier-free guidance scale.

        Returns:
            Environment actions [B,num_action_chunks,7] and OpenPI-compatible
            metadata. model_action contains normalized, padded rot6d actions.
        """
        del kwargs
        from cosmos_framework.data.generator.action.utils import action_processing
        from cosmos_framework.data.generator.action.utils.pose_utils import (
            convert_rotation,
        )

        cfg = self._rollout_config
        normalizer = self._get_action_normalizer()
        batch, forward_inputs = self._prepare_rollout_batch(env_obs)
        batch_size = len(batch["action"])
        horizon = batch["action"][0][0].shape[0]
        chunk = int(cfg.get("num_action_chunks", horizon))
        steps = int(
            num_steps if num_steps is not None else cfg.get("num_inference_steps", 30)
        )
        if seed is None:
            seed = cfg.get("inference_seed")
        if seed is None:
            seeds = torch.randint(0, 2**31, (batch_size,)).tolist()
        elif isinstance(seed, int):
            seeds = [seed] * batch_size
        else:
            seeds = list(seed)

        training_states = [(module, module.training) for module in self.omni.modules()]
        self.omni.eval()
        try:
            samples = self.omni.generate_samples_from_batch(
                batch,
                seed=seeds,
                num_steps=steps,
                guidance=float(
                    guidance if guidance is not None else cfg.get("guidance", 1.0)
                ),
                has_negative_prompt=False,
            )
        finally:
            for module, training in training_states:
                module.training = training
        predictions = samples["action"]
        normalized = []
        for prediction in predictions:
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]
            normalized.append(prediction.to(device=self.device, dtype=torch.float32))
        normalized = torch.stack(normalized)
        model_actions = action_processing.pad_action_to_max_dim(
            normalized, int(self.omni.config.max_action_dim)
        )
        raw_actions = normalizer.denormalize_action(normalized)
        rotation = convert_rotation(
            raw_actions[..., 3:9], input_format="rot6d", output_format="axisangle"
        )
        actions = torch.cat(
            [raw_actions[..., :3], rotation, raw_actions[..., 9:10]], dim=-1
        )[:, :chunk].contiguous()
        forward_inputs["action"] = actions.reshape(batch_size, -1).clone()
        forward_inputs["model_action"] = model_actions.reshape(batch_size, -1).clone()
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": forward_inputs,
            "model_actions": model_actions,
        }
