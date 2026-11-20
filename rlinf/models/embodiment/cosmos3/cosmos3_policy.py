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

"""Cosmos3 action-policy wrapper for RLinf SFT and native rollout."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

logger = get_logger()


class Cosmos3Policy(nn.Module, BasePolicy):
    """RLinf adapter around a cosmos ``OmniMoTModel``."""

    _no_split_modules: list[str] = [
        "MoTDecoderLayer",
        "PackedAttentionMoT",
        "Cosmos3VFMNetwork",
    ]
    # cosmos DomainAwareLinear reads its embedding weights directly (grouped_mm).
    _fsdp_wrap_embeddings: bool = False

    def __init__(
        self,
        model: nn.Module,
        *,
        lr_multipliers: dict[str, float] | None = None,
        ema_enabled: bool = True,
        rollout_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.omni = model
        self.lr_multipliers: dict[str, float] = dict(lr_multipliers or {})
        self._ema_enabled = bool(ema_enabled) and bool(
            getattr(getattr(model, "config", None), "ema", None)
            and model.config.ema.enabled
        )
        self._global_step: int = 0
        self._rollout_config = dict(rollout_config or {})
        self._action_normalizer = None

    @staticmethod
    def get_weight_sync_param_names(module: nn.Module) -> list[str]:
        """Resolve Cosmos checkpoint-wrapper names in the outermost model state dict."""
        trainable = {
            id(param): name
            for name, param in module.named_parameters(remove_duplicate=False)
            if param.requires_grad
        }
        persistent_buffers: dict[int, str] = {}
        for module_name, submodule in module.named_modules(remove_duplicate=False):
            non_persistent_buffers = getattr(
                submodule, "_non_persistent_buffers_set", set()
            )
            for buffer_name, buffer in submodule.named_buffers(
                recurse=False, remove_duplicate=False
            ):
                if buffer_name not in non_persistent_buffers:
                    full_name = (
                        buffer_name
                        if not module_name
                        else f"{module_name}.{buffer_name}"
                    )
                    persistent_buffers[id(buffer)] = full_name

        # Checkpoint wrappers expose canonical state keys while traversal names include
        # wrapper modules. keep_vars preserves identity without copying tensor storage.
        state_dict = module.state_dict(keep_vars=True)
        state_ids = {id(value) for value in state_dict.values()}
        selected = trainable | persistent_buffers
        missing = selected.keys() - state_ids
        if missing:
            names = sorted(selected[identity] for identity in missing)
            raise RuntimeError(
                f"State dict does not preserve selected tensor identities: {names}"
            )
        trainable_names = [
            name for name, value in state_dict.items() if id(value) in trainable
        ]
        buffer_names = [
            name
            for name, value in state_dict.items()
            if id(value) in persistent_buffers and id(value) not in trainable
        ]
        return trainable_names + buffer_names

    @property
    def device(self) -> torch.device:
        return next(self.omni.parameters()).device

    def set_global_step(self, step: int, *, update_ema: bool = True) -> None:
        """Track optimizer steps; restoration and rollout versions can skip EMA."""
        step = int(step)
        if update_ema and self._ema_enabled and step == self._global_step + 1:
            self._update_ema(self._global_step)
        self._global_step = step

    @torch.no_grad()
    def _update_ema(self, iteration: int) -> None:
        model = self.omni
        if not getattr(model, "config", None) or not model.config.ema.enabled:
            return
        net = getattr(model, "net", None)
        net_ema = getattr(model, "net_ema", None)
        worker = getattr(model, "net_ema_worker", None)
        if net is None or net_ema is None or worker is None:
            return
        if getattr(model, "_uses_aux_loss_free_load_balancing", False) or getattr(
            model, "_uses_ema_router_bias", False
        ):
            for sync_fn in ("sync_expert_biases_to_ema", "sync_router_biases_to_ema"):
                fn = getattr(model, sync_fn, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - best-effort buffer sync
                        pass
        beta = model.ema_beta(iteration)
        worker.update_average(net, net_ema, beta=beta)

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"Cosmos3Policy supports ForwardType.SFT only, got {forward_type}."
        )

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute the native SFT loss for one batch."""
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError(
                "Cosmos3 sft_forward requires `data` from the SFT dataloader."
            )

        from cosmos_framework.utils.misc import to as _cosmos_to

        # Native packing reads these fields as Python integers and token lists.
        data = {
            key: value
            if key in {"text_token_ids", "image_size"}
            else _cosmos_to(value, self.device)
            for key, value in data.items()
        }

        if not getattr(self, "_vae_patch_applied", False):
            self._vae_patch_applied = True
            _orig_fn = type(self.omni)._encode_vision_item

            def _patched(omni_self, state, *args, **kwargs):
                if hasattr(state, "device") and state.device != self.device:
                    state = state.to(self.device)
                return _orig_fn(omni_self, state, *args, **kwargs)

            import functools

            self.omni._encode_vision_item = functools.partial(_patched, self.omni)
        output_batch, loss = self.omni.training_step(data, self._global_step)

        result: dict[str, Any] = {"loss": loss}
        if isinstance(output_batch, dict):
            for key, value in output_batch.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value) and value.numel() == 1:
                    result[key] = value.detach()
        return result

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

    def prepare_lerobot_sft_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare expert actions and optional observed video for online SFT."""
        from rlinf.data.datasets.cosmos3.lerobot_batch import prepare_lerobot_batch

        return prepare_lerobot_batch(
            batch,
            horizon=int(self._rollout_config.get("num_frames", 17)) - 1,
            max_action_dim=int(self.omni.config.max_action_dim),
            normalizer=self._get_action_normalizer(),
            prepare_observation_batch=self._prepare_sft_observation_batch,
            video_key=self.omni.input_video_key,
            include_vision=self._rollout_config.get("sft_objective") == "joint",
        )

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        return None

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "Cosmos3Policy.default_forward is not implemented; native rollout "
            "uses predict_action_batch()."
        )

    @staticmethod
    def _rollout_images(
        images: Any, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Convert one batched RGB camera stream to uint8 BHWC."""
        if isinstance(images, (list, tuple)):
            images = torch.stack([torch.as_tensor(image) for image in images])
        images = torch.as_tensor(images).detach().to(device)
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.shape[-1] != 3 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        if images.is_floating_point() and images.amax() <= 1:
            images = images * 255
        return images.to(torch.uint8).contiguous()

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
        cameras = [self._rollout_images(main)]
        if env_obs.get("wrist_images") is not None:
            cameras.append(self._rollout_images(env_obs["wrist_images"]))
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
            [self._resize_camera_frames(camera, image_size) for camera in cameras],
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
                caption_data["additional_view_description"] = (
                    "The left half shows the third-person view; the right half "
                    "shows the wrist-mounted camera."
                )
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

    @staticmethod
    def _resize_camera_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
        frames = frames.permute(0, 3, 1, 2)
        if frames.shape[-2:] == (image_size, image_size):
            return frames
        return F.interpolate(
            frames.float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).to(torch.uint8)

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
