# Copyright 2025 The RLinf Authors.
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

"""ABot-M0 policy wrapper for RLinf embodied RL.

Provides rollout action generation and training-time logprob recomputation
through the `BasePolicy` interface.
"""

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class ABotM0ForRLActionPrediction(nn.Module, BasePolicy):
    """ABot-M0 wrapper with RL-specific action/value interfaces."""

    _no_split_modules = [
        "VGGT",
        "AMLFlowMatchingActionHeadRL",
        "ValueHead",
    ]

    def __init__(
        self,
        base_model: nn.Module,
        rl_head_config: dict[str, Any],
        image_size: list[int],
        num_action_chunks: int,
        qwen_max_length: int = 256,
    ):
        nn.Module.__init__(self)

        self.base_model = base_model
        self.rl_config = rl_head_config
        self.image_size = image_size
        self.num_action_chunks = num_action_chunks
        self.qwen_max_length = int(qwen_max_length)

        vl_hidden_size = self.base_model.qwen_vl_interface.model.config.hidden_size
        self.vl_hidden_size = vl_hidden_size
        self.action_dim = self.base_model.action_model.action_dim
        self.action_horizon = self.base_model.action_model.action_horizon

        from rlinf.models.embodiment.abot_m0.action_head_rl import (
            AMLFlowMatchingActionHeadRL,
        )

        self.action_head_rl = AMLFlowMatchingActionHeadRL(
            base_action_head=self.base_model.action_model,
            rl_head_config=rl_head_config,
            vl_hidden_size=vl_hidden_size,
        )

        if hasattr(self.base_model, "spatial_model"):
            for param in self.base_model.spatial_model.parameters():
                param.requires_grad = False

    def _get_runtime_action_dim(self) -> int:
        """Infer the runtime action dimension from checkpoint statistics."""
        runtime_action_dim = int(self.action_dim)

        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            return runtime_action_dim

        unnorm_key = next(iter(self.norm_stats.keys()))
        action_stats = self.norm_stats.get(unnorm_key, {}).get("action", {})
        stats_dims = action_stats.get("q01")
        if stats_dims is None:
            return runtime_action_dim

        stats_action_dim = int(len(stats_dims))
        if stats_action_dim > 0:
            runtime_action_dim = min(runtime_action_dim, stats_action_dim)

        return runtime_action_dim

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        rl_head_config: dict[str, Any],
        image_size: list[int] = (224, 224),
        num_action_chunks: int = 5,
        denoising_steps: Optional[int] = None,
        qwen_max_length: int = 256,
        **kwargs,
    ) -> "ABotM0ForRLActionPrediction":
        """Load ABot-M0 checkpoint and attach RL action head."""
        from ABot.model.framework.base_framework import baseframework

        base_model = baseframework.from_pretrained(pretrained_checkpoint)

        if denoising_steps is not None:
            base_model.action_model.num_inference_timesteps = denoising_steps

        model = cls(
            base_model=base_model,
            rl_head_config=rl_head_config,
            image_size=list(image_size),
            num_action_chunks=num_action_chunks,
            qwen_max_length=qwen_max_length,
        )

        if hasattr(base_model, "norm_stats"):
            model.norm_stats = base_model.norm_stats

        return model

    def _encode_observations(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        spatial_images: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode multimodal observations into fused latent embeddings."""
        from ABot.model.modules.vggt_tools import preprocess_images

        if device is not None:
            qwen_inputs = {
                k: v.to(device=device).contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in qwen_inputs.items()
            }
            spatial_images = spatial_images.to(device=device).contiguous()

        batch_images = self._image_tensor_to_pil_batch(spatial_images)

        if device is not None:
            for k, v in qwen_inputs.items():
                if isinstance(v, torch.Tensor):
                    qwen_inputs[k] = v.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.base_model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            with torch.no_grad():
                spatial_input = preprocess_images(
                    batch_images, batch_images[0][0].size[0],
                ).to(last_hidden.device)
                aggregated_tokens_list, ps_idx = self.base_model.spatial_model.aggregator(
                    spatial_input,
                )
            spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :]
            spatial_tokens = self.base_model.spatial_projector(spatial_tokens)
            last_hidden = self.base_model.fuser(last_hidden, spatial_tokens)

        state_tensor = None
        if state is not None:
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).to(
                    last_hidden.device, dtype=last_hidden.dtype,
                )
            elif isinstance(state, torch.Tensor):
                state_tensor = state.to(last_hidden.device, dtype=last_hidden.dtype)

            if state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(1)

        return last_hidden, state_tensor

    def _prepare_model_inputs(
        self,
        batch_images: list[list[Image.Image]],
        instructions: list[str],
        state: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        from ABot.training.trainer_utils.trainer_tools import resize_images

        if self.image_size:
            batch_images = resize_images(batch_images, target_size=self.image_size)

        qwen_inputs = self.base_model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            max_length=self.qwen_max_length,
        )
        qwen_inputs = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in qwen_inputs.items()
        }

        spatial_images = self._pil_batch_to_tensor(batch_images)

        if state is not None and state.ndim == 2:
            state = state.unsqueeze(1)

        return qwen_inputs, spatial_images, state

    def _flatten_qwen_inputs(
        self, qwen_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {f"qwen_{key}": value for key, value in qwen_inputs.items()}

    def _extract_qwen_inputs(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        qwen_inputs = {}
        for key, value in forward_inputs.items():
            if key.startswith("qwen_"):
                qwen_inputs[key.removeprefix("qwen_")] = value
        return qwen_inputs

    def _pil_batch_to_tensor(
        self,
        batch_images: list[list[Image.Image]],
    ) -> torch.Tensor:
        image_batches = []
        for views in batch_images:
            image_batches.append(
                np.stack([np.asarray(image, dtype=np.uint8) for image in views], axis=0)
            )
        return torch.from_numpy(np.stack(image_batches, axis=0))

    def _image_tensor_to_pil_batch(
        self,
        image_tensor: torch.Tensor,
    ) -> list[list[Image.Image]]:
        images_raw = image_tensor.detach().cpu().numpy()
        batch_images = []
        for batch_idx in range(images_raw.shape[0]):
            views = []
            for view_idx in range(images_raw.shape[1]):
                img = images_raw[batch_idx, view_idx]
                if img.dtype != np.uint8:
                    img = img.clip(0, 255).astype(np.uint8)
                views.append(Image.fromarray(img))
            batch_images.append(views)
        return batch_images

    def _normalize_state_for_abot(self, state: torch.Tensor) -> torch.Tensor:
        """Map runtime state to ABot-M0 expected 7D state layout."""
        if state.ndim == 2:
            state = state.unsqueeze(1)

        state_dim = state.shape[-1]
        if state_dim == 7:
            return state

        if state_dim > 7:
            pose = state[..., :6]
            gripper = state[..., 6:7]
            return torch.cat([pose, gripper], dim=-1)

        raise ValueError(
            f"ABot-M0 expects >=7 state dims, but got {state_dim}. "
            "Expected [x,y,z,ax,ay,az,gripper]."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Forward type {forward_type} not supported for ABot-M0")

    def default_forward(
        self,
        forward_inputs: dict[str, Any],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Recompute rollout logprobs and values during training."""
        if "vl_embs" in forward_inputs:
            vl_embs = forward_inputs["vl_embs"].to(
                forward_inputs["chains"].device,
            )
            state_tensor = forward_inputs.get("state")
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.from_numpy(state_tensor)
            if isinstance(state_tensor, torch.Tensor):
                state_tensor = state_tensor.to(
                    vl_embs.device,
                    dtype=vl_embs.dtype,
                )
                if state_tensor.ndim == 2:
                    state_tensor = state_tensor.unsqueeze(1)
        else:
            vl_embs, state_tensor = self._encode_observations(
                qwen_inputs=self._extract_qwen_inputs(forward_inputs),
                spatial_images=forward_inputs["main_images"],
                state=forward_inputs.get("state"),
                device=forward_inputs["chains"].device,
            )

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        with torch.autocast("cuda", dtype=torch.float32):
            log_probs, values = self.action_head_rl(
                vl_embs=vl_embs,
                state=state_tensor,
                chains=chains,
                denoise_inds=denoise_inds,
                compute_values=compute_values,
            )

        runtime_action_dim = self._get_runtime_action_dim()
        log_probs = log_probs[:, :, : self.num_action_chunks, :runtime_action_dim]

        if self.action_head_rl.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.num_action_chunks,
                :runtime_action_dim,
            ]

        values = values.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": values,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate rollout actions and cache tensors needed by training."""
        batch_images, instructions, state = self._convert_env_obs(env_obs)
        qwen_inputs, spatial_images, state = self._prepare_model_inputs(
            batch_images=batch_images,
            instructions=instructions,
            state=state,
        )

        vl_embs, state_tensor = self._encode_observations(
            qwen_inputs=qwen_inputs,
            spatial_images=spatial_images,
            state=state,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions, rl_outputs = self.action_head_rl.get_rl_action(
                vl_embs=vl_embs,
                state=state_tensor,
                mode=mode,
            )

        runtime_action_dim = self._get_runtime_action_dim()

        normalized_actions = actions.detach().cpu().numpy()
        normalized_actions = normalized_actions[..., :runtime_action_dim]
        raw_actions = self._unnormalize_actions(normalized_actions)
        raw_actions = torch.from_numpy(raw_actions).to(actions.device, dtype=actions.dtype)

        raw_actions = raw_actions[:, : self.num_action_chunks, :]
        prev_logprobs = rl_outputs["prev_logprobs"][
            :, :, : self.num_action_chunks, :runtime_action_dim
        ]

        forward_inputs = {
            "chains": rl_outputs["chains"],
            "denoise_inds": rl_outputs["denoise_inds"],
            "vl_embs": vl_embs,
            "main_images": spatial_images,
            "state": state,
        }

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": rl_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return raw_actions, result

    def _convert_env_obs(
        self, env_obs: dict[str, Any],
    ) -> tuple[list[list[Image.Image]], list[str], Optional[torch.Tensor]]:
        """Convert RLinf env observations to ABot-M0 model inputs."""
        instructions = env_obs["task_descriptions"]

        if "main_images" in env_obs:
            images_raw = env_obs["main_images"]
            if isinstance(images_raw, torch.Tensor):
                images_raw = images_raw.cpu().numpy()
            elif not isinstance(images_raw, np.ndarray):
                images_raw = np.asarray(images_raw)

            batch_images = []
            if images_raw.ndim == 5:  # [B, V, H, W, C]
                for b in range(images_raw.shape[0]):
                    views = []
                    for v in range(images_raw.shape[1]):
                        img = images_raw[b, v]
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        views.append(Image.fromarray(img))
                    batch_images.append(views)
            elif images_raw.ndim == 4:  # [B, H, W, C] single view
                for b in range(images_raw.shape[0]):
                    img = images_raw[b]
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    batch_images.append([Image.fromarray(img)])
            else:
                raise ValueError(f"Unsupported main_images shape: {images_raw.shape}")
        else:
            batch_images = env_obs.get("images", [[]])

        state = None
        if "states" in env_obs:
            state = env_obs["states"]
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state)
            state = self._normalize_state_for_abot(state)

        return batch_images, instructions, state

    def _unnormalize_actions(
        self, normalized_actions: np.ndarray,
    ) -> np.ndarray:
        """De-normalize actions using checkpoint normalization statistics."""
        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            return normalized_actions

        unnorm_key = next(iter(self.norm_stats.keys()))
        action_stats = self.norm_stats[unnorm_key]["action"]

        runtime_action_dim = self._get_runtime_action_dim()
        normalized_actions = normalized_actions[..., :runtime_action_dim]

        from ABot.model.framework.base_framework import baseframework

        action_shape = normalized_actions.shape
        if normalized_actions.ndim != 3:
            raise ValueError(
                f"Expected normalized_actions with shape [B, T, D], got {action_shape}",
            )

        _, _, action_dim = action_shape
        flattened_actions = normalized_actions.reshape(-1, action_dim)
        unnormalized_flattened = baseframework.unnormalize_actions(
            flattened_actions, action_stats,
        )

        return unnormalized_flattened.reshape(action_shape)
