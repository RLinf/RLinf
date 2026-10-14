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

import math
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils._pytree import tree_map
from torchvision.transforms.v2 import Resize
from transformers import AutoConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor

ROBOTWIN_MODEL_TO_ENV_ACTION_INDICES = list(range(6)) + [14] + list(range(6, 12)) + [15]
DEFAULT_RL_TRAINABLE_SCOPE = "action_expert"


def _denoise_timesteps(num_steps: int, device) -> torch.Tensor:
    """Return the upstream Euler schedule, accumulated in bf16 as it samples.

    Upstream steps ``t += dt`` with bf16 ``t`` and ``dt = -1 / num_steps``, so
    the grid drifts from ``linspace`` and ends slightly below zero.
    """
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.bfloat16)
    t = torch.tensor(1.0, dtype=torch.bfloat16)
    timesteps = []
    for _ in range(num_steps + 1):
        timesteps.append(t.item())
        t = t + dt
    return torch.tensor(timesteps, dtype=torch.float32, device=device)


@dataclass
class Observation:
    image: Any
    state: Any
    prompt: Optional[Any] = None
    wrist_images: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            image=data.get("image"),
            state=data.get("state"),
            prompt=data.get("prompt"),
            wrist_images=data.get("wrist_images"),
        )


def _require_lingbot_v2():
    try:
        from lingbotvla.data.vla_data.utils import FeatureTransform
        from lingbotvla.distributed.parallel_state import init_parallel_state
        from lingbotvla.models import build_processor
        from lingbotvla.models.module_utils import load_model_weights
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
            LingbotVLAV2Config,
        )
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
            LingbotVlaV2Policy,
            make_att_2d_masks,
        )
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
            apply_lingbot_qwen3_vl_patch,
        )
    except ImportError as exc:
        raise ImportError(
            "LingBot-VLA V2 support requires the upstream "
            "`robbyant/lingbot-vla-v2` package. Install with "
            "`bash requirements/install.sh embodied --model lingbotvla_v2 "
            "--env robotwin` or set LINGBOT_VLA_V2_PATH to an editable checkout."
        ) from exc

    return SimpleNamespace(
        FeatureTransform=FeatureTransform,
        LingbotVLAV2Config=LingbotVLAV2Config,
        LingbotVlaV2Policy=LingbotVlaV2Policy,
        apply_lingbot_qwen3_vl_patch=apply_lingbot_qwen3_vl_patch,
        build_processor=build_processor,
        init_parallel_state=init_parallel_state,
        load_model_weights=load_model_weights,
        make_att_2d_masks=make_att_2d_masks,
    )


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _as_literal_list(entries):
    converted = []
    for entry in entries or []:
        if isinstance(entry, str):
            converted.append(entry)
        else:
            converted.append(repr(entry))
    return converted


def _initialize_lingbot_parallel_state(v2):
    if not torch.distributed.is_initialized():
        return
    v2.init_parallel_state(
        dp_size=torch.distributed.get_world_size(),
        dp_mode="fsdp1",
    )


class LingbotVlaV2ActionModel(nn.Module, BasePolicy):
    """RLinf adapter for upstream LingBot-VLA V2."""

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "Qwen2DecoderLayer",
            "Qwen3VLTextDecoderLayer",
            "Qwen3VLVisionBlock",
            "FixQwen2RMSNorm",
            "AdaRMSNorm",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        names = [
            "visual",
            "embed_tokens",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "lm_head",
            "depth_align_embs",
            "value_head",
            "language_model_norm",
        ]
        if getattr(self.config, "rl_trainable_scope", DEFAULT_RL_TRAINABLE_SCOPE) == (
            "action_expert"
        ):
            names.extend(
                [
                    "action_time_mlp_in",
                    "action_time_mlp_out",
                    "time_mlp_in",
                    "time_mlp_out",
                    "qwen_expert_norm",
                ]
            )
        return names

    def __init__(self, config, torch_dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.logger = get_logger()
        self.global_step = 0
        self._v2 = _require_lingbot_v2()
        self._v2.apply_lingbot_qwen3_vl_patch()
        _initialize_lingbot_parallel_state(self._v2)

        lingbotvla_cfg = getattr(config, "lingbotvla_v2", config)
        model_path = getattr(config, "model_path", None)
        config_path = getattr(lingbotvla_cfg, "config_path", None) or model_path
        if not config_path:
            raise ValueError(
                "LingBot-VLA V2 requires actor.model.model_path or "
                "actor.model.lingbotvla_v2.config_path."
            )

        self.training_config = self._load_training_config(config_path, model_path)
        training_model_config = dict(self.training_config["model"])
        training_model_config.update(self.training_config["train"])
        for key in (
            "bias_update_speed",
            "sequence_wise_loss_coeff",
            "router_z_loss_coeff",
            "routed_scaling_factor",
        ):
            if key in training_model_config:
                training_model_config[key] = float(training_model_config[key])
        qwen_attention = getattr(config, "attn_implementation", "flash_attention_2")
        training_model_config["attn_implementation"] = qwen_attention
        training_model_config["_attn_implementation"] = qwen_attention
        moe_implementation = getattr(
            config,
            "moe_implementation",
            training_model_config.get("moe_implementation", "fused"),
        )
        training_model_config["_moe_implementation"] = moe_implementation
        training_model_config["moe_implementation"] = moe_implementation

        v2_config = self._v2.LingbotVLAV2Config(**training_model_config)
        for key, value in training_model_config.items():
            if not hasattr(v2_config, key):
                setattr(v2_config, key, value)
        v2_config.attention_implementation = getattr(
            config, "attention_implementation", "flex_cached"
        )
        v2_config.tokenizer_path = self._resolve_tokenizer_path(
            config, training_model_config
        )
        v2_config.loss_type = getattr(v2_config, "loss_type", "L1_fm")
        v2_config.use_cache = True
        v2_config.return_image_grid_thw = True
        v2_config.n_action_steps = int(
            getattr(v2_config, "chunk_size", getattr(config, "num_action_chunks", 50))
        )
        v2_config.num_steps = int(getattr(config, "num_steps", v2_config.num_steps))

        qwen_config = AutoConfig.from_pretrained(v2_config.tokenizer_path)
        self._merge_qwen_config(v2_config, qwen_config)
        v2_config.precompute_grid_thw = False
        if self.training_config["model"].get("vocab_size", 0) != 0:
            v2_config.vocab_size = self.training_config["model"]["vocab_size"]

        self.action_dim = int(
            getattr(v2_config, "action_dim", getattr(config, "action_dim", 55))
        )
        self.action_chunk = int(v2_config.n_action_steps)
        self.action_env_dim = int(getattr(config, "action_env_dim", 14))
        if not 0 < self.action_env_dim <= len(ROBOTWIN_MODEL_TO_ENV_ACTION_INDICES):
            raise ValueError(
                "LingBot-VLA V2 action_env_dim must be in the range "
                f"[1, {len(ROBOTWIN_MODEL_TO_ENV_ACTION_INDICES)}], got "
                f"{self.action_env_dim}."
            )
        self.num_steps = int(v2_config.num_steps)
        self.noise_method = getattr(config, "noise_method", "flow_sde")
        self.image_size = int(self.training_config.get("data", {}).get("img_size", 256))
        self._resize = Resize((self.image_size, self.image_size))

        self.vla_model = self._v2.LingbotVlaV2Policy(v2_config, eval=True).to(
            self.torch_dtype
        )
        # The alignment query tokens are part of the checkpoint's prefix and must
        # stay; only the auxiliary depth/video losses, which need teacher targets,
        # are switched off. Upstream gates the tokens on ``use_depth_align`` at
        # construction and the losses on ``config.align_params`` at forward time.
        if not getattr(lingbotvla_cfg, "use_auxiliary_alignment", False):
            self.vla_model.config.align_params = {}

        self.processor = self._v2.build_processor(v2_config.tokenizer_path)
        self.language_tokenizer = self.processor.tokenizer
        self.data_config = self._build_data_config(lingbotvla_cfg)
        robot_config_path = getattr(lingbotvla_cfg, "robot_config_path", None)
        if robot_config_path is None:
            robot_config_path = os.path.join(
                os.environ.get("LINGBOT_VLA_V2_PATH", ""),
                "configs/robot_configs/robotwin.yaml",
            )
        if not os.path.exists(robot_config_path):
            raise FileNotFoundError(
                "LingBot-VLA V2 robot config not found. Set "
                "actor.model.lingbotvla_v2.robot_config_path or "
                f"LINGBOT_VLA_V2_PATH. Checked: {robot_config_path}"
            )
        self.feature_transform = self._v2.FeatureTransform(
            robot_config_path,
            self.data_config,
            self.vla_model.config,
            self.processor,
            chunk_size=self.action_chunk,
            norm_stats_path=getattr(lingbotvla_cfg, "stats_path", None),
            use_depth_align=False,
            use_future_image=False,
        )

        if getattr(self.config, "add_value_head", False):
            self.value_head = ValueHead(
                input_dim=self.vla_model.model.config.proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            ).to(self.torch_dtype)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)
        self._mark_action_expert_fsdp_wrap_names()
        self._apply_rl_trainable_scope()

    def _resolve_tokenizer_path(self, config, training_model_config: dict) -> str:
        explicit = getattr(config, "tokenizer_path", None)
        if explicit:
            return explicit
        training_path = training_model_config.get("tokenizer_path")
        if training_path:
            return os.environ.get("QWEN3VL_PATH", training_path)
        env_path = os.environ.get("QWEN3VL_PATH")
        if env_path:
            return env_path
        raise ValueError(
            "LingBot-VLA V2 requires tokenizer_path or QWEN3VL_PATH "
            "(usually Qwen3-VL-4B-Instruct)."
        )

    def _build_data_config(self, lingbotvla_cfg):
        raw_data_config = dict(self.training_config["data"])
        raw_data_config["chunk_size"] = int(
            getattr(lingbotvla_cfg, "chunk_size", self.action_chunk)
        )
        raw_data_config["prompt_type"] = raw_data_config.get("prompt_type", "global")
        raw_data_config["img_size"] = raw_data_config.get("img_size", self.image_size)
        raw_data_config["joints"] = _as_literal_list(raw_data_config.get("joints", []))
        raw_data_config["norm_type"] = _as_literal_list(
            raw_data_config.get("norm_type", [])
        )
        raw_data_config["cameras"] = raw_data_config.get(
            "cameras",
            ["camera_top", "camera_wrist_left", "camera_wrist_right"],
        )
        return _namespace(raw_data_config)

    def _load_training_config(self, config_path: str, model_path: Optional[str]):
        candidate_paths = [
            os.path.join(config_path, "lingbotvla_cli.yaml"),
            os.path.join(config_path, "configs/vla/robotwin/robotwin.yaml"),
        ]
        if model_path:
            path = os.path.abspath(model_path)
            for _ in range(4):
                candidate_paths.append(os.path.join(path, "lingbotvla_cli.yaml"))
                path = os.path.dirname(path)

        training_config_path = next(
            (path for path in candidate_paths if os.path.exists(path)), None
        )
        if training_config_path is None:
            raise FileNotFoundError(
                "LingBot-VLA V2 requires lingbotvla_cli.yaml or the upstream "
                f"robotwin config. Checked: {candidate_paths}"
            )

        with open(training_config_path, "r") as f:
            training_config = yaml.safe_load(f)

        for section in ("model", "train", "data"):
            if section not in training_config:
                raise KeyError(
                    f"Missing `{section}` in LingBot-VLA V2 training config: "
                    f"{training_config_path}"
                )
        return training_config

    def _merge_qwen_config(self, policy_config, qwen_config):
        config_dict = (
            qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else qwen_config
        )
        text_config = config_dict.get("text_config", {})
        text_keys = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            "max_position_embeddings",
            "hidden_act",
            "tie_word_embeddings",
            "tokenizer_path",
        }
        for key in text_keys:
            if key in text_config:
                setattr(policy_config, key, text_config[key])
            elif key in config_dict:
                setattr(policy_config, key, config_dict[key])
        if "vision_config" not in config_dict:
            raise KeyError("Qwen3-VL AutoConfig is missing `vision_config`.")
        policy_config.vision_config = qwen_config.vision_config
        return policy_config

    def _mark_action_expert_fsdp_wrap_names(self):
        language_model_norm = getattr(
            self.vla_model.model.qwenvl_with_expert.qwenvl.model.language_model,
            "norm",
            None,
        )
        if language_model_norm is not None:
            language_model_norm._fsdp_wrap_name = "language_model_norm"

        if (
            getattr(self.config, "rl_trainable_scope", DEFAULT_RL_TRAINABLE_SCOPE)
            != "action_expert"
        ):
            return
        qwen_expert_norm = getattr(
            self.vla_model.model.qwenvl_with_expert.qwen_expert.model,
            "norm",
            None,
        )
        if qwen_expert_norm is not None:
            qwen_expert_norm._fsdp_wrap_name = "qwen_expert_norm"

    def _apply_rl_trainable_scope(self):
        trainable_scope = getattr(
            self.config, "rl_trainable_scope", DEFAULT_RL_TRAINABLE_SCOPE
        )
        if trainable_scope in (None, "all"):
            self._log_trainable_scope("all")
            return
        if trainable_scope not in ("action_head", "action_expert"):
            raise ValueError(
                "Unsupported LingBot-VLA V2 rl_trainable_scope: "
                f"{trainable_scope}. Expected one of: all, action_head, action_expert."
            )

        for param in self.vla_model.parameters():
            param.requires_grad = False
        for name, param in self.vla_model.named_parameters():
            if trainable_scope == "action_head" and "action_out_proj" in name:
                param.requires_grad = True
            elif trainable_scope == "action_expert" and any(
                keyword in name
                for keyword in (
                    "qwen_expert",
                    "state_proj",
                    "action_in_proj",
                    "action_out_proj",
                    "action_time_mlp",
                    "time_mlp",
                )
            ):
                param.requires_grad = True
        if hasattr(self, "value_head"):
            for param in self.value_head.parameters():
                param.requires_grad = True
        self._log_trainable_scope(trainable_scope)

    def _log_trainable_scope(self, trainable_scope: str):
        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.parameters())
        self.logger.info(
            "LingBot-VLA V2 rl_trainable_scope=%s trainable_params=%d total_params=%d",
            trainable_scope,
            trainable_params,
            total_params,
        )

    def _to_chw_image(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image tensor/array, got shape {image.shape}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = image.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 3)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")
        if image.is_floating_point() and image.max() <= 1.0 and image.min() >= 0.0:
            image = image * 255.0
        image = image.permute(2, 0, 1).to(torch.float32).contiguous()
        # Match the upstream server: float resize without uint8 re-quantization.
        return self._resize(image)

    def _select_env_action_dims(self, action_tensor: torch.Tensor) -> torch.Tensor:
        if action_tensor.shape[-1] == self.action_env_dim:
            return action_tensor
        if action_tensor.shape[-1] <= max(ROBOTWIN_MODEL_TO_ENV_ACTION_INDICES):
            raise ValueError(
                "LingBot-VLA V2 expects action tensor with at least 16 dims before "
                f"RoboTwin env action reorder, got {action_tensor.shape[-1]}."
            )
        indices = torch.as_tensor(
            ROBOTWIN_MODEL_TO_ENV_ACTION_INDICES,
            device=action_tensor.device,
            dtype=torch.long,
        )
        return action_tensor.index_select(-1, indices)[..., : self.action_env_dim]

    def _select_active_action_dims(
        self, tensor: torch.Tensor, joint_mask: torch.Tensor
    ) -> torch.Tensor:
        if joint_mask.ndim == 3:
            mask = joint_mask[0, 0]
        elif joint_mask.ndim == 2:
            mask = joint_mask[0]
        else:
            mask = joint_mask
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        if tensor.shape[-1] != mask.shape[-1]:
            raise ValueError(
                "LingBot-VLA V2 action mask dim mismatch: "
                f"tensor dim {tensor.shape[-1]}, mask dim {mask.shape[-1]}."
            )
        return tensor[..., mask]

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.vla_model, "gradient_checkpointing_enable"):
            self.vla_model.gradient_checkpointing_enable(**kwargs)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def obs_processor(self, env_obs):
        processed_obs = {
            "image": env_obs.get("main_images", env_obs.get("prep_images")),
            "prompt": env_obs.get("task_descriptions", env_obs.get("prompt")),
            "state": env_obs.get("states", env_obs.get("prep_state")),
        }
        if "wrist_images" in env_obs and env_obs["wrist_images"] is not None:
            processed_obs["wrist_images"] = env_obs["wrist_images"]
        return processed_obs

    def _single_feature_observation(self, observation: Observation, index: int):
        image = self._to_chw_image(observation.image[index])
        left = image
        right = image
        wrist_images = getattr(observation, "wrist_images", None)
        if wrist_images is not None and len(wrist_images) > index:
            if len(wrist_images[index]) > 0 and wrist_images[index][0] is not None:
                left = self._to_chw_image(wrist_images[index][0])
            if len(wrist_images[index]) > 1 and wrist_images[index][1] is not None:
                right = self._to_chw_image(wrist_images[index][1])

        state = observation.state[index]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().to(dtype=torch.float32)
        else:
            state = torch.from_numpy(np.asarray(state)).to(dtype=torch.float32)

        prompt = observation.prompt[index] if observation.prompt is not None else ""
        return {
            "observation.images.cam_high": image,
            "observation.images.cam_left_wrist": left,
            "observation.images.cam_right_wrist": right,
            "observation.state": state,
            "task": prompt,
        }

    def _pad_and_stack_tensors(self, values):
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) == 1:
            return torch.stack(values, dim=0)
        if all(value.ndim == 1 for value in values):
            max_len = max(value.shape[0] for value in values)
            fill_value = False if values[0].dtype == torch.bool else 0
            padded = []
            for value in values:
                out = torch.full(
                    (max_len,),
                    fill_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                out[: value.shape[0]] = value
                padded.append(out)
            return torch.stack(padded, dim=0)
        raise ValueError(f"Cannot batch tensors with different shapes: {shapes}")

    def _preprocess_observation(self, observation: Observation):
        device = next(self.parameters()).device
        states_raw = observation.state
        batch_size = (
            states_raw.shape[0] if hasattr(states_raw, "shape") else len(states_raw)
        )
        applied = []
        for idx in range(batch_size):
            single = self._single_feature_observation(observation, idx)
            applied.append(self.feature_transform.apply(single, policy_eval=True))

        batch = {}
        for key in applied[0]:
            values = [item[key] for item in applied]
            if isinstance(values[0], torch.Tensor):
                batch[key] = self._pad_and_stack_tensors(values).to(device)
            else:
                batch[key] = values

        return applied, batch

    def _unnormalize_actions(self, applied, actions: torch.Tensor) -> torch.Tensor:
        outputs = []
        for single_applied, action in zip(applied, actions):
            item = dict(single_applied)
            item["actions"] = action.to(dtype=torch.float32, device="cpu")
            item["state"] = item["state"].to(dtype=torch.float32, device="cpu")
            data = self.feature_transform.unapply(item)
            action_value = data["action"]
            if isinstance(action_value, torch.Tensor):
                action_value = action_value.float().cpu()
            else:
                action_value = torch.from_numpy(
                    np.asarray(action_value, dtype=np.float32)
                )
            outputs.append(action_value)
        return torch.stack(outputs, dim=0)

    def output_transform(self, outputs: dict) -> dict:
        actions = outputs["actions"][:, : self.action_chunk, :]
        applied = outputs["applied"]
        unnormalized = self._unnormalize_actions(applied, actions)
        outputs["actions"] = self._select_env_action_dims(unnormalized)
        return outputs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        processed_obs = self.obs_processor(env_obs)
        observation = Observation.from_dict(processed_obs)

        outputs = self.sample_actions(
            observation,
            mode=mode,
            compute_values=compute_values,
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "applied": outputs["applied"]}
        )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
            "lang_tokens": outputs["lang_tokens"].cpu(),
            "lang_masks": outputs["lang_masks"].cpu(),
            "image_grid_thw": outputs["image_grid_thw"].cpu(),
        }
        cloned_obs = copy_dict_tensor(
            {
                key: value
                for key, value in env_obs.items()
                if key not in ["task_descriptions", "prompt"] and value is not None
            }
        )
        forward_inputs.update(cloned_obs)

        return actions, {
            "prev_logprobs": outputs["prev_logprobs"].to(torch.float32),
            "prev_values": outputs["prev_values"].to(torch.float32),
            "forward_inputs": forward_inputs,
        }

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        mode="train",
        compute_values=True,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        applied, batch = self._preprocess_observation(observation)
        bsize = batch["state"].shape[0]
        dtype = self.torch_dtype
        state = batch["state"].to(dtype=dtype)
        noise = self.sample_noise(
            (
                bsize,
                self.action_chunk,
                self.vla_model.config.max_action_dim,
            ),
            device,
        )

        prefix = self._embed_prefix(batch, dtype=dtype)
        past_key_values = prefix["past_key_values"]
        prefix_pad_masks = prefix["prefix_pad_masks"]
        prefix_position_ids = prefix["prefix_position_ids"]

        x_t = noise
        chains = [x_t]
        log_probs = []
        values = []
        num_steps = self.num_steps
        if mode == "train":
            denoise_inds = torch.tensor(
                [random.randint(0, num_steps - 1)] * num_steps, device=device
            )
        else:
            denoise_inds = torch.tensor([-1] * num_steps, device=device)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        timesteps = _denoise_timesteps(num_steps, device)
        for idx in range(num_steps):
            sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                prefix_position_ids,
                sample_mode,
                timesteps,
                compute_values,
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            chains.append(x_t)
            log_probs.append(self.get_logprob_norm(x_t, x_t_mean, x_t_std))
            values.append(value_t)

        log_probs = torch.stack(log_probs, dim=1)
        log_probs = log_probs[
            torch.arange(log_probs.shape[0], device=device),
            denoise_inds[:, 0],
        ]
        values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": x_t,
            "applied": applied,
            "chains": torch.stack(chains, dim=1),
            "prev_logprobs": self._select_active_action_dims(
                log_probs, batch["joint_mask"]
            ),
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "lang_tokens": batch["lang_tokens"],
            "lang_masks": batch["lang_masks"],
            "image_grid_thw": batch["image_grid_thw"],
        }

    def _embed_prefix(self, batch: dict, dtype):
        images = batch["images"].to(dtype=dtype)
        img_masks = batch["img_masks"]
        lang_tokens = batch["lang_tokens"]
        lang_masks = batch["lang_masks"]
        image_grid_thw = batch["image_grid_thw"]

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.vla_model.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = self._v2.make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        _, past_key_values, _ = self.vla_model.model.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.vla_model.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_position_ids": prefix_position_ids,
            "past_key_values": past_key_values,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        prefix_position_ids,
        mode,
        timesteps,
        compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, torch.Tensor):
            idx_tensor = idx.to(device=device, dtype=torch.long)
        else:
            idx_tensor = torch.full((bsize,), idx, device=device, dtype=torch.long)
        t_input = timesteps[idx_tensor]
        delta = timesteps[idx_tensor] - timesteps[idx_tensor + 1]
        v_t = self.vla_model.model.predict_velocity(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t.to(self.torch_dtype),
            t_input,
            prefix_position_ids=prefix_position_ids,
        )
        # The flow-SDE update runs in fp32 so the chains and logprobs stay precise.
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)
        if getattr(self.config, "add_value_head", False) and compute_values:
            suffix_out = self.get_suffix_out(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_input,
                prefix_position_ids,
            )
            value_t = self.value_head(torch.mean(suffix_out, dim=1))[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device, dtype=self.torch_dtype)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            noise_level = torch.tensor(
                getattr(self.config, "noise_level", 0.5), device=device
            )
            sigmas = (
                noise_level
                * torch.sqrt(
                    timesteps
                    / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                )[:-1]
            )
            sigma_i = sigmas[idx_tensor][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        else:
            raise ValueError(f"Invalid sample mode: {mode}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids,
    ):
        flow = self.vla_model.model
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = flow.embed_suffix(
            state,
            x_t.to(self.torch_dtype),
            timestep,
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_att_2d_masks = self._v2.make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_position_ids = flow._build_full_position_ids(
            prefix_position_ids,
            prefix_pad_masks,
            suffix_pad_masks,
        )
        position_ids = full_position_ids[:, :, -suffix_len:]
        outputs_embeds, _, _ = flow.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=flow.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(flow.config, "adanorm_time", False) else None,
        )
        return outputs_embeds[1][:, -self.action_chunk :]

    def get_logprob_norm(self, sample, mu, sigma):
        # bf16 loses too much precision in the squared z-score for PPO ratios.
        sample = sample.to(torch.float32)
        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        log_prob = (
            -((sample - mu) ** 2) / (2 * sigma_safe**2)
            - torch.log(sigma_safe)
            - 0.5 * math.log(2 * math.pi)
        )
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        sigma = sigma.to(torch.float32)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return torch.where(mask, torch.zeros_like(entropy), entropy)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def enable_torch_compile(
        self,
        mode: str = "max-autotune-no-cudagraphs",
    ) -> None:
        """Compile the shared Qwen-VL expert forward for fused flex attention."""
        if getattr(self, "torch_compile_enabled", False):
            return
        self.vla_model.model.qwenvl_with_expert.forward = torch.compile(
            self.vla_model.model.qwenvl_with_expert.forward, mode=mode
        )
        self.torch_compile_enabled = True

    def sft_forward(self, data, **kwargs):
        device = next(iter(self.vla_model.model.parameters())).device
        data = tree_map(
            lambda x: (
                torch.as_tensor(x, device=device).contiguous()
                if isinstance(x, torch.Tensor)
                else x
            ),
            data,
        )
        dtype = self.torch_dtype
        outputs = self.vla_model(
            images=data["images"].to(dtype),
            img_masks=data["img_masks"],
            state=data["state"].to(dtype),
            lang_tokens=data["lang_tokens"],
            lang_masks=data["lang_masks"],
            actions=data["actions"].to(dtype),
            joint_mask=data.get("joint_mask"),
            action_is_pad=data.get("action_is_pad"),
            image_grid_thw=data.get("image_grid_thw"),
            depth_targets=data.get("depth_targets"),
            future_depth_targets=data.get("future_depth_targets"),
            future_video_targets=data.get("future_video_targets"),
            future_video_cls_targets=data.get("future_video_cls_targets"),
            future_video_current_patch=data.get("future_video_current_patch"),
        )
        (
            total_loss,
            loss_vla,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            seq_wise_loss,
            loss_dict,
            *_,
        ) = outputs
        return {
            "loss": total_loss,
            "l1_loss": loss_vla,
            "depth_loss": loss_depth,
            "future_depth_loss": loss_future_depth,
            "future_video_loss": loss_future_video,
            "seq_wise_loss": seq_wise_loss,
            **loss_dict,
        }

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        obs_dict = {
            "image": forward_inputs.get(
                "main_images",
                forward_inputs.get("prep_images", forward_inputs.get("images")),
            ),
            "state": forward_inputs.get("states", forward_inputs.get("prep_state")),
        }
        if "wrist_images" in forward_inputs:
            obs_dict["wrist_images"] = forward_inputs["wrist_images"]
        observation = Observation.from_dict(obs_dict)
        _, batch = self._preprocess_observation(observation)
        for key in ("lang_tokens", "lang_masks", "image_grid_thw"):
            if key in forward_inputs:
                batch[key] = forward_inputs[key].to(batch["state"].device)

        prefix = self._embed_prefix(batch, dtype=self.torch_dtype)
        device = batch["state"].device
        timesteps = _denoise_timesteps(self.num_steps, device)

        # Only the step sampled with noise during rollout carries a logprob.
        batch_inds = torch.arange(chains.shape[0], device=device)
        step_inds = denoise_inds[:, 0].to(device=device, dtype=torch.long)
        x_t = chains[batch_inds, step_inds].to(device, dtype=torch.float32)
        x_next = chains[batch_inds, step_inds + 1].to(device, dtype=torch.float32)
        x_t_mean, x_t_std, values = self.sample_mean_var_val(
            x_t,
            step_inds,
            batch["state"].to(dtype=self.torch_dtype),
            prefix["prefix_pad_masks"],
            prefix["past_key_values"],
            prefix["prefix_position_ids"],
            "train",
            timesteps,
            compute_values,
        )
        log_probs = self._select_active_action_dims(
            self.get_logprob_norm(x_next, x_t_mean, x_t_std), batch["joint_mask"]
        )
        entropy = self._select_active_action_dims(
            self.gaussian_entropy(x_t_std), batch["joint_mask"]
        ).mean(dim=[1, 2])[:, None]
        outputs = {
            "logprobs": log_probs[:, : self.action_chunk].to(torch.float32),
            "values": values[:, None].to(torch.float32),
            "entropy": entropy.to(torch.float32),
        }
        if "prev_logprobs" in kwargs:
            outputs["prev_logprobs"] = outputs["logprobs"].detach()
        return outputs
