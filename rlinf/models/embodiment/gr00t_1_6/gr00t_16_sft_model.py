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

import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from PIL import Image
from torch import nn
from torch.utils import _pytree
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor

try:
    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    print(
        "[Module Pre-registration] Successfully registered Gr00tN1d6 architecture mapping to Transformers"
    )
except Exception:
    pass

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


def patched_sample_time(self, batch_size, device, dtype):
    alpha = torch.tensor(
        self.config.noise_beta_alpha, dtype=torch.float32, device=device
    )
    beta_val = torch.tensor(
        self.config.noise_beta_beta, dtype=torch.float32, device=device
    )
    from torch.distributions import Beta

    temp_beta_dist = Beta(alpha, beta_val)

    sample = temp_beta_dist.sample([batch_size]).to(device, dtype=dtype)
    sample = (1 - sample) * self.config.noise_s
    return sample


def patched_process_backbone_output(self, backbone_output):
    backbone_features = backbone_output["backbone_features"]
    if hasattr(self, "vlln") and isinstance(self.vlln, nn.LayerNorm):
        orig_dtype = backbone_features.dtype
        weight = self.vlln.weight.float() if self.vlln.weight is not None else None
        bias = self.vlln.bias.float() if self.vlln.bias is not None else None

        backbone_features = F.layer_norm(
            backbone_features.float(),
            self.vlln.normalized_shape,
            weight,
            bias,
            self.vlln.eps,
        ).to(orig_dtype)
    else:
        backbone_features = self.vlln(backbone_features)

    backbone_output["backbone_features"] = backbone_features
    return backbone_output


def patched_action_head_forward(self, backbone_output, action_input):
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)

    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    state_features = self.state_encoder(action_input.state, embodiment_id)

    if self.state_dropout_prob > 0:
        do_dropout = (
            torch.rand(state_features.shape[0], device=state_features.device)
            < self.state_dropout_prob
        )
        do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
        state_features = (
            state_features * (1 - do_dropout) + self.mask_token * do_dropout
        )

    if self.training and self.state_additive_noise_scale > 0:
        noise = torch.randn_like(state_features) * self.state_additive_noise_scale
        state_features = state_features + noise

    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]

    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise

    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(
        noisy_trajectory, t_discretized, embodiment_id
    )

    if self.config.add_pos_embed:
        pos_ids = torch.arange(
            action_features.shape[1], dtype=torch.long, device=device
        )
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask

    if self.config.use_alternate_vl_dit:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )

    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]
    action_mask = action_input.action_mask

    pred_actions = torch.nan_to_num(pred_actions, nan=0.0, posinf=1.0, neginf=-1.0)
    velocity = torch.nan_to_num(velocity, nan=0.0)

    pred_actions_fp32 = pred_actions.float()
    velocity_fp32 = velocity.float()
    action_mask_fp32 = action_mask.float()

    action_loss_fp32 = (
        F.mse_loss(pred_actions_fp32, velocity_fp32, reduction="none")
        * action_mask_fp32
    )
    loss_fp32 = action_loss_fp32.sum() / (action_mask_fp32.sum() + 1e-6)

    loss = loss_fp32.to(pred_actions.dtype)
    action_loss = action_loss_fp32.to(pred_actions.dtype)

    return {
        "loss": loss,
        "action_loss": action_loss,
        "action_mask": action_mask,
        "backbone_features": vl_embeds,
        "state_features": state_features,
    }


class GR00T_1_6_SFT_Model(Gr00tN1d6, BasePolicy):
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        embodiment_tag: Optional[str] = None,
        local_model_path: Optional[str] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        modality_config: Optional[Any] = None,
        modality_transform: Optional[Any] = None,
        **kwargs,
    ):
        if embodiment_tag is None:
            from gr00t.data.embodiment_tags import EmbodimentTag

            embodiment_tag = EmbodimentTag.ROBOCASA_PANDA_OMRON

        if local_model_path is None:
            local_model_path = getattr(
                config, "_name_or_path", "/workspace/test/RLinf/GR00T-N1.6-3B"
            )

        processor_path = kwargs.pop("processor_path", None)
        kwargs.pop("denoising_steps", None)
        kwargs.pop("output_action_chunks", None)
        kwargs.pop("obs_converter_type", None)
        kwargs.pop("libero_action_mode", None)
        kwargs.pop("rl_head_config", None)
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )

        def _patched_normalize_values_minmax(values, params):
            """Patched version of normalize_values_minmax for dimension broadcasting.

            This version properly handles broadcasting when params are 1D but values
            are multi-dimensional. Uses broadcasting multiplication instead of
            boolean indexing to avoid dimension mismatch errors.
            """
            import numpy as np

            min_vals = np.asarray(params["min"])
            max_vals = np.asarray(params["max"])

            # Compute diff with safe handling for zero range
            diff = max_vals - min_vals
            safe_diff = np.where(np.abs(diff) < 1e-8, 1.0, diff)

            # Use broadcasting for normalization
            # (values - min) / (max - min) * 2 - 1
            # Broadcasting works when values is (T, D) and min_vals is (D,)
            normalized = (values - min_vals) / safe_diff * 2.0 - 1.0

            return normalized

        # CRITICAL: Patch normalize_values_minmax BEFORE Gr00tN1d6Processor is created.
        # StateActionProcessor imports normalize_values_minmax at module load time,
        # so we must patch it BEFORE importing Gr00tN1d6Processor.
        from gr00t.data import utils as gr00t_utils

        if hasattr(gr00t_utils, "normalize_values_minmax"):
            gr00t_utils.normalize_values_minmax = _patched_normalize_values_minmax
            print("[RLinf Patch] Patched gr00t.data.utils.normalize_values_minmax")

        # Also patch StateActionProcessor's module-level reference
        try:
            from gr00t.data.state_action import state_action_processor as sap_module

            if hasattr(sap_module, "normalize_values_minmax"):
                sap_module.normalize_values_minmax = _patched_normalize_values_minmax
                print(
                    "[RLinf Patch] Patched state_action_processor.normalize_values_minmax"
                )
        except ImportError:
            pass

        super().__init__(
            config, transformers_loading_kwargs=transformers_loading_kwargs, **kwargs
        )

        self.embodiment_tag = embodiment_tag
        self.compute_dtype = compute_dtype
        self.model_path = Path(local_model_path)

        if hasattr(self, "action_head"):
            self.action_head.sample_time = types.MethodType(
                patched_sample_time, self.action_head
            )
            self.action_head.process_backbone_output = types.MethodType(
                patched_process_backbone_output, self.action_head
            )
            self.action_head.forward = types.MethodType(
                patched_action_head_forward, self.action_head
            )
            print("providing patch for Action Head (Monkey Patch)")

        if modality_config is None or modality_transform is None:
            print("Loading Processor...")
            if processor_path is not None:
                import json

                from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
                    Gr00tN1d6Processor,
                )

                processor_path = Path(processor_path)
                with open(processor_path / "processor_config.json", "r") as f:
                    processor_cfg = json.load(f)["processor_kwargs"]
                with open(processor_path / "statistics.json", "r") as f:
                    processor_cfg["statistics"] = json.load(f)
                with open(processor_path / "embodiment_id.json", "r") as f:
                    processor_cfg["embodiment_id_mapping"] = json.load(f)
                if "base_motion" not in processor_cfg:
                    processor_cfg["base_motion"] = None
                modality_transform = Gr00tN1d6Processor(**processor_cfg)
                modality_config = getattr(modality_transform, "modality_configs", None)
            else:
                processor = AutoProcessor.from_pretrained(
                    str(local_model_path), trust_remote_code=True
                )
                modality_transform = processor
                modality_config = getattr(processor, "modality_config", None)
            print("Processor loaded successfully.")

            # After Processor is loaded, patch the StateActionProcessor's
            # normalize_values_minmax reference in sys.modules
            import sys

            if "gr00t.data.state_action.state_action_processor" in sys.modules:
                sap_module = sys.modules[
                    "gr00t.data.state_action.state_action_processor"
                ]
                sap_module.normalize_values_minmax = _patched_normalize_values_minmax
                print(
                    "[RLinf Patch] Patched sys.modules state_action_processor.normalize_values_minmax"
                )

        self._modality_config = modality_config
        self._modality_transform = modality_transform

        self.to(self.compute_dtype)
        print(
            f"all model residual parameters have been forcibly unified and shuffled to: {self.compute_dtype}"
        )

        self.requires_grad_(False)

        if hasattr(self, "action_head") and self.action_head is not None:
            self.action_head.requires_grad_(True)

        self._restore_official_backbone_trainable_parameters(config)
        self._restore_official_trainable_backbone_fp32(config)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[SFT Override] Number of parameters participating in training: {trainable_params / 1e6:.2f} M"
        )

        print("[monitor loading] registering gradient monitors...")
        for name, param in self.named_parameters():
            if param.requires_grad:

                def create_check_grad_fn(layer_name):
                    def check_grad(grad):
                        if grad is not None:
                            if torch.isnan(grad).any() or torch.isinf(grad).any():
                                print(
                                    f"[gradient exposion] find NaN gradient! Layer: {layer_name}"
                                )
                                return torch.nan_to_num(
                                    grad, nan=0.0, posinf=0.0, neginf=0.0
                                )
                        return grad

                    return check_grad

                param.register_hook(create_check_grad_fn(name))

    def forward(self, forward_type=ForwardType.SFT, **kwargs):
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: dict, **kwargs):
        obs = dict(data["observation"])
        actions = data["actions"]
        action_pad_mask = None
        for pad_key in ("actions_is_pad", "action_is_pad"):
            if pad_key in obs:
                action_pad_mask = obs.pop(pad_key)
                break

        for k, v in obs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                decoded_texts = []
                for row in v:
                    valid_bytes = row[row != 0].tolist()
                    decoded_texts.append(bytes(valid_bytes).decode("utf-8"))
                obs[k] = decoded_texts

        def safe_to_tensor(x):
            if x is None:
                return x
            if isinstance(x, str) or (
                isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)
            ):
                return x
            return torch.as_tensor(x, device=self.device).contiguous().clone()

        obs = _pytree.tree_map(safe_to_tensor, obs)

        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        if action_pad_mask is not None:
            action_pad_mask = torch.as_tensor(
                action_pad_mask, device=self.device, dtype=torch.bool
            )
            if action_pad_mask.dim() == 1:
                action_pad_mask = action_pad_mask.unsqueeze(1)

        model_inputs = self.apply_transforms(
            obs, actions=actions, action_pad_mask=action_pad_mask
        )

        if action_pad_mask is not None and "action_mask" in model_inputs:
            if action_pad_mask.shape[1] != model_inputs["action_mask"].shape[1]:
                fixed_mask = torch.ones(
                    model_inputs["action_mask"].shape[:2],
                    device=self.device,
                    dtype=torch.bool,
                )
                valid_horizon = min(action_pad_mask.shape[1], fixed_mask.shape[1])
                fixed_mask[:, :valid_horizon] = action_pad_mask[:, :valid_horizon]
                action_pad_mask = fixed_mask
            valid_action_mask = (
                (~action_pad_mask)
                .unsqueeze(-1)
                .to(
                    dtype=model_inputs["action_mask"].dtype,
                    device=model_inputs["action_mask"].device,
                )
            )
            model_inputs["action_mask"] = (
                model_inputs["action_mask"] * valid_action_mask
            )

        for key in [
            "eagle_input_ids",
            "eagle_attention_mask",
            "eagle_pixel_values",
            "eagle_image_sizes",
        ]:
            model_inputs.pop(key, None)

        with torch.amp.autocast("cuda", enabled=False):
            model_inputs_fp32 = {
                k: v.float()
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                else v
                for k, v in model_inputs.items()
            }

            backbone_in, action_in = super().prepare_input(model_inputs_fp32)

            backbone_out = self.backbone(backbone_in)

            outputs = self.action_head(backbone_out, action_in)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if torch.isnan(loss):
            print(
                "[detect NaN Loss] force to zero and continue training to prevent collapse."
            )
            loss = torch.zeros(
                [], device=loss.device, dtype=loss.dtype, requires_grad=True
            )

        return loss

    def _restore_official_backbone_trainable_parameters(self, config):
        """Restore GR00T backbone fine-tuning flags after the global freeze."""
        if not hasattr(self, "backbone") or self.backbone is None:
            return

        backbone_model = getattr(self.backbone, "model", None)
        if backbone_model is None:
            return

        if getattr(config, "tune_llm", False) and hasattr(
            backbone_model, "language_model"
        ):
            backbone_model.language_model.requires_grad_(True)
            print("[SFT Override] Restored full LLM backbone training.")
        else:
            tune_top_llm_layers = int(getattr(config, "tune_top_llm_layers", 0) or 0)
            language_model = getattr(backbone_model, "language_model", None)
            layers = getattr(getattr(language_model, "model", None), "layers", None)
            if tune_top_llm_layers > 0 and layers is not None:
                for layer in layers[-tune_top_llm_layers:]:
                    layer.requires_grad_(True)
                print(
                    "[SFT Override] Restored training for top "
                    f"{tune_top_llm_layers} LLM layers."
                )

        if getattr(config, "tune_visual", False):
            vision_model = getattr(backbone_model, "vision_model", None)
            if vision_model is not None:
                vision_model.requires_grad_(True)
            mlp1 = getattr(backbone_model, "mlp1", None)
            if mlp1 is not None:
                mlp1.requires_grad_(True)
            print("[SFT Override] Restored visual backbone training.")

    def _restore_official_trainable_backbone_fp32(self, config):
        """Match official GR00T behavior for trainable backbone parameter dtype."""
        if not getattr(config, "backbone_trainable_params_fp32", False):
            return
        if not getattr(config, "load_bf16", False):
            return
        if not hasattr(self, "backbone") or self.backbone is None:
            return

        cast_count = 0
        for _, param in self.backbone.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                cast_count += 1
        if cast_count:
            print(
                "[SFT Override] Restored official fp32 dtype for "
                f"{cast_count} trainable backbone parameters."
            )

    def apply_transforms(self, obs: dict, actions=None, action_pad_mask=None) -> dict:
        class SimulationContent:
            def __init__(self, embodiment, states, actions, images, text):
                self.embodiment = embodiment
                self.states = states
                self.actions = actions
                self.images = images
                self.text = text

        batch_size = len(next(iter(obs.values())))

        text_key = None
        image_keys = []
        state_keys = []
        for k in obs.keys():
            k_lower = k.lower()
            if "task" in k_lower or "lang" in k_lower or "instruction" in k_lower:
                text_key = k
            elif "image" in k_lower or "rgb" in k_lower or "cam" in k_lower:
                image_keys.append(k)
            else:
                state_keys.append(k)

        processed_outputs = []

        for i in range(batch_size):
            text = obs[text_key][i] if text_key else ""
            if isinstance(text, (list, np.ndarray)):
                text = str(text[0]) if len(text) > 0 else ""
            else:
                text = str(text)

            def to_numpy(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().float().numpy()
                return np.asarray(value, dtype=np.float32)

            def split_libero_state(value):
                value = to_numpy(value).reshape(-1)
                if value.shape[0] < 7:
                    raise ValueError(
                        f"LIBERO state should have at least 7 dims, got {value.shape}"
                    )
                return {
                    "x": value[0:1][None, :],
                    "y": value[1:2][None, :],
                    "z": value[2:3][None, :],
                    "roll": value[3:4][None, :],
                    "pitch": value[4:5][None, :],
                    "yaw": value[5:6][None, :],
                    "gripper": value[6:8][None, :],
                }

            def split_libero_action(value):
                value = to_numpy(value)
                if value.ndim == 1:
                    value = value[None, :]
                if value.shape[-1] < 7:
                    raise ValueError(
                        f"LIBERO action should have at least 7 dims, got {value.shape}"
                    )
                return {
                    "x": value[:, 0:1],
                    "y": value[:, 1:2],
                    "z": value[:, 2:3],
                    "roll": value[:, 3:4],
                    "pitch": value[:, 4:5],
                    "yaw": value[:, 5:6],
                    "gripper": value[:, 6:7],
                }

            def split_robocasa_action(value):
                """Convert raw LIBERO action [x,y,z,roll,pitch,yaw,gripper] to gr00t robocasa format."""
                value = to_numpy(value)
                if value.ndim == 1:
                    value = value[None, :]
                if value.shape[-1] < 7:
                    raise ValueError(
                        f"LIBERO action should have at least 7 dims, got {value.shape}"
                    )
                T = value.shape[0]  # Get time dimension for padding
                # gr00t robocasa_panda_omron action keys: end_effector_position, end_effector_rotation, gripper_close, base_motion, control_mode
                return {
                    "end_effector_position": value[:, 0:3],
                    "end_effector_rotation": value[:, 3:6],
                    "gripper_close": value[:, 6:7],
                    # Add dummy keys expected by processor
                    "base_motion": np.zeros((T, 4), dtype=np.float32),
                    "control_mode": np.zeros((T, 1), dtype=np.float32),
                }

            def _axis_angle_to_quat(axis_angle):
                """Convert axis-angle [x,y,z] to quaternion [x,y,z,w].
                axis_angle: shape (..., 3)
                Returns: shape (..., 4)
                """
                axis_angle = np.asarray(axis_angle, dtype=np.float64)
                norm = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
                angle = norm
                norm = np.where(norm > 1e-8, norm, 1.0)
                axis = axis_angle / norm
                half_angle = angle / 2.0
                sin_half = np.sin(half_angle)
                cos_half = np.cos(half_angle)
                quat = np.concatenate([axis * sin_half, cos_half], axis=-1)
                quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
                return quat.astype(np.float32)

            def build_robocasa_states(raw_state):
                """Convert raw LIBERO state [x,y,z,roll,pitch,yaw,gripper] to
                gr00t robocasa_panda_omron format. Handles any input shape.
                """
                raw_state = to_numpy(raw_state)
                if raw_state.ndim == 1:
                    if raw_state.shape[0] >= 7 and raw_state.shape[0] % 7 == 0:
                        raw_state = raw_state.reshape(raw_state.shape[0] // 7, 7)
                    else:
                        raw_state = raw_state.reshape(1, -1)
                elif raw_state.ndim == 2:
                    raw_state = (
                        raw_state.reshape(-1, 7)
                        if raw_state.shape[1] >= 7
                        else raw_state
                    )
                else:
                    raw_state = raw_state.reshape(-1, 7)

                if raw_state.ndim == 1:
                    raw_state = raw_state.reshape(1, -1)

                if raw_state.shape[-1] < 7:
                    if raw_state.ndim == 1:
                        raw_state = np.concatenate(
                            [raw_state, np.zeros(7 - raw_state.shape[0])]
                        )
                        raw_state = raw_state.reshape(1, 7)
                    else:
                        padded = np.zeros((raw_state.shape[0], 7), dtype=np.float32)
                        d = min(raw_state.shape[-1], 7)
                        padded[:, :d] = raw_state[:, :d]
                        raw_state = padded

                # Extract parts
                ref_T = raw_state.shape[0]
                eef_pos = raw_state[:, 0:3]  # (T, 3)  -- correct
                axis_angle = raw_state[:, 3:6]  # (T, 3)  -- axis-angle
                grip_val = raw_state[:, 6:7]  # (T, 1)  -- scalar gripper

                # Convert axis-angle -> quat: (T, 3) -> (T, 4)
                eef_quat = _axis_angle_to_quat(eef_pos)  # temp holder
                eef_quat = _axis_angle_to_quat(axis_angle)  # (T, 4) quat

                # Gripper: (T, 1) -> (T, 2) by duplicating (two fingers)
                grip_2f = np.concatenate([grip_val, grip_val], axis=-1)  # (T, 2)

                return {
                    "end_effector_position_relative": eef_pos,
                    "end_effector_rotation_relative": eef_quat,
                    "gripper_qpos": grip_2f,
                    "base_position": np.zeros((ref_T, 3), dtype=np.float32),
                    "base_rotation": np.zeros((ref_T, 4), dtype=np.float32),
                }

            tag_val = (
                self.embodiment_tag.value
                if hasattr(self.embodiment_tag, "value")
                else str(self.embodiment_tag)
            )
            state_key = None
            for candidate in ("state", "observation.state"):
                if candidate in obs:
                    state_key = candidate
                    break
            if tag_val == "libero_panda" and state_key is not None:
                states_dict = split_libero_state(obs[state_key][i])
            elif tag_val == "robocasa_panda_omron" and state_key is not None:
                states_dict = build_robocasa_states(obs[state_key][i])
            else:
                states_dict = {}
                for k in state_keys:
                    v = obs[k][i]
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().float().numpy()
                    states_dict[k] = np.array(v)

                ref_T = next(iter(states_dict.values())).shape[0] if states_dict else 1
                if tag_val == "robocasa_panda_omron":
                    robocasa_requirements = {
                        "end_effector_position_relative": 3,
                        "end_effector_rotation_relative": 4,
                        "gripper_qpos": 2,
                        "base_position": 3,
                        "base_rotation": 4,
                    }
                    for req_k, req_dim in robocasa_requirements.items():
                        if req_k not in states_dict:
                            states_dict[req_k] = np.zeros(
                                (ref_T, req_dim), dtype=np.float32
                            )

            actions_dict = None
            if actions is not None:
                if tag_val == "libero_panda":
                    actions_dict = split_libero_action(actions[i])
                elif tag_val == "robocasa_panda_omron":
                    actions_dict = split_robocasa_action(actions[i])
                else:
                    actions_dict = {"action": to_numpy(actions[i])}

            raw_images_list = []
            for img_k in image_keys:
                img_data = obs[img_k][i]
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.cpu().float().numpy()

                frames = []
                if img_data.ndim == 3:
                    img_data = np.expand_dims(img_data, 0)

                for t in range(img_data.shape[0]):
                    frame = img_data[t]
                    if frame.shape[0] in [1, 3] and frame.shape[2] > 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    frames.append(Image.fromarray(frame))
                raw_images_list.append(frames)

            images_dict = {}
            req_img_keys = []
            try:
                req_img_keys = self._modality_transform.modality_configs[tag_val][
                    "video"
                ].modality_keys
            except Exception:
                req_img_keys = [
                    "res256_image_side_0",
                    "res256_image_wrist_0",
                    "res256_image_front_0",
                ]

            self.image_nums = len(req_img_keys)

            if tag_val == "libero_panda":
                raw_by_key = dict(zip(image_keys, raw_images_list))
                source_for_req = {
                    "image": raw_by_key.get("base_0_rgb")
                    or raw_by_key.get("observation.images.image"),
                    "wrist_image": raw_by_key.get("right_wrist_0_rgb")
                    or raw_by_key.get("observation.images.wrist_image"),
                }
                for r_key in req_img_keys:
                    images_dict[r_key] = source_for_req.get(r_key) or (
                        raw_images_list[-1] if raw_images_list else []
                    )
            else:
                for idx, r_key in enumerate(req_img_keys):
                    if idx < len(raw_images_list):
                        images_dict[r_key] = raw_images_list[idx]
                    else:
                        images_dict[r_key] = (
                            raw_images_list[-1] if raw_images_list else []
                        )

            # --- DIAGNOSTIC: print state dict shapes ---
            with open("/tmp/state_diag.txt", "a") as _df:
                _df.write(f"tag={tag_val}  i={i}\n")
                for _k, _v in sorted(states_dict.items()):
                    _df.write(
                        f"  {_k}: shape={getattr(_v, 'shape', type(_v).__name__)}\n"
                    )

            content = SimulationContent(
                embodiment=self.embodiment_tag,
                states=states_dict,
                actions=actions_dict,
                images=images_dict,
                text=text,
            )

            messages = [{"role": "user", "content": content}]
            out = self._modality_transform(messages=messages)
            processed_outputs.append(out)

        collated_batch = self._modality_transform.collator(processed_outputs)

        if hasattr(collated_batch, "data") and "inputs" in collated_batch.data:
            batched_out = collated_batch.data["inputs"]
        elif "inputs" in collated_batch:
            batched_out = collated_batch["inputs"]
        else:
            batched_out = dict(collated_batch)

        if "input_ids" in batched_out:
            batched_out["eagle_input_ids"] = batched_out["input_ids"]
        if "attention_mask" in batched_out:
            batched_out["eagle_attention_mask"] = batched_out["attention_mask"]
        if "pixel_values" in batched_out:
            batched_out["eagle_pixel_values"] = batched_out["pixel_values"]
        if "image_sizes" in batched_out:
            batched_out["eagle_image_sizes"] = batched_out["image_sizes"]

        if "eagle_pixel_values" not in batched_out and "images" in batched_out:
            batched_out["eagle_pixel_values"] = batched_out["images"]
            batched_out["pixel_values"] = batched_out["images"]

        return batched_out

    def default_forward(self, *args, **kwargs):
        raise NotImplementedError("Pure SFT model does not support default_forward")

    def predict_action_batch(self, *args, **kwargs):
        raise NotImplementedError(
            "Pure SFT model does not support predict_action_batch"
        )


try:
    AutoModel.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    AutoModelForCausalLM.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    AutoModelForCausalLM.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    print("[register model] successfully registered GR00T_1_6_SFT_Model！")
except Exception as exc:
    print(f"[register model] failed to register GR00T_1_6_SFT_Model: {exc}")


def build_gr00t_dataloader(worker_instance, eval_dataset: bool = False):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader, Subset

    print(
        "[Lazy Load] Successfully triggered GR00T model file registration! "
        "Loading LeRobot Dataset..."
    )

    action_key = worker_instance.cfg.data.get("action_key", "actions")
    action_horizon = int(worker_instance.cfg.actor.model.get("num_action_chunks", 1))
    dataset_fps = float(worker_instance.cfg.data.get("fps", 20))
    delta_timestamps = None
    if action_horizon > 1:
        delta_timestamps = {
            action_key: [step / dataset_fps for step in range(action_horizon)]
        }

    dataset = LeRobotDataset(
        repo_id=worker_instance.cfg.data.train_data_paths,
        # local_files_only=True
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    valid_indices = _compute_gr00t_unpadded_indices(
        worker_instance.cfg.data.train_data_paths,
        action_horizon,
    )
    if valid_indices is not None:
        dataset = Subset(dataset, valid_indices)
        print(
            "[GR00T SFT] Filtered padded tail action chunks: "
            f"using {len(valid_indices)} unpadded samples."
        )
    else:
        print(
            "[GR00T SFT] Could not precompute unpadded sample indices; "
            "falling back to collate-time filtering."
        )

    def gr00t_collate_fn(batch):
        filtered_batch = []
        for item in batch:
            pad_key = next(
                (
                    key
                    for key in item.keys()
                    if key.endswith("_is_pad") and "action" in key.lower()
                ),
                None,
            )
            if pad_key is None or not bool(item[pad_key].any().item()):
                filtered_batch.append(item)
        if filtered_batch:
            batch = filtered_batch

        batch_action_key = action_key if action_key in batch[0] else None
        if batch_action_key is None:
            candidates = ["actions", "action"]
            batch_action_key = next(
                (key for key in candidates if key in batch[0]), None
            )
        if batch_action_key is None:
            batch_action_key = next(
                (
                    key
                    for key in batch[0].keys()
                    if "action" in key.lower() and not key.endswith("_is_pad")
                ),
                None,
            )
        if batch_action_key is None:
            raise KeyError("Could not find an action key!")

        actions = torch.stack([item[batch_action_key] for item in batch])
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        action_pad_key = f"{batch_action_key}_is_pad"
        obs = {}
        for key in batch[0].keys():
            if key == batch_action_key:
                continue
            if (
                "action" in key.lower()
                and key.endswith("_is_pad")
                and key != action_pad_key
            ):
                continue
            item_val = batch[0][key]
            if isinstance(item_val, str) or (
                isinstance(item_val, (list, tuple))
                and len(item_val) > 0
                and isinstance(item_val[0], str)
            ):
                byte_ts = []
                for item in batch:
                    text = item[key]
                    if isinstance(text, (list, tuple)):
                        text = text[0]
                    byte_ts.append(
                        torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
                    )
                obs[key] = torch.nn.utils.rnn.pad_sequence(
                    byte_ts, batch_first=True, padding_value=0
                )
            elif isinstance(item_val, torch.Tensor):
                obs[key] = torch.stack([item[key] for item in batch])
            else:
                try:
                    obs[key] = torch.tensor([item[key] for item in batch])
                except (TypeError, ValueError):
                    obs[key] = [item[key] for item in batch]
        return obs, actions

    dataloader = DataLoader(
        dataset,
        batch_size=worker_instance.cfg.actor.micro_batch_size
        * worker_instance._world_size,
        shuffle=not eval_dataset,
        num_workers=worker_instance.cfg.data.get("num_workers", 4),
        collate_fn=gr00t_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, None


def _compute_gr00t_unpadded_indices(dataset_path, action_horizon: int):
    """Return indices whose full action horizon exists, matching official loader."""
    if action_horizon <= 1:
        return None

    import json
    from pathlib import Path

    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"

    v3_episode_root = meta_dir / "episodes"
    if v3_episode_root.is_dir():
        try:
            import pandas as pd

            valid_indices = []
            episode_files = sorted(v3_episode_root.glob("chunk-*/*.parquet"))
            for episode_file in episode_files:
                episode_df = pd.read_parquet(episode_file)
                for _, row in episode_df.iterrows():
                    start = int(row["dataset_from_index"])
                    length = int(row["length"])
                    valid_length = max(0, length - action_horizon + 1)
                    valid_indices.extend(range(start, start + valid_length))
            return valid_indices
        except Exception as exc:
            print(f"[GR00T SFT] Failed to read v3 episode metadata: {exc}")
            return None

    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        try:
            valid_indices = []
            cursor = 0
            with open(episodes_jsonl, "r") as f:
                for line in f:
                    episode = json.loads(line)
                    length = int(episode["length"])
                    valid_length = max(0, length - action_horizon + 1)
                    valid_indices.extend(range(cursor, cursor + valid_length))
                    cursor += length
            return valid_indices
        except Exception as exc:
            print(f"[GR00T SFT] Failed to read v2 episode metadata: {exc}")
            return None

    return None
