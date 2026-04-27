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
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d
from tianshou.data import Batch
from transformers.configuration_utils import PretrainedConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    env_action_dim: int = field(
        default=None, metadata={"help": "Environment action dimension."}
    )
    num_action_chunks: int = field(
        default=16, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    gradient_checkpointing: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    _no_split_modules = [
        "CausalWanAttentionBlock",  # action head
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            self._patch_causal_wan_model_forward_train(diffusion_model)
            enabled = self.config.gradient_checkpointing
            if diffusion_model is not None:
                if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                    diffusion_model._set_gradient_checkpointing(
                        diffusion_model, enabled
                    )
                elif hasattr(diffusion_model, "gradient_checkpointing"):
                    diffusion_model.gradient_checkpointing = enabled
        except Exception:
            pass

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to model input for end-effector control"""
        main = env_obs["main_images"]
        wrist = env_obs.get("wrist_images", None)
        states = env_obs.get("states", None)
        prompts = env_obs.get("task_descriptions", None)
        if torch.is_tensor(main):
            main = main.detach().cpu().numpy()
        else:
            main = np.asarray(main)
        B = main.shape[0]
        if wrist is not None:
            if torch.is_tensor(wrist):
                wrist = wrist.detach().cpu().numpy()
            else:
                wrist = np.asarray(wrist)

        def _resize_bt_hwc_uint8(x, h=256, w=256):
            # x: [B,H,W,C
            B = x.shape[0]
            out = np.empty((B, h, w, 3), dtype=np.uint8)
            for b in range(B):
                frame = x[b]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out[b] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return out

        main = _resize_bt_hwc_uint8(main)
        if wrist is not None:
            wrist = _resize_bt_hwc_uint8(wrist)
        if main.ndim == 4:
            main = main[:, None, ...]
        if wrist is not None and wrist.ndim == 4:
            wrist = wrist[:, None, ...]
        if states is not None:
            if torch.is_tensor(states):
                s_np = states.detach().cpu().numpy()
            else:
                s_np = np.asarray(states)
        else:
            s_np = np.zeros((B, 8), dtype=np.float32)
        if s_np.ndim == 1:
            s_np = s_np[None, :]
        elif s_np.ndim > 2:
            s_np = s_np.reshape(B, -1)
        s_np = s_np.astype(np.float32)
        state_bt = s_np[:, None, :]
        prompts = prompts if prompts is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        converted_obs = {
            "video.image": main,  # [B,H,W,C]
            "video.wrist_image": wrist,  # [B,H,W,C]
            "state.state": state_bt,  # [B,1,8]
            "annotation.language.action_text": list(prompts),  # list[str], len=B
        }
        return converted_obs

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - extra_view_images: [B,H,W,C]
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, 8]  # 6ee + 1 gripper
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        # Unnormalize actions (pass obs for relative action normalization)
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": normalized_action.cpu()}
        )
        batch.act = unnormalized_action

        actions = batch.act["action.actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
            actions.dtype
        )

        assert actions.shape[-1] == self.config.env_action_dim, (
            f"Action shape mismatch: {actions.shape} != {self.config.env_action_dim}"
        )

        flat = (
            torch.as_tensor(actions, dtype=torch.float32)
            .reshape(actions.shape[0], -1)
            .cpu()
        )
        forward_inputs = {"action": flat}
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        outputs = super().forward(data)
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError

    def _patch_causal_wan_model_forward_train(self, model: torch.nn.Module) -> bool:
        """
        Monkey-patch DreamZero CausalWanModel._forward_train to support:
        - micro-batch (B) > 1

        Returns True if patched, False otherwise.
        """
        if model is None or not hasattr(model, "_forward_train"):
            return False

        def _forward_train_patched(
            self,
            x,
            timestep,
            timestep_action,
            context,
            seq_len,
            clean_x=None,
            aug_t=None,
            y=None,
            clip_feature=None,
            action=None,
            state=None,
            embodiment_id=None,
        ):
            # This is a minimally-edited copy of DreamZero's CausalWanModel._forward_train.
            # The only intentional behavioral change is checkpoint invocation.
            if self.model_type == "i2v":
                assert clip_feature is not None and y is not None

            if y is not None and self.concat_first_frame_latent:
                x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

            x = self.patch_embedding(x)
            grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
            freqs = self._create_freqs(
                grid_size=grid_size,
                start_frame=0,
            )

            x = x.flatten(start_dim=2).transpose(1, 2)
            assert x.shape[1] == seq_len

            B = x.shape[0]
            F = timestep.shape[1]

            if action is not None:
                embodiment_id = (
                    torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
                )
                action_features = self.action_encoder(
                    action, timestep_action, embodiment_id
                )
                action_length = action_features.shape[1]
                state_features = self.state_encoder(state, embodiment_id)
                action_register = torch.cat([action_features, state_features], dim=1)
                action_register_length = action_register.shape[1]
                x = torch.cat([x, action_register], dim=1)
            else:
                action_features = None
                action_length = None
                state_features = None
                action_register = None
                action_register_length = None

            timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
            timestep_original = timestep.clone()

            if action is not None:
                assert timestep_action is not None
                assert state_features is not None
                stride = timestep_action.shape[1] // state_features.shape[1]
                timestep_state = timestep_action[:, ::stride]
                timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
            )
            e = e.unflatten(dim=0, sizes=(B, -1))
            e0 = self.time_projection(e)
            e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

            assert context.shape[1] == self.text_len
            context = self.text_embedding(context)
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context = torch.cat([clip_embedding, context], dim=1)

            if clean_x is not None:
                if y is not None and self.concat_first_frame_latent:
                    clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
                clean_x = self.patch_embedding(clean_x)
                clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
                assert clean_x.shape[1] == seq_len

                x = torch.cat([clean_x, x], dim=1)

                if aug_t is None:
                    aug_t = torch.zeros_like(timestep_original)

                e_clean = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
                )
                e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
                e0_clean = self.time_projection(e_clean)
                e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
                e0 = torch.cat([e0_clean, e0], dim=1)

            kwargs = {
                "e": e0,
                "freqs": freqs,
                "freqs_action": self.freqs_action,
                "freqs_state": self.freqs_state,
                "action_register_length": action_register_length,
                "context": context,
                "is_tf": clean_x is not None,
            }

            def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    outputs, updated_kv_cache = module(*inputs, **kwargs)
                    assert updated_kv_cache is None
                    return outputs

                return custom_forward

            for block in self.blocks:
                use_ckpt = (
                    torch.is_grad_enabled()
                    and self.gradient_checkpointing
                    and not (action_register_length is not None and x.shape[0] > 1)
                )

                if use_ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        **kwargs,
                        use_reentrant=False,
                    )
                else:
                    x, _ = block(x, **kwargs)

            if clean_x is not None:
                x = x[:, clean_x.shape[1] :]

            if action is not None:
                action_noise_pred = x[:, seq_len : seq_len + action_length]
                action_noise_pred = self.action_decoder(
                    action_noise_pred, embodiment_id
                )
            else:
                action_noise_pred = None

            x_video = x[:, :seq_len]
            e_video = e[:, :seq_len]
            x_video = self.head(x_video, e_video.unsqueeze(2))
            video_noise_pred = self.unpatchify(x_video, grid_size)
            return video_noise_pred, action_noise_pred

        model._forward_train = types.MethodType(_forward_train_patched, model)
        return True
