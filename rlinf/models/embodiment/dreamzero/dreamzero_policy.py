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

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from tianshou.data import Batch
from transformers.configuration_utils import PretrainedConfig

from rlinf.data.datasets.dreamzero import q99_normalize
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

    # CausalWanModel has to be wrapped to avoid a FSDP2 bug
    # when using with gradient checkpointing
    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanModel",  # action head
        "CausalWanAttentionBlock",  # action head layer
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config

    # This method is called in FSDPModelManager.setup_model_and_optimizer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = True
            use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

            if diffusion_model is None:
                raise ValueError("DreamZero policy must have action_head.")

            if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                diffusion_model._set_gradient_checkpointing(diffusion_model, enabled)
            elif hasattr(diffusion_model, "gradient_checkpointing"):
                diffusion_model.gradient_checkpointing = enabled

            setattr(
                diffusion_model, "gradient_checkpointing_use_reentrant", use_reentrant
            )

            logging.warning(
                "DreamZero gradient checkpointing is enabled. If you encounter errors "
                "or memory leaks, consider: (1) upgrading to PyTorch 2.10 or later; "
                "(2) using use_reentrant=True to avoid issues when CUDA graphs and "
                "gradient checkpointing are used together."
            )

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
    
    _DAGGER_FORWARD_RESERVED_KEYS: frozenset[str] = frozenset(
        {"action", "model_action", "prev_logprobs", "prev_values"}
    )

    def _dagger_obs_tensors_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Copy tensors into forward_inputs (CPU) for replay use."""
        out: dict[str, torch.Tensor] = {}
        for k, v in normalized_input.items():
            if not torch.is_tensor(v):
                continue
            # TrajectoryReplayBuffer._flatten_trajectory only keeps tensors with dim>=2
            if v.dim() < 2:
                continue
            out[k] = v.detach().cpu().contiguous()
        return out

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
        # normalized action (same distribution as action_pred); DAgger relabel will overwrite with expert's model_action
        norm_flat = normalized_action.reshape(normalized_action.shape[0], -1).detach().cpu().contiguous()

        forward_inputs: dict[str, Any] = {}
        forward_inputs.update(
            self._dagger_obs_tensors_from_normalized_input(normalized_input)
        )
        forward_inputs["model_action"] = norm_flat
        forward_inputs["action"] = flat
        # Keep replay payload stable for downstream split/send logic.
        for key, value in list(forward_inputs.items()):
            if torch.is_tensor(value):
                forward_inputs[key] = value.detach().cpu().contiguous()

        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def _dagger_get_rollout_layout(self) -> tuple[int, int, int, int]:
        """Layout of tensors written by ``predict_action_batch`` (per env step)."""
        rollout_action_steps = int(self.action_horizon)
        max_action_dim = int(self.action_dim)
        num_chunks = int(getattr(self.config, "num_action_chunks", 16) or 16)
        env_action_dim = int(getattr(self.config, "env_action_dim", 7) or 7)
        transform = self.config.data_transforms
        if transform is not None:
            for t in getattr(transform, "transforms", []):
                if hasattr(t, "action_horizon") and t.action_horizon:
                    rollout_action_steps = int(t.action_horizon)
                if hasattr(t, "max_action_dim") and t.max_action_dim:
                    max_action_dim = int(t.max_action_dim)
                if hasattr(t, "num_chunks") and t.num_chunks:
                    num_chunks = int(t.num_chunks)
        return rollout_action_steps, max_action_dim, num_chunks, env_action_dim

    def _dagger_get_training_layout(self) -> tuple[int, int, int, int, int]:
        """Layout for WAN ``super().forward`` (aligned with DreamZeroLiberoDataset)."""
        max_action_dim = int(self.action_dim)
        max_state_dim = 64
        num_video_frames = 33
        state_horizon = 4
        action_horizon = 64
        ah = getattr(self, "action_head", None)
        if ah is not None:
            cfg = getattr(ah, "config", None)
            if cfg is not None:
                num_video_frames = int(getattr(cfg, "num_frames", num_video_frames) or 33)
                max_state_dim = int(getattr(cfg, "max_state_dim", max_state_dim) or 64)
        return action_horizon, max_action_dim, state_horizon, max_state_dim, num_video_frames

    def _dagger_get_sft_layout(self) -> tuple[int, int, int, int]:
        """Backward-compatible alias: training action horizon + rollout chunk count."""
        action_horizon, max_action_dim, _, _, _ = self._dagger_get_training_layout()
        _, _, num_chunks, env_action_dim = self._dagger_get_rollout_layout()
        return action_horizon, max_action_dim, num_chunks, env_action_dim

    def _dagger_get_action_q01_q99(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load action q01/q99 from experiment metadata (same source as SFT dataset)."""
        transform = self.config.data_transforms
        if transform is None:
            return None, None
        try:
            metadata = transform.dataset_metadata
            action_stats = metadata.statistics.action["actions"]
            q01 = np.asarray(action_stats.q01, dtype=np.float32)
            q99 = np.asarray(action_stats.q99, dtype=np.float32)
            return q01, q99
        except Exception:
            return None, None

    @staticmethod
    def _dagger_pad_time_first_last(
        sequence: np.ndarray, target_len: int
    ) -> np.ndarray:
        """Match DreamZeroLiberoDataset: repeat last step instead of zero-padding."""
        if sequence.shape[0] >= target_len:
            return sequence[:target_len]
        last = sequence[-1:]
        pad = np.repeat(last, target_len - sequence.shape[0], axis=0)
        return np.concatenate([sequence, pad], axis=0)

    @staticmethod
    def _dagger_pad_time_first_last_tensor(
        tensor: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Repeat last timestep along dim=1 (batch-first [B, T, ...])."""
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len].contiguous()
        last = tensor[:, -1:].contiguous()
        pad = last.repeat(1, target_len - tensor.shape[1], *([1] * (tensor.dim() - 2)))
        return torch.cat([tensor, pad], dim=1)

    def _dagger_align_observation_for_training(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Expand rollout (T=1) observations to WAN training layout (33 frames, 4 states)."""
        _, _, state_horizon, max_state_dim, num_video_frames = (
            self._dagger_get_training_layout()
        )
        out = dict(observation)

        if "images" in out and torch.is_tensor(out["images"]):
            images = out["images"]
            if images.dim() == 4:
                images = images.unsqueeze(1)
            if images.shape[1] < num_video_frames:
                images = self._dagger_pad_time_first_last_tensor(
                    images, num_video_frames
                )
            out["images"] = images.contiguous()

        if "state" in out and torch.is_tensor(out["state"]):
            state = out["state"]
            if state.dim() == 2:
                state = state.unsqueeze(1)
            if state.shape[1] < state_horizon:
                state = self._dagger_pad_time_first_last_tensor(state, state_horizon)
            if state.shape[-1] < max_state_dim:
                pad = torch.zeros(
                    *state.shape[:-1],
                    max_state_dim - state.shape[-1],
                    dtype=state.dtype,
                    device=state.device,
                )
                state = torch.cat([state, pad], dim=-1)
            elif state.shape[-1] > max_state_dim:
                state = state[..., :max_state_dim]
            out["state"] = state.contiguous()

        if "state_mask" in out and torch.is_tensor(out["state_mask"]):
            mask = out["state_mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.shape[1] < state_horizon:
                mask = self._dagger_pad_time_first_last_tensor(mask, state_horizon)
            if mask.shape[-1] < max_state_dim:
                pad = torch.zeros(
                    *mask.shape[:-1],
                    max_state_dim - mask.shape[-1],
                    dtype=torch.bool,
                    device=mask.device,
                )
                mask = torch.cat([mask, pad], dim=-1)
            elif mask.shape[-1] > max_state_dim:
                mask = mask[..., :max_state_dim]
            out["state_mask"] = mask.contiguous()

        return out

    @staticmethod
    def _dagger_pad_action_dim(
        action: np.ndarray, max_action_dim: int
    ) -> np.ndarray:
        """Zero-pad action dim to max_action_dim (LIBERO: 7 -> 32)."""
        action_pad = np.zeros(
            (action.shape[0], max_action_dim), dtype=np.float32
        )
        valid_dim = min(action.shape[-1], max_action_dim)
        action_pad[:, :valid_dim] = action[:, :valid_dim]
        return action_pad

    def _dagger_format_action_like_sft_dataset(
        self,
        action: np.ndarray,
        *,
        apply_q99: bool,
    ) -> np.ndarray:
        """(T, D_env) or (T, D_pred) -> (train_action_horizon, max_action_dim) in [-1, 1]."""
        action_horizon, max_action_dim, _, _, _ = self._dagger_get_training_layout()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        action = self._dagger_pad_time_first_last(action, action_horizon)
        if apply_q99:
            q01, q99 = self._dagger_get_action_q01_q99()
            if q01 is not None and q99 is not None:
                dim = min(action.shape[-1], q01.shape[0], q99.shape[0])
                action = q99_normalize(
                    action[:, :dim], q01[:dim], q99[:dim]
                )
            else:
                action = np.clip(action, -1.0, 1.0)
        else:
            action = np.clip(action, -1.0, 1.0)
        return self._dagger_pad_action_dim(action, max_action_dim)

    def _dagger_prepare_model_action(
        self, target_flat: torch.Tensor, bsz: int
    ) -> torch.Tensor:
        """``model_action`` is already normalized (rollout ``action_pred``); pad like SFT."""
        train_horizon, max_action_dim, _, _, _ = self._dagger_get_training_layout()
        rollout_steps, _, num_chunks, _ = self._dagger_get_rollout_layout()
        flat = target_flat.reshape(bsz, -1).float().cpu().numpy()
        batch_actions = []
        for i in range(bsz):
            row = flat[i]
            numel = row.size
            if numel == train_horizon * max_action_dim:
                act = row.reshape(train_horizon, max_action_dim)
            elif numel == rollout_steps * max_action_dim:
                act = self._dagger_format_action_like_sft_dataset(
                    row.reshape(rollout_steps, max_action_dim), apply_q99=False
                )
            elif numel % num_chunks == 0:
                chunk_dim = numel // num_chunks
                chunk = row.reshape(num_chunks, chunk_dim)
                act = self._dagger_format_action_like_sft_dataset(
                    chunk, apply_q99=False
                )
            else:
                raise ValueError(
                    "DreamZero DAgger: cannot reshape model_action with "
                    f"numel={numel}; expected {train_horizon * max_action_dim}, "
                    f"{rollout_steps * max_action_dim}, or {num_chunks}*D."
                )
            batch_actions.append(np.clip(act, -1.0, 1.0))
        return torch.as_tensor(np.stack(batch_actions, axis=0), dtype=torch.float32)

    def _dagger_prepare_env_action(
        self, env_action_flat: torch.Tensor, bsz: int
    ) -> torch.Tensor:
        """Env-executed actions: q99-normalize and pad exactly like SFT ``__getitem__``."""
        _, _, num_chunks, env_action_dim = self._dagger_get_rollout_layout()
        env_actions = env_action_flat.reshape(bsz, num_chunks, env_action_dim)
        env_actions_np = env_actions.detach().cpu().numpy()
        batch_actions = [
            self._dagger_format_action_like_sft_dataset(
                env_actions_np[i], apply_q99=True
            )
            for i in range(bsz)
        ]
        return torch.as_tensor(np.stack(batch_actions, axis=0), dtype=torch.float32)

    def _dagger_collect_obs_dict(self, batch: dict[str, Any]) -> dict[str, Any]:
        obs_dict: dict[str, Any] = {}
        for key in batch:
            if key.startswith("observation/"):
                obs_dict[key] = batch[key]
        for key, value in batch.items():
            if key in self._DAGGER_FORWARD_RESERVED_KEYS:
                continue
            if key in ("tokenized_prompt", "tokenized_prompt_mask"):
                continue
            if key.startswith("observation/"):
                continue
            obs_dict[key] = value
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
        return obs_dict

    def _dagger_observation_to_device(
        self, observation: dict[str, Any], device: torch.device, param_dtype: torch.dtype
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in observation.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.to(device=device)
            if tensor.dtype == torch.float32 and param_dtype != torch.float32:
                tensor = tensor.to(dtype=param_dtype)
            out[key] = tensor.contiguous()
        return out

    def prepare_dagger_sft_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare replay-buffer ``forward_inputs`` for DAgger WAN SFT (``sft_forward``)."""
        device = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype
        obs_dict = self._dagger_collect_obs_dict(batch)

        bsz = batch["action"].shape[0]

        if "model_action" in batch:
            actions = self._dagger_prepare_model_action(batch["model_action"], bsz)
        else:
            actions = self._dagger_prepare_env_action(batch["action"], bsz)

        observation = self._dagger_observation_to_device(
            obs_dict, device, param_dtype
        )
        observation = self._dagger_align_observation_for_training(observation)

        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(device),
        }

    def _merge_sft_observation_and_actions(
        self,
        observation: dict[str, Any],
        actions: torch.Tensor,
    ) -> dict[str, Any]:
        """Build VLA.forward inputs from observation + actions (SFT / DAgger)."""
        inputs = dict(observation)
        inputs["action"] = actions
        device = actions.device
        bsz = actions.shape[0]
        action_horizon = actions.shape[1]
        max_action_dim = actions.shape[2]

        if "action_mask" not in inputs:
            _, _, _, env_action_dim = self._dagger_get_rollout_layout()
            valid_dim = min(max_action_dim, env_action_dim)
            action_mask = torch.zeros(
                bsz, action_horizon, max_action_dim, dtype=torch.bool, device=device
            )
            action_mask[..., :valid_dim] = True
            inputs["action_mask"] = action_mask

        if "has_real_action" not in inputs:
            inputs["has_real_action"] = torch.ones(bsz, dtype=torch.bool, device=device)

        if "embodiment_id" not in inputs:
            embodiment_id = 21
            transform = self.config.data_transforms
            if transform is not None:
                for t in getattr(transform, "transforms", []):
                    mapping = getattr(t, "embodiment_tag_mapping", None)
                    tag = getattr(self.config, "embodiment_tag", None)
                    if mapping is not None and tag is not None and tag in mapping:
                        embodiment_id = int(mapping[tag])
                        break
            inputs["embodiment_id"] = torch.full(
                (bsz,), embodiment_id, dtype=torch.long, device=device
            )

        if "state_mask" not in inputs and "state" in inputs and torch.is_tensor(
            inputs["state"]
        ):
            state = inputs["state"]
            state_mask = torch.zeros_like(state, dtype=torch.bool, device=device)
            # LIBERO state dim = 8 (see DreamZeroLiberoDataset state_mask comment)
            valid_state_dim = 8
            try:
                metadata = self.config.data_transforms.dataset_metadata
                st = metadata.statistics.state["state"]
                valid_state_dim = len(np.asarray(st.q01))
            except Exception:
                valid_state_dim = min(
                    state.shape[-1],
                    int(getattr(self.config, "max_state_dim", state.shape[-1])),
                )
            valid_state_dim = min(valid_state_dim, state.shape[-1])
            state_mask[..., :valid_state_dim] = True
            inputs["state_mask"] = state_mask

        return inputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        # Mark the start of each training iteration so PyTorch knows when
        # to reclaim memory held by CUDA graphs from the previous iteration.
        torch.compiler.cudagraph_mark_step_begin()

        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")

        # Standard SFT dataloader already provides a full VLA batch dict.
        if "observation" not in data and "actions" not in data:
            outputs = super().forward(data)
        else:
            observation = data["observation"]
            actions = data["actions"]
            inputs = self._merge_sft_observation_and_actions(observation, actions)
            outputs = super().forward(inputs)

        loss = outputs.get("loss") if hasattr(outputs, "get") else None
        if loss is None:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError
