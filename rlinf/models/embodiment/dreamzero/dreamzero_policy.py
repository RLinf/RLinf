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

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config

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
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError
