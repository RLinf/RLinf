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
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.model.dreamzero.base_vla import VLA
from tianshou.data import Batch

from rlinf.data.datasets.dreamzero.data_transforms import (
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    rollout_obs_layout_for_embodiment,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.dreamzero_config import DreamZeroConfig


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
        embodiment_tag = config.embodiment_tag
        if embodiment_tag is None:
            raise ValueError(
                "DreamZeroPolicy requires config.embodiment_tag (set in get_model)."
            )
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(embodiment_tag)
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            config.data_transforms, embodiment_tag
        )
        self._action_keys = tuple(action_keys)

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
        """Run the forward modality pipeline on rollout observations.

        Input ``batch.obs`` is already in DreamZero modality keys (e.g.
        ``video.image``, ``state.state``, language key) from
        ``_observation_convert``. This method delegates to
        ``config.data_transforms``, built in ``get_model`` from Hydra cfg and
        ``metadata.json`` (via ``load_dreamzero_dataset_metadata`` +
        ``data_transforms.set_metadata``).

        Pipeline (libero_sim example, see ``libero_sim._build_composed_transform``):

        1. Video / state / action preprocessing and normalization
           (``StateActionTransform`` uses q99 stats from metadata).
        2. ``ConcatTransform.apply``: concat per-key tensors into flat
           ``state`` / ``action`` vectors. Per-key widths come from metadata
           (e.g. ``action.actions`` shape ``[7]`` for Libero).
        3. ``DreamTransform.apply``: pad state/action to ``max_state_dim`` /
           ``max_action_dim`` (typically 32 from yaml) so the WAN action head
           always sees a fixed width. Extra padded dims are zeros and masked
           during training; at inference the model still outputs width 32.

        The returned ``batch.normalized_obs`` is the dict consumed by
        ``lazy_joint_video_action_causal`` (tokens, video, padded actions, etc.).
        """
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Invert model actions back to environment-scale per-modality tensors.

        ``batch.normalized_action`` is ``action_pred`` from the WAN head, shape
        ``[..., max_action_dim]`` (e.g. 32), matching the padded width from
        ``DreamTransform.apply``. Environment DOF is smaller (e.g. Libero 7);
        that width is **not** taken from Hydra ``action_dim`` on the policy—it
        comes from ``metadata.json`` loaded at build time:

        - ``get_model`` calls ``data_transforms.set_metadata(metadata)``.
        - ``ConcatTransform.set_metadata`` sets ``action_dims["action.actions"]``
          from ``metadata.modalities.action.<key>.shape[0]`` (7 for libero_sim).
        - On ``unapply``, transforms run in reverse order:
          ``DreamTransform.unapply`` (passthrough) →
          ``ConcatTransform.unapply`` slices ``[..., 0:env_dim]`` per
          ``action_concat_order`` → ``StateActionTransform.unapply`` reverses
          q99 normalization.

        Output is a dict like ``{"action.actions": tensor}`` with **env** width
        (7 for Libero). ``predict_action_batch`` then merges keys via
        ``_actions_from_unapply`` for the sim.

        If ``relative_action`` / ``relative_action_per_horizon`` is enabled,
        optionally adds the last ``state.*`` from ``obs`` (converted rollout
        obs passed from ``predict_action_batch``) to obtain absolute actions.
        """
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

    def _process_batch(self, batch: Batch) -> dict[str, Any]:
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
        """Map RLinf rollout observations to DreamZero modality keys."""
        return convert_rollout_env_obs(self.config.embodiment_tag, env_obs)

    def _actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate per-key unnormalized actions in dataset concat order."""
        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - wrist_images: [B,H,W,C] (optional, embodiment-specific)
                - extra_view_images: [B,N,H,W,C] (optional, e.g. oxe_droid)
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, action_dim]
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        batch = self.unapply(
            Batch(normalized_action=normalized_action),
            obs=converted_obs,
        )
        actions = self._actions_from_unapply(batch.act)

        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
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
        # Mark the start of each training iteration so PyTorch knows when
        # to reclaim memory held by CUDA graphs from the previous iteration.
        torch.compiler.cudagraph_mark_step_begin()

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
