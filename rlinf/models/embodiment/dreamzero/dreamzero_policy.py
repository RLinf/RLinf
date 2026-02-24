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
# WITHOUT WARRANTIES OR CONDITIONS FOR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DreamZero policy implementing RLinf BasePolicy interface for DROID embodiment."""

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Batch

from rlinf.models.embodiment.base_policy import BasePolicy


def _ensure_groot_importable():
    """Ensure groot package is on Python path (dreamzero repo structure)."""
    if "groot" in sys.modules:
        return
    # Path: .../dreamzero/RLinf/rlinf/models/embodiment/dreamzero/dreamzero_policy.py
    # parents[5] = dreamzero repo root (contains RLinf/, dreamzero/, groot/)
    dreamzero_root = Path(__file__).resolve().parents[5]
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


def _convert_rlinf_obs_to_dreamzero(env_obs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert RLinf env observation to DreamZero/Groot format.

    RLinf format:
        - main_images: [B, H, W, C]
        - wrist_images: [B, H, W, C]
        - extra_view_images: [B, N_IMG, H, W, C] (optional)
        - states: [B, state_dim] (7 joint + 1 gripper = 8 for DROID)
        - task_descriptions: list[str]

    DreamZero DROID format:
        - video.exterior_image_1_left: [B, T, H, W, C]
        - video.exterior_image_2_left: [B, T, H, W, C]
        - video.wrist_image_left: [B, T, H, W, C]
        - state.joint_position: [B, 1, 7]
        - state.gripper_position: [B, 1, 1]
        - annotation.language.action_text: list[str]
    """
    droid_obs = {}

    # Images: add temporal dim [B, H, W, C] -> [B, 1, H, W, C]
    def add_time_dim(arr):
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 4:  # Already [B, H, W, C]
            return np.expand_dims(arr, axis=1)  # [B, 1, H, W, C]
        return arr

    main = env_obs["main_images"]
    wrist = env_obs.get("wrist_images")
    extra = env_obs.get("extra_view_images")

    droid_obs["video.exterior_image_1_left"] = add_time_dim(main)
    if extra is not None and extra.shape[1] >= 1:
        droid_obs["video.exterior_image_2_left"] = add_time_dim(extra[:, 0])
    else:
        droid_obs["video.exterior_image_2_left"] = add_time_dim(main)

    if wrist is not None:
        droid_obs["video.wrist_image_left"] = add_time_dim(wrist)
    else:
        droid_obs["video.wrist_image_left"] = add_time_dim(main)

    # State: [B, D] or [B, 1, D] -> joint [B, 1, 7], gripper [B, 1, 1]
    # DROID: 7 joint positions + 1 gripper = 8. Use first 7 for joint, last 1 for gripper.
    states = env_obs["states"]
    if torch.is_tensor(states):
        states = states.cpu().numpy()
    if states.ndim == 2:
        states = np.expand_dims(states, axis=1)
    state_dim = states.shape[-1]
    joint_dim = min(7, state_dim)
    droid_obs["state.joint_position"] = states[:, :, :joint_dim].astype(np.float64)
    if state_dim >= 8:
        droid_obs["state.gripper_position"] = states[:, :, 7:8].astype(np.float64)
    else:
        droid_obs["state.gripper_position"] = np.zeros(
            (*states.shape[:2], 1), dtype=np.float64
        )

    # Language
    task_desc = env_obs.get("task_descriptions", [])
    if isinstance(task_desc, str):
        task_desc = [task_desc]
    droid_obs["annotation.language.action_text"] = list(task_desc)

    return droid_obs


def _convert_dreamzero_action_to_rlinf(act_dict: dict[str, np.ndarray], num_chunks: int) -> np.ndarray:
    """
    Convert DreamZero action dict to RLinf format.

    DreamZero DROID: action.joint_position [B, T, 7], action.gripper_position [B, T, 1]
    RLinf: [B, num_action_chunks, 8] - 7 joints + 1 gripper
    """
    joint = act_dict.get("action.joint_position")
    gripper = act_dict.get("action.gripper_position")

    if joint is None:
        return np.zeros((1, num_chunks, 8), dtype=np.float32)

    if isinstance(joint, torch.Tensor):
        joint = joint.cpu().numpy()
    if isinstance(gripper, torch.Tensor):
        gripper = gripper.cpu().numpy()

    if joint.ndim == 2:
        joint = np.expand_dims(joint, axis=0)
    if gripper is None:
        gripper = np.zeros((joint.shape[0], joint.shape[1], 1), dtype=np.float32)
    elif gripper.ndim == 2:
        gripper = np.expand_dims(gripper, axis=-1)
        gripper = np.broadcast_to(gripper, (*gripper.shape[:-1], 1))

    joint = joint[:, :num_chunks, :7]
    gripper = gripper[:, :num_chunks, :1]
    # Binarize gripper for env compatibility (0/1)
    gripper_bin = (gripper > 0.5).astype(np.float32)
    actions = np.concatenate([joint, gripper_bin], axis=-1).astype(np.float32)
    return actions


class DreamZeroForRLActionPrediction(nn.Module, BasePolicy):
    """
    DreamZero policy wrapping GrootSimPolicy for RLinf embodied evaluation.

    Implements BasePolicy with:
    - predict_action_batch: inference for rollout/eval
    - default_forward: stub (DreamZero is inference-only, no online RL training)

    Action space: 8D (7 joint positions + 1 gripper) for DROID.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "oxe_droid",
        device: str | int = "cuda",
        num_action_chunks: int = 24,
        action_dim: int = 8,
    ):
        nn.Module.__init__(self)
        _ensure_groot_importable()

        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

        self.model_path = model_path
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.device = device
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim

        self._groot_policy = GrootSimPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=model_path,
            device=device,
            lazy_load=False,
        )
        self._groot_policy.eval()

    def default_forward(self, **kwargs):
        """DreamZero is inference-only; no online RL training support."""
        raise NotImplementedError(
            "DreamZero policy is inference-only. Use predict_action_batch for rollout/eval."
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Predict action chunk from env observation.

        Args:
            env_obs: RLinf env obs (main_images, wrist_images, states, task_descriptions)
            mode: train/eval (both use deterministic inference for DreamZero)

        Returns:
            actions: [B, num_action_chunks, 8] numpy
            result: dict with prev_logprobs, prev_values, forward_inputs for rollout compatibility
        """
        droid_obs = _convert_rlinf_obs_to_dreamzero(env_obs)

        # Ensure numpy for GrootSimPolicy
        for k, v in droid_obs.items():
            if torch.is_tensor(v):
                droid_obs[k] = v.cpu().numpy()

        batch = Batch(obs=droid_obs)
        result_batch = self._groot_policy.forward(batch)

        actions = _convert_dreamzero_action_to_rlinf(
            result_batch.act, self.num_action_chunks
        )

        batch_size = actions.shape[0]
        dev = torch.device(
            self.device if isinstance(self.device, str) else f"cuda:{self.device}"
        )

        actions_tensor = torch.from_numpy(actions).to(dev)
        result = {
            "prev_logprobs": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=dev),
            "prev_values": torch.zeros(batch_size, 1, device=dev),
            "forward_inputs": {"action": actions_tensor},
        }
        return actions, result
