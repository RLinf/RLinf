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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Optional
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tianshou.data import Batch
from einops import rearrange
import os
import datetime

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead


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
    DreamZero policy wrapping GrootSimPolicy for RLinf embodied evaluation and PPO.

    Implements BasePolicy with:
    - predict_action_batch: inference for rollout/eval
    - default_forward: PPO forward using Gaussian logprob around model prediction + optional value head

    Action space: 8D (7 joint positions + 1 gripper) for DROID.
    """

    # Fixed std for Gaussian logprob (policy is treated as deterministic + Gaussian for PPO)
    LOGPROB_STD: float = 0.1
    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "oxe_droid",
        device: str | int = "cuda",
        num_action_chunks: int = 24,
        action_dim: int = 8,
        add_value_head: bool = False,
        train_backbone: bool = False,
        video_output_dir: str = None,
    ):
        nn.Module.__init__(self)
        _ensure_groot_importable()

        from groot.vla.data.schema import EmbodimentTag
        from rlinf.models.embodiment.dreamzero.sim_policy import GrootSimPolicy

        self.model_path = model_path
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.device = device
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.add_value_head = add_value_head
        # Whether we allow gradients to flow into the underlying DreamZero backbone
        # (typically when using a LoRA checkpoint for RL fine-tuning).
        self.train_backbone = train_backbone
        self.video_output_dir = video_output_dir
        # Frame buffers for accumulation (per camera view)
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }

        self._call_count = 0
        self._is_first_call = True
        # Video across time for saving (similar to original server)
        self.video_across_time = []
        self._msg_index = 0

        self._groot_policy = GrootSimPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=model_path,
            device=device,
            lazy_load=False,
            enable_grad=train_backbone,
        )
        self._groot_policy.eval()

        if add_value_head:
            self.value_head = ValueHead(
                input_dim=num_action_chunks * action_dim,
                hidden_sizes=(256, 64),
                output_dim=1,
            )
        else:
            self.value_head = None

    def _run_policy_forward(
        self,
        env_obs: dict[str, Any],
        use_grad: bool = False,
    ) -> torch.Tensor:
        """Run GrootSimPolicy forward and return action tensor [B, num_chunks, action_dim].

        Args:
            env_obs: RLinf-formatted observations.
            use_grad: If True, keep the computation graph for PPO updates.
        """
        droid_obs = _convert_rlinf_obs_to_dreamzero(env_obs)
        for k, v in droid_obs.items():
            if torch.is_tensor(v):
                droid_obs[k] = v.detach().cpu().numpy()
        batch = Batch(obs=droid_obs)
        ctx = nullcontext() if use_grad else torch.no_grad()
        with ctx:
            result_batch = self._groot_policy.forward(batch)
        actions_np = _convert_dreamzero_action_to_rlinf(
            result_batch.act, self.num_action_chunks
        )
        dev = torch.device(
            self.device if isinstance(self.device, str) else f"cuda:{self.device}"
        )
        return torch.from_numpy(actions_np).float().to(dev)

    def default_forward(
        self,
        forward_inputs: dict[str, Any],
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        PPO forward: logprob under Gaussian(mean=model_pred, std=fixed), optional value head.
        """
        action = forward_inputs["action"]
        dev = action.device
        if action.dim() == 2:
            action = action.reshape(-1, self.num_action_chunks, self.action_dim)

        # Rebuild env_obs from forward_inputs (stored in predict_action_batch)
        env_obs = {}
        for key in ("main_images", "wrist_images", "states", "task_descriptions"):
            if key in forward_inputs:
                env_obs[key] = forward_inputs[key]
        if not env_obs:
            # Fallback: no obs -> use zeros logprobs/values so training does not crash
            batch_size = action.shape[0]
            logprobs = torch.zeros(
                batch_size, self.num_action_chunks, self.action_dim, device=dev
            )
            entropy = torch.zeros_like(logprobs)
            values = (
                self.value_head(action.reshape(batch_size, -1))
                if self.value_head is not None and compute_values
                else None
            )
            out = {"logprobs": logprobs}
            if compute_entropy:
                out["entropy"] = entropy
            if compute_values and values is not None:
                out["values"] = values
            return out

        # For PPO training, we want gradients to flow into the DreamZero backbone
        # only when explicitly enabled via `train_backbone`.
        action_pred = self._run_policy_forward(
            env_obs, use_grad=self.train_backbone
        )
        std = torch.full_like(action_pred, self.LOGPROB_STD, device=dev)
        dist = Normal(action_pred, std)

        output_dict = {}
        if compute_logprobs:
            output_dict["logprobs"] = dist.log_prob(action)
        if compute_entropy:
            output_dict["entropy"] = dist.entropy()
        if compute_values and self.value_head is not None:
            flat = action.reshape(action.shape[0], -1)
            output_dict["values"] = self.value_head(flat)
        elif compute_values and self.value_head is None:
            batch_size = action.shape[0]
            output_dict["values"] = torch.zeros(batch_size, 1, device=dev)
        return output_dict

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

        actions_tensor = torch.from_numpy(actions).float().to(dev)
        std = torch.full_like(actions_tensor, self.LOGPROB_STD, device=dev)
        dist = Normal(actions_tensor, std)
        prev_logprobs = dist.log_prob(actions_tensor)

        if self.value_head is not None:
            prev_values = self.value_head(actions_tensor.reshape(batch_size, -1))
        else:
            prev_values = torch.zeros(batch_size, 1, device=dev)

        forward_inputs = {"action": actions_tensor}
        for key in ("main_images", "wrist_images", "states", "task_descriptions"):
            if key in env_obs and env_obs[key] is not None:
                v = env_obs[key]
                if torch.is_tensor(v):
                    forward_inputs[key] = v.to(dev)
                else:
                    forward_inputs[key] = v

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.
        
        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str
        
        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}
        
        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }
        
        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK
        
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
        
        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
        
        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""
        
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action

    def eval(
        self,
        obs: dict,
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Evaluate DreamZero policy on a single environment observation.

        Args:
            obs: RLinf env obs (main_images, wrist_images, states, task_descriptions)
            mode: train/eval (both use deterministic inference for DreamZero)

        Returns:
            actions: [B, 8] numpy
            result: dict with prev_logprobs, prev_values, forward_inputs for rollout compatibility
        """
        # Convert observation format
        print("--------------from eval: obs------------------")
        print(obs)
        converted_obs = self._convert_observation(obs)
        print("--------------from eval: converted_obs------------------")
        print(converted_obs)
        batch = Batch(obs=converted_obs)
        # result_batch = self._groot_policy.forward(batch)
        # actions = result_batch.act
        # print("--------------from eval------------------")
        # print(result_batch)
        # print(actions)
        # print("--------------------------------")

        with torch.no_grad():
            result_batch, video_pred = self._groot_policy.lazy_joint_forward_causal(batch)
        print("--------------from eval: result_batch------------------")
        print(result_batch)
        print("--------------from eval: video_pred------------------")
        print(video_pred)
        print("--------------------------------")

        # Store video predictions for potential saving
        self.video_across_time.append(video_pred)

        # Extract and convert action
        action_chunk_dict = result_batch.act

        # Convert Batch to dict
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        
        action = self._convert_action(action_dict)

        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False

        #self._reset_state(save_video=self.video_output_dir is not None)
        return action

    def _reset_state(self, save_video: bool = False) -> None:
        """Internal method to reset policy state.
        
        Args:
            save_video: Whether to save accumulated video before reset.
        """
        # Optionally save accumulated video before reset
        if save_video and len(self.video_across_time) > 0 and self.video_output_dir:
            try:
                frame_list = []
                video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                frames = self._groot_policy.trained_model.action_head.vae.decode(
                    video_across_time_cat,
                    tiled=self._groot_policy.trained_model.action_head.tiled,
                    tile_size=(self._groot_policy.trained_model.action_head.tile_size_height, self._groot_policy.trained_model.action_head.tile_size_width),
                    tile_stride=(self._groot_policy.trained_model.action_head.tile_stride_height, self._groot_policy.trained_model.action_head.tile_stride_width),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")
                frames = frames[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                for frame in frames:
                    frame_list.append(frame)
                
                if len(frame_list) > 0:
                    sample_frame = frame_list[0]
                    if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                        save_dir = self.video_output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                        num_frames = len(frame_list)
                        n = (num_frames - 1) // 8
                        output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                        imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                        print(f"Saved video on reset to: {output_path}")
            except Exception as e:
                print(f"Failed to save video on reset: {e}")
        
        # Clear frame buffers
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

