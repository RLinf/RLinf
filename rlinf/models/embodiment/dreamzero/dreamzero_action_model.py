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

"""DreamZero action model with IdentityBackbone + WANPolicyHead for eval and SFT."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from hydra.utils import instantiate
import argparse
import gymnasium as gym
import cv2
import mediapy
from tqdm import tqdm
import numpy as np
import torch
import dataclasses
import logging
from tianshou.data import Batch
from omegaconf import OmegaConf
from groot.vla.data.schema import DatasetMetadata, EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
import uuid
import datetime
import imageio

# Ensure groot is importable (dreamzero repo structure)
def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


# OXE_DROID embodiment id (used in DreamZero DROID)
EMBODIMENT_ID_OXE_DROID = 17
# Default tokenizer for text (matches DreamZero training)
TOKENIZER_NAME = "google/umt5-xxl"
MAX_SEQ_LEN = 512


class DreamZeroActionModel:
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead.

    - predict_action_batch: for eval (inference)
    - forward: for SFT training (returns loss)
    """

    def __init__(
        self,
        model_path: str,
        device: str | int = "cuda",
        eval_bf16: bool = True,
        force_identity_backbone: bool = True,
    ):
        _ensure_groot_importable()

        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

        self.model_path = Path(model_path)
        self.device = torch.device(device if isinstance(device, str) else f"cuda:{device}")
        self.eval_bf16 = eval_bf16
        self._tokenizer = None
        exp_cfg_dir = self.model_path / "experiment_cfg"
        train_cfg_path = exp_cfg_dir / "conf.yaml"
        train_cfg = OmegaConf.load(train_cfg_path)
        self.train_cfg = train_cfg
        self.eval_bf16 = self.train_cfg.get("eval_bf16", False)

        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Force IdentityBackbone for lightweight structure
        if force_identity_backbone:
            config_dict["backbone_cfg"] = {
                "_target_": "groot.vla.model.dreamzero.backbone.identity.IdentityBackbone"
            }

        config = VLAConfig(**config_dict)

        # Disable defer_lora_injection for immediate loading
        if "config" in config.action_head_cfg and isinstance(config.action_head_cfg["config"], dict):
            config.action_head_cfg["config"]["defer_lora_injection"] = False

        # Load model: use custom loading when forcing IdentityBackbone (from_pretrained ignores config)
        if force_identity_backbone:
            self.model = self._load_model_with_config(str(self.model_path), config)
        else:
            self.model = VLA.from_pretrained(str(self.model_path))
        self.model.eval()

        if eval_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        self.model = self.model.to(device=self.device)

        if hasattr(self.model, "post_initialize"):
            try:
                self.model.post_initialize()
            except Exception as e:
                print(f"post_initialize skipped: {e}")

        self.action_horizon = self.model.action_horizon
        self.action_dim = self.model.action_dim
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": []
        }
        self._is_first_call = True
        self._call_count = 0
        self.video_across_time = []

        self._pending_actions = None
        self._pending_idx = 0
        self._chunk_action_horizon = 24

        # 2. Load the action, video, and state transforms
        # 2.1. Load the metadata for normalization stats
        # We have an assumption: one policy is only for rolling out one type of env, i.e., one embodiment_tag
        # metadata_versions = train_cfg.metadata_versions
        # metadata = get_metadata(self.embodiment_tag, metadata_versions[self.embodiment_tag.value])
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        metadata = DatasetMetadata.model_validate(metadatas["oxe_droid"])

        # 2.2. Get the eval transforms

        eval_transform_cfg = train_cfg.transforms["oxe_droid"]

        eval_transform = instantiate(train_cfg.transforms["oxe_droid"])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs

        normalized_input = self.eval_transform(obs)
        batch.normalized_obs = normalized_input
        return batch
    
    def unapply(self, batch: Batch, obs: dict = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )
        
        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.train_cfg.get('relative_action', False)
        relative_action_per_horizon = self.train_cfg.get('relative_action_per_horizon', False)
        relative_action_keys = self.train_cfg.get('relative_action_keys', [])
        print("relative_action_per_horizon", relative_action_per_horizon)
        if (relative_action or relative_action_per_horizon) and relative_action_keys and obs is not None:
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
                        if 'state' in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break
                    
                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and 'state' in obs:
                        state_data = obs['state']
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
                    last_state = np.expand_dims(last_state, axis=-2)  # Add horizon dimension
                
                # Add state to relative action to get absolute action
                print("last_state", last_state.shape, "unnormalized_action[action_key]", unnormalized_action[action_key].shape)
                unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
        
        batch.act = unnormalized_action
        return batch

    def _load_model_with_config(self, model_path: str, config) -> "VLA":
        """Load VLA with custom config (e.g. IdentityBackbone) and weights."""
        from safetensors.torch import load_file

        from groot.vla.model.dreamzero.base_vla import VLA

        state_dict = {}
        safetensors_path = Path(model_path) / "model.safetensors"
        safetensors_index_path = Path(model_path) / "model.safetensors.index.json"

        if safetensors_index_path.exists():
            with open(safetensors_index_path) as f:
                index = json.load(f)
            for shard_file in set(index["weight_map"].values()):
                shard_path = Path(model_path) / shard_file
                state_dict.update(load_file(str(shard_path)))
        elif safetensors_path.exists():
            state_dict.update(load_file(str(safetensors_path)))
        else:
            raise FileNotFoundError(f"No weights at {model_path}")

        model = VLA(config)
        if hasattr(model, "post_initialize"):
            model.post_initialize()

        has_base_layer = any(".base_layer." in k for k in state_dict)
        if has_base_layer:
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        return model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
            self._tokenizer = HuggingfaceTokenizer(
                name=TOKENIZER_NAME,
                seq_len=MAX_SEQ_LEN,
                clean="whitespace",
            )
        return self._tokenizer

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
         # 1. check if the observation is batched
        def _is_batched(obs: dict) -> bool:
            for k, v in obs.items():
                if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                    return False
            return True

        # 2. ensure the observation has batch dimension
        is_batched = _is_batched(batch.obs)

        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 3. normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # 4. if the normalized input is still a Batch, flatten it into a pure dict (same as sim_policy)
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
         # 5. do bf16 cast if needed (same as sim_policy's eval_bf16 logic)
        # here we assume DreamZeroActionModel has self.eval_bf16 / self.device
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)
        return normalized_input
    
    def _process_obs(self, obs: dict) -> dict:
        """Process observation."""
        normalized_input = self.eval_transform(obs)

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

    @torch.no_grad()
    def predict_action_batch(self, obs: dict) -> np.ndarray:
        """Eval: predict actions from batch.

        Args:
            obs: observation dict

        Returns:
            actions: (B, action_horizon, action_dim) numpy array
        """
        self.model.eval()
        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.model.lazy_joint_video_action_causal(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        self.video_across_time.append(video_pred)
        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it

        batch.act = squeeze_dict_values(batch.act)

        action_chunk_dict = batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        action = self._convert_action(action_dict)
        return action

    def predict_action_one_step(self, obs: dict) -> np.ndarray:
        if self._pending_idx == 0 or self._pending_idx >= self._chunk_action_horizon:
            converted_obs = self._convert_observation(obs)
            batch = Batch(obs=converted_obs)
            original_obs_for_relative = {k: v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v 
                                     for k, v in batch.obs.items()}
            original_obs_for_relative = unsqueeze_dict_values(original_obs_for_relative)
            normalized_input = self._process_batch(batch)
            with torch.no_grad():
                model_pred = self.model.lazy_joint_video_action_causal(normalized_input)
            normalized_action = model_pred["action_pred"].float()
            video_pred = model_pred["video_pred"]

            self.video_across_time.append(video_pred)
            # 4. Unnormalize actions (pass obs for relative action conversion)
            batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs_for_relative)

            # 5. Remove batch dimension if we added it

            batch.act = squeeze_dict_values(batch.act)

            action_chunk_dict = batch.act
            action_dict = {}
            for k in dir(action_chunk_dict):
                if k.startswith("action."):
                    action_dict[k] = getattr(action_chunk_dict, k)
            actions = self._convert_action(action_dict)
            self._pending_actions = actions
            self._pending_idx = 0
        action = self._pending_actions[self._pending_idx]
        self._pending_idx += 1
        return action

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """SFT training: forward pass and return loss.

        Args:
            batch: dict with obs keys + "action" (B, horizon, action_dim) normalized to [-1, 1]

        Returns:
            dict with "loss", optionally "action_pred"
        """
        self.model.train()
        inputs = self._prepare_inputs(batch, for_training=True)

        if inputs.get("images") is None:
            raise ValueError("batch must contain video/images for forward")

        if "action" not in inputs:
            raise ValueError("batch must contain 'action' for SFT forward")

        out = self.model.forward(inputs)
        result = {"loss": out["loss"]}
        if "action_pred" in out:
            result["action_pred"] = out["action_pred"]
        return result

@dataclasses.dataclass
class PolicyImageConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    # It's beneficial to resize images to the desired resolution client-side for faster communication.
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    n_external_cameras: int = 1  # can be in [0, 1, 2]
    # Whether or not stereo camera image(s) should be sent.
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent (e.g. for policies that want to keep track of history).
    needs_session_id: bool = False
    # Which action space to use.
    action_space: str = "joint_position"  # can be in ["joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity"]

def _make_zero_observation(
    image_config: PolicyImageConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.
    
    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}
    if image_config.image_resolution is not None:
        h, w = image_config.image_resolution
    else:
        h, w = 180, 320
    # external cameras (0 - index)
    for i in range(image_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if image_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # wrist camera
    if image_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if image_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # session id
    if image_config.needs_session_id:
        import uuid
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    # state observations (7 DOF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)

    # prompt
    obs["prompt"] = prompt

    return obs

def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build an observation dict from real video frames.

    For 1 frame: each image key is (H, W, 3).
    For 4 frames: each image key is (4, H, W, 3).
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs

def droid_obs_to_roboarena(
    env_obs: Dict[str, Any],
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> Dict[str, Any]:
    """convert DROID env obs to the same format as _make_zero_observation"""
    policy = env_obs.get("policy", env_obs)

    # 1. cameras: external_cam / external_cam_2 / wrist_cam -> observation/*, and resize to 180x320
    ext1 = policy["external_cam"]      # (1, H, W, 3)
    ext2 = policy["external_cam_2"]    # (1, H, W, 3)
    wrist = policy["wrist_cam"]        # (1, H, W, 3)

    def _to_numpy_hw3_first(t):
        if isinstance(t, torch.Tensor):
            arr = t[0].detach().cpu().numpy()   # (H, W, 3)
        else:
            arr = np.asarray(t)[0]
        return arr

    def _resize_hw(img: np.ndarray, h: int, w: int) -> np.ndarray:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    h, w = 180, 320
    ext1_np = _resize_hw(_to_numpy_hw3_first(ext1), h, w)    # (h, w, 3)
    ext2_np = _resize_hw(_to_numpy_hw3_first(ext2), h, w)
    wrist_np = _resize_hw(_to_numpy_hw3_first(wrist), h, w)

    obs: Dict[str, Any] = {}
    # external cameras (0 / 1)
    obs["observation/exterior_image_0_left"] = ext1_np
    obs["observation/exterior_image_1_left"] = ext2_np
    # wrist camera
    obs["observation/wrist_image_left"] = wrist_np

    # 2. joints / gripper: arm_joint_pos -> observation/joint_position, gripper_pos -> observation/gripper_position
    arm = policy["arm_joint_pos"]      # (7,)
    grip = policy["gripper_pos"]       # (1,)

    if isinstance(arm, torch.Tensor):
        arm_np = arm.detach().cpu().numpy().astype(np.float32)
    else:
        arm_np = np.asarray(arm, dtype=np.float32)

    if isinstance(grip, torch.Tensor):
        grip_np = grip.detach().cpu().numpy().astype(np.float32)
    else:
        grip_np = np.asarray(grip, dtype=np.float32)

    obs["observation/joint_position"] = arm_np           # (7,)
    obs["observation/gripper_position"] = grip_np        # (1,)
    # cartesian_position is not provided by the env, so we fill it with 0 vector
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)

    # 3. session_id and prompt
    if session_id is None:
        session_id = str(uuid.uuid4())
    obs["session_id"] = session_id
    obs["prompt"] = prompt

    return obs

def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data

def action_to_droid(action: np.ndarray) -> Dict[str, Any]:
    T = min(12, action.shape[0])
    action_tensor = []
    for i in range(T):
        action = action[i]

        if action[-1] > 0.5:
            action = np.concatenate([action[:-1], np.ones(1)],dtype=action.dtype)
        else:
            action = np.concatenate([action[:-1], np.zeros(1)],dtype=action.dtype)
        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device="cuda")

    return action_tensor

    
def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            unsqueezed_data[k] = np.array([v])
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data

def test_droid_env():
    _ensure_groot_importable()
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    print("parser", parser)
    AppLauncher.add_app_launcher_args(parser)

    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    scene = 1
    episodes = 10
    headless = True
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "pick up the can and put it in the mug"
        case 3:
            instruction = "put the banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    import uuid
    session_id = str(uuid.uuid4())

    video_dir = Path("videoOutputs")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    #max_steps = 100
    model = DreamZeroActionModel(model_path="path/to/models/DreamZero-DROID", device="cuda")
    with torch.no_grad():
        for ep in range(episodes):
            for _step in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                roboarena_obs = droid_obs_to_roboarena(obs, instruction, session_id)
                normalized_action = model.predict_action_one_step(roboarena_obs)

                normalized_action = torch.from_numpy(normalized_action).float()
                action = normalized_action.clone()      # (8,)

                # binary gripper
                if action[-1].item() > 0.5:
                    action[-1] = 1.0
                else:
                    action[-1] = 0.0

                action_tensor = action.unsqueeze(0).to(device="cuda")  # (1, 8)

                obs, reward, term, trunc, _  = env.step(action_tensor)

                policy = obs.get("policy", obs)
                ext1 = policy["external_cam"]      # (1, H, W, 3)
                ext2 = policy["external_cam_2"]    # (1, H, W, 3)
                wrist = policy["wrist_cam"]        # (1, H, W, 3)
                def _to_numpy_hw3_first(t: torch.Tensor) -> np.ndarray:
                    if isinstance(t, torch.Tensor):
                        arr = t[0].detach().cpu().numpy()   # (H, W, 3)
                    else:
                        arr = np.asarray(t)[0]
                    return arr
                ext1_np = _to_numpy_hw3_first(ext1)
                ext2_np = _to_numpy_hw3_first(ext2)
                wrist_np = _to_numpy_hw3_first(wrist)
                frame = np.concatenate([ext1_np, ext2_np, wrist_np], axis=1)
                video.append(frame)
                if term or trunc:
                    break
            imageio.mimsave(video_dir / f"episode_{ep}.mp4", video, fps=15, codec='libx264')
            frames = []
            for v in model.video_across_time:    # v 是 torch.Tensor, shape 类似 (T, H, W, C) 或 (H, W, C)
                if isinstance(v, torch.Tensor):
                    frames.append(v.detach().cpu().to(torch.float32).numpy())
                else:
                    frames.append(np.asarray(v))

            imageio.mimsave(
                video_dir / f"episode_{ep}_model_video.mp4",
                frames,          # now it's a list of numpy arrays
                fps=15,
                codec='libx264',
            )
            video = []
            model._pending_idx = 0
            model._pending_actions = None
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    test_droid_env()