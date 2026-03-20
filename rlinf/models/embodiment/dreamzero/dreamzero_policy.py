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
from omegaconf import OmegaConf, DictConfig
from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform import ComposedModalityTransform
import os
# Ensure groot is importable (dreamzero repo structure)
def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


class DreamZeroPolicy:
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
        tokenizer_path: str = "google/umt5-xxl",
        max_seq_len: int = 512,
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
        #self.model.eval()

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
                name=self.tokenizer_path,
                seq_len=self.max_seq_len,
                clean="whitespace",
            )
        return self._tokenizer

def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.
    """
    model_path = cfg.actor.model.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    tokenizer_path = cfg.actor.model.get("tokenizer_path", "google/umt5-xxl")
    print("tokenizer_path", tokenizer_path)
    max_seq_len = cfg.actor.model.get("max_seq_len", 512)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = DreamZeroPolicy(
        model_path=model_path,
        device=device,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )

    return model