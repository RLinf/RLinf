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
import torch.nn as nn
import dataclasses
import logging
from tianshou.data import Batch
from omegaconf import OmegaConf, DictConfig
from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform import ComposedModalityTransform
import os
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


logger = logging.getLogger(__name__)


# Ensure groot is importable (dreamzero repo structure)
def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


class DreamZeroPolicy(nn.Module, BasePolicy):
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
        cpu_init: bool = False,
    ):
        nn.Module.__init__(self)
        _ensure_groot_importable()

        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

        self.model_path = Path(model_path)
        self.device = torch.device("cpu") if cpu_init else torch.device(
            device if isinstance(device, str) else f"cuda:{device}"
        )
        self.eval_bf16 = eval_bf16
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        exp_cfg_dir = self.model_path / "experiment_cfg"
        train_cfg_path = exp_cfg_dir / "conf.yaml"
        self.train_cfg = OmegaConf.create({})
        if train_cfg_path.exists():
            self.train_cfg = OmegaConf.load(train_cfg_path)
            self.eval_bf16 = self.train_cfg.get("eval_bf16", self.eval_bf16)
        else:
            logger.info(
                "DreamZero checkpoint has no experiment_cfg/conf.yaml: %s. "
                "SFT path is still supported; eval transform-dependent APIs are disabled.",
                train_cfg_path,
            )

        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Normalize target paths for checkpoints saved with old dreamvla module names.
        self._normalize_config_targets(config_dict)

        # Force IdentityBackbone for lightweight structure
        if force_identity_backbone:
            config_dict["backbone_cfg"] = {
                "_target_": "groot.vla.model.dreamzero.backbone.identity.IdentityBackbone"
            }

        # Patch sub-module pretrained paths to local Wan checkpoint if available
        self._patch_pretrained_paths(config_dict)

        config = VLAConfig(**config_dict)

        # Disable defer_lora_injection for immediate loading
        if "config" in config.action_head_cfg and isinstance(config.action_head_cfg["config"], dict):
            config.action_head_cfg["config"]["defer_lora_injection"] = False

        # Load model: use custom loading when forcing IdentityBackbone (from_pretrained ignores config)
        if force_identity_backbone:
            self.model = self._load_model_with_config(str(self.model_path), config)
        else:
            self.model = VLA.from_pretrained(str(self.model_path))

        if eval_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        if not cpu_init:
            self.model = self.model.to(device=self.device)

        self._fix_scalar_parameters()

        self._no_split_modules = [
            "WanAttentionBlock",
            "GanAttentionBlock",
            "CausalWanAttentionBlock",
            "DiTBlock",
        ]

        should_post_initialize = train_cfg_path.exists()
        if hasattr(self.model, "post_initialize") and should_post_initialize:
            try:
                self.model.post_initialize()
            except Exception as e:
                logger.warning("post_initialize skipped: %s", e)

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
        self.eval_transform = None
        metadata_path = exp_cfg_dir / "metadata.json"
        if (
            train_cfg_path.exists()
            and metadata_path.exists()
            and "transforms" in self.train_cfg
            and "oxe_droid" in self.train_cfg.transforms
        ):
            with open(metadata_path, "r") as f:
                metadatas = json.load(f)
            if "oxe_droid" in metadatas:
                metadata = DatasetMetadata.model_validate(metadatas["oxe_droid"])
                eval_transform = instantiate(self.train_cfg.transforms["oxe_droid"])
                assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
                eval_transform.set_metadata(metadata)
                eval_transform.eval()
                self.eval_transform = eval_transform

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"forward_type {forward_type} is not supported.")

    def default_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("DreamZero default_forward requires `data`.")
        return self.sft_forward(data=data)

    def sft_forward(self, data, **kwargs):
        inputs = data
        if not isinstance(inputs, dict):
            raise TypeError("DreamZero sft_forward expects a dict input batch.")
        outputs = self.model(inputs)
        if "loss" not in outputs:
            raise KeyError("DreamZero model output does not contain `loss`.")
        return outputs["loss"]

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        if self.eval_transform is None:
            raise RuntimeError("DreamZero eval transform is unavailable for this checkpoint.")
        obs = batch.obs

        normalized_input = self.eval_transform(obs)
        batch.normalized_obs = normalized_input
        return batch
    
    def unapply(self, batch: Batch, obs: dict = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        if self.eval_transform is None:
            raise RuntimeError("DreamZero eval transform is unavailable for this checkpoint.")
        unnormalized_action = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )
        
        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.train_cfg.get('relative_action', False)
        relative_action_per_horizon = self.train_cfg.get('relative_action_per_horizon', False)
        relative_action_keys = self.train_cfg.get('relative_action_keys', [])
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
                unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
        
        batch.act = unnormalized_action
        return batch

    def _patch_pretrained_paths(self, config_dict: dict) -> None:
        """Fill in local pretrained paths for sub-modules so VLA() never downloads."""
        wan_dir = self.model_path.parent / "Wan2.1-I2V-14B-480P"
        if not wan_dir.is_dir():
            return

        path_map = {
            "text_encoder_pretrained_path": wan_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            "image_encoder_pretrained_path": wan_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "vae_pretrained_path": wan_dir / "Wan2.1_VAE.pth",
        }

        ah_cfg = config_dict.get("action_head_cfg", {}).get("config", {})
        if not isinstance(ah_cfg, dict):
            return

        for sub_key in ("text_encoder_cfg", "image_encoder_cfg", "vae_cfg"):
            sub_cfg = ah_cfg.get(sub_key)
            if not isinstance(sub_cfg, dict):
                continue
            for path_key, local_path in path_map.items():
                if path_key in sub_cfg and local_path.exists():
                    sub_cfg[path_key] = str(local_path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the DiT backbone."""
        from torch.utils.checkpoint import checkpoint

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            if hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = checkpoint

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

    def _fix_scalar_parameters(self) -> None:
        """Reshape scalar (0-D) parameters to 1-D so FSDP can manage them."""
        for name, param in list(self.named_parameters()):
            if param.dim() == 0:
                parts = name.split(".")
                parent = self
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                new_param = torch.nn.Parameter(
                    param.data.unsqueeze(0), requires_grad=param.requires_grad
                )
                setattr(parent, parts[-1], new_param)
                logger.info("Reshaped scalar param %s to 1-D for FSDP", name)

    def _normalize_config_targets(self, config_dict: dict) -> None:
        """Rewrite legacy target paths so Hydra can instantiate modules in this repo."""
        def _rewrite(value):
            if isinstance(value, str):
                value = value.replace("groot.vla.model.dreamvla.", "groot.vla.model.dreamzero.")
                value = value.replace(
                    "wan_flow_matching_action_tf_efficient_weighted",
                    "wan_flow_matching_action_tf",
                )
                return value
            if isinstance(value, dict):
                return {k: _rewrite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_rewrite(v) for v in value]
            return value

        rewritten = _rewrite(config_dict)
        config_dict.clear()
        config_dict.update(rewritten)

    def _load_model_with_config(self, model_path: str, config) -> "VLA":
        """Load VLA with custom config (e.g. IdentityBackbone).

        If model safetensors are present, load them.
        If not, keep config-initialized weights and rely on WANPolicyHead
        component loading paths from config (Wan base training mode).
        """
        from groot.vla.model.dreamzero.base_vla import VLA

        safetensors_path = Path(model_path) / "model.safetensors"
        safetensors_index_path = Path(model_path) / "model.safetensors.index.json"
        has_weights = safetensors_index_path.exists() or safetensors_path.exists()

        model = VLA(config)

        if not has_weights:
            logger.info(
                "No model safetensors found at %s; initialized from config only.",
                model_path,
            )
            return model

        from safetensors.torch import load_file

        state_dict = {}

        if safetensors_index_path.exists():
            with open(safetensors_index_path) as f:
                index = json.load(f)
            for shard_file in set(index["weight_map"].values()):
                shard_path = Path(model_path) / shard_file
                state_dict.update(load_file(str(shard_path)))
        elif safetensors_path.exists():
            state_dict.update(load_file(str(safetensors_path)))

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
    
    def _convert_observation(self, env_obs: dict) -> dict:
        """Convert environment observation to model input"""
        # ---------- 1) RLinf obs -> DreamZero obs ----------
        main = env_obs["main_images"]
        extra = env_obs.get("extra_view_images", None)
        states = env_obs.get("states", None)
        task_desc = env_obs.get("task_descriptions", None)

        if torch.is_tensor(main):
            main_np = main.detach().cpu().numpy()
        else:
            main_np = np.asarray(main)

        B = main_np.shape[0]

        # get 3 images: ext0 / ext1 / wrist
        ext0 = main_np  # [B,H,W,C]
        if extra is not None:
            if torch.is_tensor(extra):
                extra_np = extra.detach().cpu().numpy()
            else:
                extra_np = np.asarray(extra)  # [B,N,H,W,C]
        else:
            extra_np = None

        if extra_np is not None and extra_np.ndim == 5 and extra_np.shape[1] > 0:
            ext1 = extra_np[:, 0]  # [B,H,W,C]
            wrist = extra_np[:, 1] if extra_np.shape[1] > 1 else extra_np[:, 0]
        else:
            ext1 = ext0
            wrist = ext0

        # states -> joint/gripper (according to panda-droid 8D convention)
        if states is not None:
            if torch.is_tensor(states):
                s_np = states.detach().cpu().numpy()
            else:
                s_np = np.asarray(states)
        else:
            s_np = np.zeros((B, 8), dtype=np.float32)

        if s_np.ndim == 1:
            s_np = s_np.reshape(1, -1)

        # prioritize joint(7)+gripper(1) if possible, otherwise fallback to 7D joint
        if s_np.shape[-1] >= 8:
            joint = s_np[:, :7]
            gripper = s_np[:, 7:8]
        elif s_np.shape[-1] >= 7:
            joint = s_np[:, :7]
            gripper = np.zeros((B, 1), dtype=s_np.dtype)
        else:
            joint = np.zeros((B, 7), dtype=np.float32)
            gripper = np.zeros((B, 1), dtype=np.float32)

        prompts = task_desc if task_desc is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B

        converted_obs = {
            "video.exterior_image_1_left": ext0,                     # [B,H,W,C]
            "video.exterior_image_2_left": ext1,                     # [B,H,W,C]
            "video.wrist_image_left": wrist,                         # [B,H,W,C]
            "state.joint_position": joint.astype(np.float64),        # [B,7]
            "state.gripper_position": gripper.astype(np.float64),    # [B,1]
            "annotation.language.action_text": list(prompts),        # list[str], len=B
        }
        return converted_obs
    
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

    def predict_action_batch(self, env_obs: dict) -> np.ndarray:
        """
        input:
        env_obs:
            - main_images: [B,H,W,C] uint8
            - extra_view_images: [B,N,H,W,C] or None
            - states: [B,D]
            - task_descriptions: list[str] or None
        output:
        actions: np.ndarray [B, num_action_chunks, 8]  # 7 joint + 1 gripper
        result: dict  # compatible with rollout interface"""
        converted_obs = self._convert_observation(env_obs)
        batch = Batch(obs=converted_obs)
        # relative action unnormalization needs to preserve original obs
        original_obs_for_relative = {
            k: v.copy() if isinstance(v, np.ndarray)
            else (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch.obs.items()
        }
        original_obs_for_relative = unsqueeze_dict_values(original_obs_for_relative)
        
        # ---------- 2) DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.model.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        self.video_across_time.append(video_pred)

        # 4. Unnormalize actions (pass obs for relative action normalization)
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs_for_relative)

        # 5. Remove batch dimension if we added it

        batch.act = squeeze_dict_values(batch.act)

        action_chunk_dict = batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        actions = self._convert_action(action_dict)

        forward_inputs = {
        "action": torch.as_tensor(actions).reshape(actions.shape[0], -1).cpu()
        if isinstance(actions, np.ndarray)
        else actions.reshape(actions.shape[0], -1).cpu(),
        }
        result = {
            "prev_logprobs": torch.zeros_like(forward_inputs["action"], dtype=torch.float32),
            "prev_values": torch.zeros((forward_inputs["action"].shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

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

def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.
    """
    model_cfg = cfg.actor.model if hasattr(cfg, "actor") and hasattr(cfg.actor, "model") else cfg
    model_path = model_cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = model_cfg.get("max_seq_len", 512)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = DreamZeroPolicy(
        model_path=model_path,
        device=device,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )

    return model