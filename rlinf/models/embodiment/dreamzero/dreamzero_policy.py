import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from hydra.utils import instantiate
#import argparse
#import gymnasium as gym
#import cv2
#import mediapy
from tqdm import tqdm
import numpy as np
import torch
#import dataclasses
#import logging
from tianshou.data import Batch
from omegaconf import OmegaConf, DictConfig
# from groot.vla.data.schema import DatasetMetadata
# from groot.vla.data.transform import ComposedModalityTransform
import os
from rlinf.models.embodiment.base_policy import BasePolicy
# Ensure groot is importable (dreamzero repo structure)
def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    print("================= dreamzero_root ==================")
    print("dreamzero_root", dreamzero_root)
    dreamzero_root = dreamzero_root / "dreamzero"
    print("================= dreamzero_root ==================")
    print("dreamzero_root", dreamzero_root)
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


class DreamZeroPolicy(BasePolicy):
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
        from groot.vla.data.schema import DatasetMetadata
        from groot.vla.data.transform import ComposedModalityTransform

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

        #eval_transform_cfg = train_cfg.transforms["oxe_droid"]

        print("================= eval_transform_cfg ==================")
        print(train_cfg.transforms["oxe_droid"])

        train_cfg.transforms["oxe_droid"].transforms[-1].tokenizer_path = tokenizer_path
        print("================= after tokenizer_path ==================")
        print(tokenizer_path)
        print(train_cfg.transforms["oxe_droid"])


        eval_transform = instantiate(train_cfg.transforms["oxe_droid"])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform

    def eval(self):
        self.model.eval()
        #self.eval_transform.eval()
        return self
    
    def cuda(self):
        self.model = self.model.cuda()
        #self.eval_transform = self.eval_transform.cuda()
        return self
    def to(self, device: str | int):
        self.model = self.model.to(device)
        #self.eval_transform = self.eval_transform.to(device)
        return self 

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
        wrist = env_obs.get("wrist_images",None)
        states = env_obs.get("states", None)
        task_desc = env_obs.get("task_descriptions", None)
        def _ensure_video_bt_hwc(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 4:
                x = x[:, None, ...]   # [B,1,H,W,C]
            elif x.ndim != 5: # expect [B,T,H,W,C]
                raise ValueError(f"Unexpected image shape: {x.shape}")
            return x

        import cv2

        def _resize_bt_hwc_uint8(x, h=180, w=320):
            # x: [B,H,W,C
            B = x.shape[0]
            out = np.empty((B, h, w, 3), dtype=np.uint8)
            for b in range(B):
                frame = x[b]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out[b] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return out


        if torch.is_tensor(main):
            main_np = main.detach().cpu().numpy()
        else:
            main_np = np.asarray(main)
        print("=========main.shape==============")
        print(main.shape)
        #main_np = _ensure_video_bt_hwc(main)
        #print("===========ensure.shape============")
        #print(main_np.shape)
        main_np = _resize_bt_hwc_uint8(main_np)
        print("===========resize.shape===============")
        print(main_np.shape)

        B = main_np.shape[0]

        # get 3 images: ext0 / ext1 / wrist
        #ext0 = main_np  # [B,H,W,C]
        #if extra is not None:
        #    if torch.is_tensor(extra):
        #        extra_np = extra.detach().cpu().numpy()
        #    else:
        #        extra_np = np.asarray(extra)  # [B,N,H,W,C]
        #else:
        #    extra_np = None

        #if extra_np is not None and extra_np.ndim == 5 and extra_np.shape[1] > 0:
        #    ext1 = extra_np[:, 0]  # [B,H,W,C]
            #wrist = extra_np[:, 1] if extra_np.shape[1] > 1 else extra_np[:, 0]
        #else:
        #    ext1 = ext0
            #wrist = ext0
        #if wrist is not None:
        #    if torch.is_tensor(wrist):
        #        wrist_np = wrist.detach().cpu().numpy()
        #    else:
        #        wrist_np = np.asarray(wrist)  # [B,N,H,W,C]
        #else:
        #    extra_np = None

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
            "video.exterior_image_1_left": main_np,                     # [B,H,W,C]
            "video.exterior_image_2_left": main_np,                     # [B,H,W,C]
            "video.wrist_image_left": main_np,                         # [B,H,W,C]
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

    def predict_action_batch(self, env_obs, mode,**kwargs) -> np.ndarray:
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
        print("================= env_obs ==================")
        print(env_obs)
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

        #batch.act = squeeze_dict_values(batch.act)
        print("===========batch.act===============")
        print(batch.act.shape)
        print(batch.act)

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
    def default_forward(
    self,
    forward_inputs: dict[str, Any],
    compute_logprobs: bool = True,
    compute_entropy: bool = True,
    compute_values: bool = True,
    **kwargs,
    ) -> dict[str, Any]:
        """
        DreamZero PPO/GRPO forward for embodied training.

        Expected output keys (same style as openpi_action_model.py):
        - logprobs: Tensor [B, num_chunks, action_dim] (float32)
        - values:   Tensor [B, 1] 或 None
        - entropy:  Tensor [B, num_chunks, action_dim] 或 None

        Notes:
        - rollout/huggingface_worker 当前 DreamZeroPolicy 的 forward_inputs 可能只含 "action"，
        没有图像/状态等观测；这种情况下我们只能退化为用 action 直接构造分布（可跑但梯度可能为 0）。
        """
        device = None

        if forward_inputs is None or "action" not in forward_inputs:
            # 兜底：尽量不要 crash，返回空 logprobs（shape 无法推断就按 1x1xaction_dim）
            action_dim = getattr(self, "action_dim", 8)
            device = next(self.parameters()).device
            logprobs = torch.zeros((1, 1, action_dim), device=device, dtype=torch.float32)
            out = {"logprobs": logprobs}
            if compute_entropy:
                out["entropy"] = torch.zeros_like(logprobs)
            if compute_values:
                out["values"] = torch.zeros((1, 1), device=device, dtype=torch.float32)
            return out

        action_in = forward_inputs["action"]
        if not torch.is_tensor(action_in):
            action_in = torch.as_tensor(action_in)

        action_in = action_in.to(dtype=torch.float32)
        device = action_in.device
        action_dim = getattr(self, "action_dim", None)
        num_chunks = getattr(self, "_chunk_action_horizon", None)

        if action_dim is None:
            action_dim = 8
        if num_chunks is None:
            num_chunks = 24

        # reshape to [B, num_chunks, action_dim]
        if action_in.dim() == 2:
            B = action_in.shape[0]
            action = action_in.reshape(B, num_chunks, action_dim)
        elif action_in.dim() == 3:
            action = action_in
        else:
            # best-effort
            B = action_in.shape[0]
            action = action_in.reshape(B, num_chunks, action_dim)

        # ---- 1) 估计当前策略的均值 mu ----
        # 如果 forward_inputs 里带了观测，就用 model 预测 mu；
        # 否则退化：mu = action（这能保证数值可用，但梯度可能为 0）
        mu = None

        obs_keys = ("main_images", "extra_view_images", "states", "task_descriptions")
        has_main_images = "main_images" in forward_inputs
        has_states = ("states" in forward_inputs) or ("states" not in forward_inputs)

        if has_main_images and has_states:
            env_obs = {
                "main_images": forward_inputs.get("main_images"),
                "extra_view_images": forward_inputs.get("extra_view_images", None),
                "states": forward_inputs.get("states", None),
                "task_descriptions": forward_inputs.get("task_descriptions", None),
            }

            # Convert + process (沿用你 predict_action_batch 里的工具函数)
            droid_obs = self._convert_observation(env_obs)
            # original_obs_for_relative：用于 unapply 的相对动作反归一化
            original_obs_for_relative = {
                k: (v.copy() if isinstance(v, np.ndarray) else (v.clone() if torch.is_tensor(v) else v))
                for k, v in droid_obs.items()
            }
            # Batch -> apply/process
            batch = Batch(obs=droid_obs)
            # 参考 predict_action_batch：unsqueeze 让 batch 维度一致
            original_obs_for_relative = unsqueeze_dict_values(original_obs_for_relative)

            # 你当前实现里 _process_batch 返回 normalized_input (dict)
            normalized_input = self._process_batch(batch)
            # 训练时需要梯度：只有 compute_values/compute_logprobs 可能为真（actor worker 里 compute_logprobs=True）
            ctx = nullcontext() if (compute_logprobs or compute_values) else torch.no_grad()
            with ctx:
                model_pred = self.model.lazy_joint_video_action_causal(normalized_input)

            normalized_action = model_pred["action_pred"].float()

            # unapply：把 normalized_action 转回动作空间，再走 _convert_action
            batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs_for_relative)
            batch.act = squeeze_dict_values(batch.act)

            action_chunk_dict = batch.act
            action_dict = {}
            for k in dir(action_chunk_dict):
                if k.startswith("action."):
                    action_dict[k] = getattr(action_chunk_dict, k)

            actions_pred_np = self._convert_action(action_dict)  # numpy
            mu = torch.as_tensor(actions_pred_np, device=device, dtype=torch.float32)

            # reshape mu to [B, num_chunks, action_dim] if possible
            if mu.dim() == 2:
                # e.g. [B, num_chunks*action_dim]
                if mu.shape[0] == action.shape[0]:
                    mu = mu.reshape(action.shape[0], num_chunks, action_dim)
            elif mu.dim() == 3:
                pass
            else:
                mu = mu.reshape(action.shape[0], num_chunks, action_dim)

        if mu is None:
            # 退化策略：mu = action（避免缺字段导致 crash）
            mu = action.detach() if not (compute_logprobs or compute_values) else action

        # ---- 2) 固定方差的高斯分布 ----
        std_val = 0.1  # DreamZero old 版本是 0.1
        std = torch.full_like(mu, std_val)

        dist = torch.distributions.Normal(mu, std)

        out: dict[str, Any] = {}

        # actor worker 里 compute_logprobs 传 True，因此这里默认算 logprobs
        logprobs = dist.log_prob(action)  # [B, num_chunks, action_dim]
        out["logprobs"] = logprobs.to(dtype=torch.float32)

        if compute_entropy:
            out["entropy"] = dist.entropy().to(dtype=torch.float32)

        if compute_values:
            # 当前 DreamZeroPolicy (new) 没有 value_head；先返回 zeros，保证结构正确
            # 如果你后续加了 value_head，这里改成 value_head(action.reshape(B, -1))
            B = action.shape[0]
            out["values"] = torch.zeros((B, 1), device=device, dtype=torch.float32)

        return out

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
