import importlib
import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from tianshou.data import Batch
from tianshou.policy import BasePolicy as BaseTianshouPolicy
import torch
import tree
import time

from groot.vla.data.schema import DatasetMetadata, EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform


class ModelManager:
    """
    Manages model loading/offloading to handle memory efficiently when using multiple models.
    Modified to keep VLM always loaded and only manage text_encoder/vae components.
    """
    def __init__(self):
        self.active_components = None  # Track which components are active for action_head
        self.models = {}
        self.vlm_policy = None  # Keep VLM always loaded
        self.action_head_policy = None  # Action head policy for component management
        
    def register_model(self, name: str, policy_instance):
        """Register a policy instance with the manager."""
        self.models[name] = policy_instance
        
        if name == "vlm":
            self.vlm_policy = policy_instance
            # Load VLM to GPU and keep it there
            self.load_vlm_model()
        elif name == "action_head":
            self.action_head_policy = policy_instance
            # Ensure vram management is enabled for action head
            self.enable_action_head_vram_management()
            # Initially offload action head components to save memory
            self.offload_action_head_components()
    
    def load_vlm_model(self):
        """Load VLM model to GPU and keep it there."""
        if self.vlm_policy is None:
            return
            
        policy = self.vlm_policy
        print(f"Loading VLM model to GPU (keeping it loaded)...")
        
        # Move VLM model to GPU
        policy.trained_model.to(device=policy.device)
            
        # Apply bf16 if needed
        if policy.eval_bf16:
            policy.trained_model = policy.trained_model.to(dtype=torch.bfloat16)
            
        torch.cuda.empty_cache()  # Clear cache after loading
    
    def activate_model(self, name: str):
        """Activate a model - for VLM this is a no-op, for action_head this manages components."""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
            
        if name == "vlm":
            # VLM is always loaded, just ensure it's ready
            if self.vlm_policy:
                self.vlm_policy.ensure_model_on_gpu()
            return
        elif name == "action_head":
            # For action head, activate the text_encoder and vae components
            self.activate_action_head_components(['text_encoder', 'vae'])
            self.active_components = ['text_encoder', 'vae']
        
    def activate_action_head_components(self, component_names):
        """Activate specific components of the action head model."""
        if self.action_head_policy is None:
            return
            
        policy = self.action_head_policy
        print(f"Loading action head components {component_names} to GPU...")
        
        # Check if the action head has the load_models_to_device method
        if (hasattr(policy.trained_model, "action_head") and 
            hasattr(policy.trained_model.action_head, "load_models_to_device")):
            # Use the selective loading method
            policy.trained_model.action_head.load_models_to_device(component_names)
        else:
            # Fallback to loading the entire action head
            if hasattr(policy.trained_model, "action_head") and hasattr(policy.trained_model.action_head, "image_encoder"):
                policy.trained_model.action_head.enable_vram_management()
            else:
                policy.trained_model.to(device=policy.device)
                
        # Apply bf16 if needed
        if policy.eval_bf16:
            policy.trained_model = policy.trained_model.to(dtype=torch.bfloat16)
            
        torch.cuda.empty_cache()  # Clear cache after loading
        
    def offload_action_head_components(self):
        """Offload action head components to CPU."""
        if self.action_head_policy is None:
            return
            
        policy = self.action_head_policy
        print(f"Offloading action head components to CPU...")
        
        # Check if the action head has the load_models_to_device method
        if (hasattr(policy.trained_model, "action_head") and 
            hasattr(policy.trained_model.action_head, "load_models_to_device")):
            # Use the selective offloading method - pass empty list to offload all
            policy.trained_model.action_head.load_models_to_device([])
        else:
            # Fallback to offloading the entire action head
            if hasattr(policy.trained_model, "action_head") and hasattr(policy.trained_model.action_head, "image_encoder"):
                if hasattr(policy.trained_model.action_head, 'disable_vram_management'):
                    policy.trained_model.action_head.disable_vram_management()
                policy.trained_model.to(device='cpu')
            else:
                policy.trained_model.to(device='cpu')
                
        torch.cuda.empty_cache()  # Clear cache after offloading
        self.active_components = None

    def load_model(self, name: str):
        """Load a model to GPU. (Deprecated - use activate_model instead)"""
        print(f"Warning: load_model is deprecated. Use activate_model instead.")
        self.activate_model(name)
        
    def offload_model(self, name: str):
        """Offload a model to CPU. (Deprecated - components are managed automatically)"""
        if name == "action_head":
            self.offload_action_head_components()
        # VLM is never offloaded in this new approach
    
    def get_status(self):
        """Get current status of model manager for debugging."""
        status = {
            "vlm_loaded": self.vlm_policy is not None,
            "action_head_available": self.action_head_policy is not None,
            "active_components": self.active_components
        }
        return status

    def enable_action_head_vram_management(self):
        """Enable vram management for the action head model."""
        if self.action_head_policy is None:
            return
            
        policy = self.action_head_policy
        
        # Check if the action head has vram management capability
        if (hasattr(policy.trained_model, "action_head") and 
            hasattr(policy.trained_model.action_head, "enable_vram_management")):
            print("Enabling vram management for action head...")
            policy.trained_model.action_head.enable_vram_management()
        else:
            print("Warning: Action head does not support vram management")


class BaseGrootSimPolicy(BaseTianshouPolicy):
    def __init__(self, embodiment_tag: EmbodimentTag, model_path: str, device: int | str):
        super().__init__()
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = device

    def forward(self, batch, state=None, **kwargs):
        raise NotImplementedError

    @property
    def video_delta_indices(self) -> np.ndarray:
        return np.array([0])

    @property
    def state_delta_indices(self) -> np.ndarray:
        return np.array([0])

    @property
    def raw_data_image_transform(self) -> Callable:
        return lambda x: x

    def on_env_init(self, env: gym.Env):
        pass

    def learn(self, batch: Batch, **kwargs) -> dict:
        """Dummy learn method for BasePolicy.learn as this is an inference-only wrapper.

        Args:
            batch: Input batch of experiences
            **kwargs: Additional arguments

        Returns:
            Empty dict as no learning occurs
        """
        return {}


class GrootSimPolicy(BaseGrootSimPolicy):
    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        model_config_overrides: list[str] | None = [],
        skip_assert_delta_indices: bool = False,
        skip_img_transform: bool = False,
        lazy_load: bool = False,
    ):
        """
        Initialize the GrootSimPolicy.

        Args:
            env_name (str): The name of the environment.
            model_path (str): Path to the model checkpoint.
            device (int | str): Device to run the model on.
            lazy_load (bool): If True, don't load model to GPU immediately.
        """
        super().__init__(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        model_dir = Path(model_path)
        self.rank = 0

        exp_cfg_dir = model_dir / "experiment_cfg"
        train_cfg_path = exp_cfg_dir / "conf.yaml"
        train_cfg = OmegaConf.load(train_cfg_path)
        self.train_cfg = train_cfg
        self.lazy_load = lazy_load

        # Store model loading parameters for lazy loading
        self.model_config_overrides = model_config_overrides
        self.model_dir = model_dir
        
        # 1. Load the model
        if (
            train_cfg.model._target_.endswith(".from_pretrained")
            or train_cfg.model._target_.endswith(".from_pretrained_for_tuning")
            or train_cfg.model._target_.endswith(".from_pretrained_with_wrapped_action_head")
        ):
            # Compatibility with the finetuned model
            model_target = train_cfg.model._target_.rsplit(".", 1)[0]
        else:
            model_target = train_cfg.model._target_
            
        self.model_target = model_target
        if model_config_overrides is not None and len(model_config_overrides) != 0:
            print(f"Applying model config overrides: {model_config_overrides}")
            # Only apply new logic if there are config overrides

            # Import the model class from the target module path
            module_path, class_name = model_target.rsplit(".", 1)
            if "lora" in class_name:
                module_path, class_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            model_config_class = model_class.config_class

            # Load the model config from the model directory
            model_config = json.load(open(model_dir / "config.json", "r"))
            model_config = OmegaConf.create(model_config)
            model_config.merge_with_dotlist(list(model_config_overrides))
            model_config = OmegaConf.to_container(
                model_config, resolve=True
            )  # need dict for later model instantiation
            model_config = model_config_class.from_dict(model_config)

            # Instantiate the model
            if hasattr(train_cfg, "save_lora_only") and train_cfg.save_lora_only is True:
                model = model_class.load_lora(model_path)
            else:
                model = model_class.from_pretrained(model_path, config=model_config)
        else:
            print(f"No model config overrides provided")
            # Otherwise, just call from_pretrained directly
            cls_module, cls_name = model_target.rsplit(".", 1)
            if 'lora' in cls_name:
                cls_module, cls_name = cls_module.rsplit(".", 1)
            if hasattr(train_cfg, "save_lora_only") and train_cfg.save_lora_only is True:
                print(f"Loading LoRA weights from pretrained")
                cls = getattr(importlib.import_module(cls_module), cls_name)
                model = cls.load_lora(model_path)
            else:
                print("="*100)
                print(f"Loading model from pretrained directly")
                cls = getattr(importlib.import_module(cls_module), cls_name)
                from_pretrained = getattr(cls, "from_pretrained")
                print(model_path, cls_module)
                model = from_pretrained(model_path)

        model.eval()
        model.requires_grad_(False)
        if model.action_head.train_architecture == "lora":
            print(f"Merging LoRA weights into main model weights")
            # Merge the LoRA weights into the main model weights, and delete the LoRA
            # weights to save memory. Note that the WanModel is in model.action_head.model.
            model.action_head.model = model.action_head.model.merge_and_unload()
        else:
            print(f"Skipping merging LoRA weights into main model weights")

        self.eval_bf16 = self.train_cfg.get("eval_bf16", False)
        if self.eval_bf16 and not lazy_load:
            model = model.to(dtype=torch.bfloat16)

        # Store model initially on CPU if lazy loading
        if lazy_load:
            model.to(device='cpu')
        else:
            model.to(device=device)

        # Post initialize, move RoPE freqs to cuda.
        model.post_initialize()

        # Parallelize the model across devices.
        try:
            model.parallelize(device_mesh=torch.device("cuda"))
        except Exception as e:
            print("Skipping parallelization")

        torch.cuda.empty_cache()

        self.trained_model = model

        # 2. Load the action, video, and state transforms
        # 2.1. Load the metadata for normalization stats
        # We have an assumption: one policy is only for rolling out one type of env, i.e., one embodiment_tag
        # metadata_versions = train_cfg.metadata_versions
        # metadata = get_metadata(self.embodiment_tag, metadata_versions[self.embodiment_tag.value])
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        if "gr1_unified_offline_rl" in metadatas and self.embodiment_tag.value == "gr1_unified":
            self.embodiment_tag = EmbodimentTag.GR1_UNIFIED_OFFLINE_RL
        metadata = DatasetMetadata.model_validate(metadatas[self.embodiment_tag.value])

        # 2.2. Get the eval transforms
        assert (
            self.embodiment_tag.value in train_cfg.transforms
        ), f"{self.embodiment_tag.value=}, {train_cfg.transforms.keys()=}"
        eval_transform_cfg = train_cfg.transforms[self.embodiment_tag.value]
        if skip_img_transform:
            for t in eval_transform_cfg.transforms:
                if t._target_ == "groot.vla.data.transform.VideoCrop":
                    scale = t.scale
                    for key in metadata.modalities.video.keys():
                        metadata.modalities.video[key].resolution = (
                            int(metadata.modalities.video[key].resolution[0] * scale),
                            int(metadata.modalities.video[key].resolution[1] * scale),
                        )
                elif t._target_ == "groot.vla.data.transform.VideoResize":
                    height, width = t.height, t.width
                    for key in metadata.modalities.video.keys():
                        metadata.modalities.video[key].resolution = (height, width)

            # assume it's always ComposedModalityTransform
            assert (
                eval_transform_cfg._target_ == "groot.vla.data.transform.ComposedModalityTransform"
            )
            skipped_transforms = [
                "groot.vla.data.transform.VideoCrop",
                "groot.vla.data.transform.VideoResize",
                "groot.vla.data.transform.VideoColorJitter",
            ]
            eval_transform_cfg.transforms = [
                t for t in eval_transform_cfg.transforms if t._target_ not in skipped_transforms
            ]

        eval_transform = instantiate(train_cfg.transforms[self.embodiment_tag.value])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)
        
        # Set per-horizon statistics for PerHorizonActionTransform if using relative_action_per_horizon
        relative_action_per_horizon = self.train_cfg.get('relative_action_per_horizon', False)
        print(f"DEBUG: relative_action_per_horizon = {relative_action_per_horizon}")
        if relative_action_per_horizon:
            # Extract per-horizon statistics from metadata
            # The metadata has format: {embodiment: {statistics: {action: {key: {stat: [[h0], [h1], ...]}}}}
            action_stats = metadata.statistics.action
            print(f"DEBUG: action_stats keys = {list(action_stats.keys())}")
            per_horizon_stats = {}
            for action_key in action_stats:
                stats_dict = action_stats[action_key].model_dump()
                print(f"DEBUG: action_key={action_key}, stats_dict keys={list(stats_dict.keys())}")
                # Check if stats are per-horizon (2D lists) by examining q01/q99
                if 'q01' in stats_dict:
                    q01_val = stats_dict['q01']
                    print(f"DEBUG: q01 type={type(q01_val)}, value sample={q01_val[:2] if hasattr(q01_val, '__getitem__') else q01_val}")
                    # Handle both list and numpy array
                    is_2d = False
                    if isinstance(q01_val, (list, np.ndarray)) and len(q01_val) > 0:
                        first_elem = q01_val[0]
                        print(f"DEBUG: q01[0] type={type(first_elem)}")
                        if isinstance(first_elem, (list, np.ndarray)):
                            is_2d = True
                    
                    if is_2d:
                        # This is per-horizon stats (2D array) - convert to list if numpy
                        if isinstance(q01_val, np.ndarray):
                            for k in stats_dict:
                                if isinstance(stats_dict[k], np.ndarray):
                                    stats_dict[k] = stats_dict[k].tolist()
                        per_horizon_stats[action_key] = stats_dict
                        print(f"DEBUG: Added {action_key} to per_horizon_stats")
            
            if per_horizon_stats:
                print(f"Setting per-horizon statistics for keys: {list(per_horizon_stats.keys())}")
                eval_transform.set_per_horizon_statistics(per_horizon_stats)
            else:
                print(f"WARNING: No per-horizon statistics found despite relative_action_per_horizon=True")
        
        eval_transform.eval()
        self.eval_transform = eval_transform

        # 3. Load horizons needed
        if self.embodiment_tag.value in train_cfg.modality_configs:
            self.modality_configs = instantiate(
                train_cfg.modality_configs[self.embodiment_tag.value]
            )
        else:
            self.modality_configs = instantiate(train_cfg.modality_configs)

        self._video_delta_indices = np.array(self.modality_configs.video.eval_delta_indices)
        # self._video_delta_indices = np.array([0])
        # self.assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)

        # We might not use `state`, which will be a vision-only policy
        if "state" in self.modality_configs:
            self._state_delta_indices = np.array(self.modality_configs.state.eval_delta_indices)
            if not skip_assert_delta_indices:
                self.assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None
        self._raw_data_image_transform = None

    def offload_to_cpu(self):
        """Offload the model to CPU to free GPU memory."""
        if hasattr(self.trained_model, "action_head") and hasattr(self.trained_model.action_head, "image_encoder"):
            # For models with vram management, disable it and move to CPU
            if hasattr(self.trained_model.action_head, 'disable_vram_management'):
                self.trained_model.action_head.disable_vram_management()
        
        self.trained_model.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"Model offloaded to CPU")

    def load_to_gpu(self):
        """Load the model to GPU for inference."""
        print(f"Loading model to GPU...")
        
        # Move model to GPU
        if hasattr(self.trained_model, "action_head") and hasattr(self.trained_model.action_head, "image_encoder"):
            self.trained_model.action_head.enable_vram_management()
        else:
            self.trained_model.to(device=self.device)
            
        # Apply bf16 if needed
        if self.eval_bf16:
            self.trained_model = self.trained_model.to(dtype=torch.bfloat16)
            
        torch.cuda.empty_cache()

    def ensure_model_on_gpu(self):
        """Ensure the model is loaded on GPU before inference."""
        # Check if model is on CPU
        model_device = next(self.trained_model.parameters()).device
        if model_device.type == 'cpu':
            self.load_to_gpu()

    def assert_delta_indices(self, delta_indices: np.ndarray):
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

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

    def forward(self, batch, state=None, **kwargs):
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.get_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch
    
    def joint_forward(self, batch, video=None, state=None, **kwargs):
        # 0. Ensure model is on GPU
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.joint_video_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def lazy_joint_forward_causal(self, batch, video=None, latent_video=None, state=None, video_only=False, **kwargs):
        
        transform_start_time = time.perf_counter()

        # Save original observation before any modification (for relative action conversion)
        original_obs_for_relative = {k: v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v 
                                     for k, v in batch.obs.items()}

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)
            # Also unsqueeze the saved original obs
            original_obs_for_relative = unsqueeze_dict_values(original_obs_for_relative)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        transform_time = time.perf_counter() - transform_start_time

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        model_start_time = time.perf_counter()

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_causal(normalized_input, latent_video=latent_video)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        model_time = time.perf_counter() - model_start_time

        untransform_start_time = time.perf_counter()

        # 4. Unnormalize actions (pass obs for relative action conversion)
        if not video_only:
            batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs_for_relative)
        else:
            batch = Batch(normalized_action=normalized_action)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)

        untransform_time = time.perf_counter() - untransform_start_time
        total_time = transform_time + model_time + untransform_time

        if self.rank == 0:
            print(f"Inference Time: Total {total_time:.3f} seconds, "
                  f"Transform: {transform_time:.3f} seconds, "
                  f"Model: {model_time:.3f} seconds, "
                  f"Untransform: {untransform_time:.3f} seconds")

        return batch, video_pred
    
    def lazy_joint_forward_causal_gt_cond(self, batch, video=None, latent_video=None, state=None, **kwargs):
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_causal_gt_cond(normalized_input, latent_video=latent_video)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def lazy_joint_forward(self, batch, video=None, state=None, **kwargs):
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred
    
    def lazy_joint_forward_efficient(self, batch, video=None, state=None, prompt_embs=None, prompt_emb_nega=None, **kwargs):
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_efficient(normalized_input, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred
    
    def language_forward(self, batch, state=None, **kwargs):
        
        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.get_language(normalized_input)
        # normalized_action = model_pred["action_pred"].float()
        output = model_pred["output"]
        # 4. Unnormalize actions
        # batch = self.unapply(Batch(normalized_action=normalized_action))

        # # 5. Remove batch dimension if we added it
        # if not is_batched:
        #     batch.act = squeeze_dict_values(batch.act)
        return output
        # return batch


    def gt_video_action_pred(self, batch, video=None, state=None, **kwargs):

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.gt_video_action_pred(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    @property
    def raw_data_image_transform(self) -> Callable:
        """
        Get the raw data image transform for the policy
        """
        return lambda x: x

    @property
    def state_delta_indices(self) -> np.ndarray:
        return self._state_delta_indices

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self._video_delta_indices


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


def tree_get_leading_dim(data: dict[str, Any]) -> int:
    """
    Get the batch size (leading dimension) of all tensors in the structure.

    Args:
        data: Nested structure containing tensors
        strict: If True, raises error if batch sizes don't match.
                If False, returns None if inconsistent.

    Returns:
        The common batch size, or None if inconsistent (when strict=False)

    Raises:
        ValueError: If batch sizes are inconsistent (when strict=True)
        ValueError: If no tensors found
    """
    batch_sizes = []
    tensor_paths = []

    def collect_batch_sizes(path, x):
        if hasattr(x, "shape") and len(x.shape) > 0:
            batch_sizes.append(x.shape[0])
            tensor_paths.append(".".join(map(str, path)))

    # Collect all batch sizes with their paths
    tree.map_structure_with_path(collect_batch_sizes, data)

    if not batch_sizes:
        raise ValueError("No tensors found in the data structure")

    # Check if all batch sizes are the same
    first_batch_size = batch_sizes[0]

    if not all(bs == first_batch_size for bs in batch_sizes):
        inconsistent_info = [f"{path}: {bs}" for path, bs in zip(tensor_paths, batch_sizes)]
        error_msg = "Inconsistent batch sizes found:\n" + "\n".join(inconsistent_info)

        raise ValueError(error_msg)

    return first_batch_size


class GrootSimRLPolicy(BaseGrootSimPolicy):
    """
    A class for Offline RL policies,
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        q_fn_path: str | None = None,
        n_action_samples: int = 1,
        n_denoising_steps: int = 4,
        gaussian_std: float = 0.0,
        inference_batch_size: int = 1,
        model_config_overrides: list[str] | None = [],
    ):
        """
        Implicit Policy Extraction (IPE) GrootSimPolicy.
        """
        super().__init__(embodiment_tag=embodiment_tag, model_path=model_path, device=device)

        self.base_policy = GrootSimPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(model_path),
            device=device,
        )
        self.base_policy.trained_model.action_head.num_inference_timesteps = n_denoising_steps

        if q_fn_path is None:
            if n_action_samples > 1:
                raise ValueError("If n_action_samples > 1, q_fn_path should be provided.")
            self.qfn = None
        else:
            assert n_action_samples > 1, "If q_fn_path is provided, n_action_samples should be > 1."
            self.qfn = GrootSimPolicy(
                embodiment_tag=EmbodimentTag.GR1_UNIFIED_OFFLINE_RL,
                model_path=q_fn_path,
                device=device,
                skip_assert_delta_indices=True,
            )

        self.n_action_samples = n_action_samples
        self.gaussian_std = gaussian_std
        self.inference_batch_size = inference_batch_size

    def batch_obs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """
        Check if input is batched and add batch dimension if needed
        """
        is_batched = self.base_policy._check_state_is_batched(obs)
        if not is_batched:
            obs = unsqueeze_dict_values(obs)

        return obs, is_batched

    def add_action_noise(self, action: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0.0:
            return action
        noise = torch.randn_like(action) * std
        noisy_action = action + noise
        return noisy_action

    def sample_actions_from_base_policy(
        self,
        batch: Batch,
        n_action_samples: int,  # number of actions to sample PER item in the batch
        inference_batch_size: int = 1,
    ) -> torch.Tensor:
        # a) Apply transforms/normalization
        batch = self.base_policy.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.base_policy.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            model_pred = self.base_policy.trained_model.get_action(
                normalized_input,
                num_action_samples=n_action_samples,
                inference_batch_size=inference_batch_size,
                validate=False,
            )
            normalized_action = model_pred["action_pred"]
            normalized_action = self.add_action_noise(normalized_action, self.gaussian_std)
        return normalized_action  # [num_envs, n_action_samples, horizon, action_dim]

    def get_action_values(
        self, batch: Batch, actions: torch.Tensor, inference_batch_size: int = 1
    ) -> torch.Tensor:
        assert self.qfn is not None, "Q-function must be provided to evaluate actions."

        # a) Apply transforms/normalization
        batch = self.qfn.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.qfn.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            # [num_envs, ...]
            backbone_inputs, action_inputs = self.qfn.trained_model.prepare_input(normalized_input)
            backbone_outputs = self.qfn.trained_model.backbone(backbone_inputs)
            q_values = self.qfn.trained_model.action_head.get_q_pred(
                backbone_outputs,
                action_inputs,
                actions,
                inference_batch_size=inference_batch_size,
            )

        return q_values

    def forward(self, batch, state=None, **kwargs):
        orig_obs = batch.obs.copy()
        orig_obs, is_batched = self.batch_obs(orig_obs)

        # Get number of environments
        num_envs = 1 if not is_batched else tree_get_leading_dim(orig_obs)

        # 1/ Sample normalized actions from the base policy
        actions = self.sample_actions_from_base_policy(
            Batch(obs=orig_obs),
            n_action_samples=self.n_action_samples,
            inference_batch_size=self.inference_batch_size,
        )  # [num_envs, n_action_samples, horizon, action_dim]

        # 2/ Evaluate the sampled actions with the Q-function
        if self.qfn is not None:
            q_values = self.get_action_values(
                Batch(obs=orig_obs),
                actions=actions,
                inference_batch_size=self.inference_batch_size,
            )

            best_indices = torch.argmax(q_values, dim=1)  # [num_envs]
            normalized_action = actions[
                torch.arange(num_envs), best_indices
            ].float()  # [num_envs, horizon, action_dim]
        else:
            normalized_action = actions.float()

        batch = self.base_policy.unapply(Batch(normalized_action=normalized_action), obs=orig_obs)

        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self.base_policy.video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray:
        return self.base_policy.state_delta_indices

    @property
    def raw_data_image_transform(self) -> Callable:
        return lambda x: x
