"""
OpenPI PI0 Configuration compatible with transformers PretrainedConfig.

This configuration class follows the OpenPI Pi0Config structure exactly 
while being compatible with transformers training infrastructure.
"""

from transformers import PretrainedConfig
from typing import Optional


class PI0Config(PretrainedConfig):
    """
    OpenPI PI0 model configuration.
    
    This configuration class follows the exact OpenPI Pi0Config structure
    and is compatible with transformers PretrainedConfig.
    
    Note: Supports both `dtype` (OpenPI JAX convention) and `precision` 
    (OpenPI PyTorch checkpoint convention) for compatibility.
    """
    
    model_type = "pi0"
    
    def __init__(
        self,
        # Core OpenPI parameters - support both dtype and precision for compatibility
        dtype: str = None,
        precision: str = None,  # Alias for dtype (used in some checkpoints)
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = None,
        
        # Model architecture variants
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m", 
        pi05: bool = False,
        
        # Training/freezing parameters (matching LeRobot Pi0Config)
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        train_state_proj: bool = True,
        
        **kwargs,
    ):
        """
        Initialize PI0Config.
        
        Args:
            dtype: Model precision dtype ("bfloat16", "float32"). 
                   If not provided, falls back to `precision` or default "bfloat16".
            precision: Alias for dtype (for checkpoint compatibility).
            action_dim: Action space dimension (OpenPI default: 32)
            action_horizon: Number of action steps to predict (OpenPI default: 50) 
            max_token_len: Maximum token sequence length. Defaults to 200 for PI0.5, 48 for PI0.
            paligemma_variant: PaliGemma model variant ("gemma_2b", "gemma_300m")
            action_expert_variant: Action expert model variant ("gemma_300m", "gemma_2b")
            pi05: Whether to use PI0.5 (adaptive mixture model)
            freeze_vision_encoder: Whether to freeze vision encoder parameters
            train_expert_only: Whether to train only action expert parameters
            train_state_proj: Whether to train state projection layer
        """
        # Call parent constructor FIRST (it may set some attributes)
        super().__init__(**kwargs)
        
        # Handle dtype/precision compatibility: dtype takes priority, then precision, then default
        # Note: We set self.dtype AFTER super().__init__() because PretrainedConfig 
        # has its own dtype handling that would overwrite our value
        if dtype is not None:
            self.dtype = dtype
        elif precision is not None:
            self.dtype = precision
        else:
            self.dtype = "bfloat16"
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.pi05 = pi05
        
        # Set max_token_len based on pi05 if not provided (matching OpenPI behavior)
        if max_token_len is None:
            self.max_token_len = 200 if pi05 else 48
        else:
            self.max_token_len = max_token_len
        
        # Store training/freezing parameters
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_state_proj = train_state_proj
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiate a PI0Config from a Python dictionary of parameters.
        """
        return cls(**config_dict, **kwargs)
    
    def to_dict(self):
        """
        Serialize this instance to a Python dictionary.
        """
        output = super().to_dict()
        
        # Add OpenPI-specific parameters
        # Include both dtype and precision for compatibility with different checkpoint formats
        output.update({
            "dtype": self.dtype,
            "precision": self.dtype,  # Alias for checkpoint compatibility
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "pi05": self.pi05,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "train_state_proj": self.train_state_proj,
        })
        
        return output
