"""
PI0.5 Configuration for extended training modes.

Extends PI0Config to support three forwarding modes:
- VLA: Standard flow matching action prediction
- VLM: Pure language model with cross-entropy loss (no action expert)
- VLM+VLA: Reasoning text generation + action prediction (combined losses)
"""

from typing import Literal, Optional
from transformers import PretrainedConfig


ForwardMode = Literal["vla", "vlm", "vlm_vla"]


class PI05Config(PretrainedConfig):
    """
    Configuration for PI0.5 model with extended training modes.
    
    Supports three forwarding modes:
    - "vla": Standard VLA mode with flow matching (action expert only)
    - "vlm": Pure VLM mode with cross-entropy loss (no action expert)
    - "vlm_vla": Combined mode with reasoning + action prediction
    
    Inherits core structure from PI0Config but adds:
    - forward_mode: Which mode to use during training/inference
    - max_language_len: Maximum length for generated reasoning text
    - language_temperature: Temperature for reasoning text sampling
    """
    
    model_type = "pi05"
    
    def __init__(
        self,
        # Core parameters (matching PI0Config)
        dtype: Optional[str] = None,
        precision: Optional[str] = None,
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = 200,
        
        # Model architecture
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        
        # Training/freezing
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        
        # PI0.5 specific parameters
        forward_mode: ForwardMode = "vla",
        max_language_len: int = 50,
        language_temperature: float = 0.0,
        
        # Loss weights for VLM+VLA mode
        language_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        
        # EOS token for language generation termination
        # Note: Model will auto-detect from PaliGemma config; this is a fallback
        eos_token_id: int = 1,  # PaliGemma default EOS token
        
        # Knowledge insulation for FAST dual-objective training (Pi0 paper Section 5.2)
        # When True: Splits attention into backbone self-attention (full gradients) and
        # cross-attention (stop gradient on backbone k/v). This ensures:
        # - CE loss: FULL gradients to all VLM parameters (Q, K, V, O, MLP)
        # - Flow loss: gradients to action expert only, blocked from VLM backbone
        stop_gradient_to_vlm: bool = False,
        
        # State input mode: if True, robot state is discretized into language tokens
        # and appended to the prompt; if False, state is only used as continuous input
        discrete_state_input: bool = False,
        
        # Separate CoT training mode (matching vla-scratch architecture):
        # When True, CoT/reasoning tokens are EXCLUDED from the KV cache for action expert
        # (kv_cache_mask=False), while still being supervised by CE loss (loss_mask=True).
        # This ensures:
        # - VLM learns reasoning via CE loss on CoT tokens
        # - Action expert only attends to core observation (images + prompt), not reasoning
        # Effect: Reasoning learning is isolated from action prediction.
        exclude_cot_from_kv_cache: bool = False,
        
        **kwargs,
    ):
        # Call super first with kwargs only
        super().__init__(**kwargs)
        
        # Handle dtype/precision compatibility
        if dtype is not None:
            self.dtype = dtype
        elif precision is not None:
            self.dtype = precision
        else:
            self.dtype = "bfloat16"
        
        # Core parameters
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        
        # Training/freezing
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        
        # PI0.5 specific
        self.pi05 = True  # Always True for PI0.5
        self.forward_mode = forward_mode
        self.max_language_len = max_language_len
        self.language_temperature = language_temperature
        self.language_loss_weight = language_loss_weight
        self.action_loss_weight = action_loss_weight
        self.eos_token_id = eos_token_id
        self.stop_gradient_to_vlm = stop_gradient_to_vlm
        self.discrete_state_input = discrete_state_input
        self.exclude_cot_from_kv_cache = exclude_cot_from_kv_cache
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict, **kwargs)
    
    def to_dict(self):
        output = super().to_dict()
        output.update({
            "dtype": self.dtype,
            "precision": self.dtype,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "pi05": self.pi05,
            "forward_mode": self.forward_mode,
            "max_language_len": self.max_language_len,
            "language_temperature": self.language_temperature,
            "language_loss_weight": self.language_loss_weight,
            "action_loss_weight": self.action_loss_weight,
            "eos_token_id": self.eos_token_id,
            "stop_gradient_to_vlm": self.stop_gradient_to_vlm,
            "discrete_state_input": self.discrete_state_input,
            "exclude_cot_from_kv_cache": self.exclude_cot_from_kv_cache,
        })
        return output

