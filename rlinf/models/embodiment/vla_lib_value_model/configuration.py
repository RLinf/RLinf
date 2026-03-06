"""
PI0.5 Configuration for value model training.

Provides:
- PI05Config: Base configuration for PI0.5 model architecture
- PI05CriticConfig: Extended configuration for value critic models
"""

from typing import Literal, Optional

from transformers import PretrainedConfig

ForwardMode = Literal["vla", "vlm", "vlm_vla"]


class PI05Config(PretrainedConfig):
    """Configuration for PI0.5 model."""

    model_type = "pi05"

    def __init__(
        self,
        dtype: Optional[str] = None,
        precision: Optional[str] = None,
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = 200,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        train_expert_only: bool = False,
        forward_mode: ForwardMode = "vla",
        max_language_len: int = 50,
        language_temperature: float = 0.0,
        language_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        eos_token_id: int = 1,
        stop_gradient_to_vlm: bool = False,
        discrete_state_input: bool = False,
        exclude_cot_from_kv_cache: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dtype is not None:
            self.dtype = dtype
        elif precision is not None:
            self.dtype = precision
        else:
            self.dtype = "bfloat16"

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.train_expert_only = train_expert_only
        self.pi05 = True
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
        output.update(
            {
                "dtype": self.dtype,
                "precision": self.dtype,
                "action_dim": self.action_dim,
                "action_horizon": self.action_horizon,
                "max_token_len": self.max_token_len,
                "paligemma_variant": self.paligemma_variant,
                "action_expert_variant": self.action_expert_variant,
                "freeze_vision_encoder": self.freeze_vision_encoder,
                "freeze_vlm": getattr(self, "freeze_vlm", False),
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
            }
        )
        return output


class PI05CriticConfig(PI05Config):
    """Configuration for PI05 Critic models (V function)."""

    def __init__(
        self,
        critic_expert_variant: str = "gemma_100m",
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        **kwargs,
    ):
        # Accept and ignore legacy parameters for checkpoint compatibility
        kwargs.pop("critic_forward_mode", None)
        kwargs.pop("expert_loss_type", None)
        kwargs.pop("expert_loss_weight", None)
        kwargs.pop("vlm_loss_weight", None)
        super().__init__(**kwargs)
        self.critic_expert_variant = critic_expert_variant
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
