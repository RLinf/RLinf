import dataclasses
import glob
import math
import os
import json
import re

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.modeling_outputs import CausalLMOutput, ModelOutput
from transformers import PreTrainedModel, GenerationMixin
from transformers.utils import logging as hf_logging
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import load_file
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .configs import get_config
from .paligemma_with_expert import PaliGemmaWithExpertModel, _requires_uniform_dtype
from .configuration_pi0 import PI0Config
from rlinf.utils.dist_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PI0CausalLMOutputWithPast(ModelOutput):
    """
    Base class for PI0 causal language model (or autoregressive) outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    actions: Optional[torch.FloatTensor] = None
    losses: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PI0Config
    base_model_prefix = "model"
    main_input_name = "input_ids"  # Standard for transformers FLOP calculation
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True
    # Override to ensure all weights are saved (including nested lm_head)
    # PI0's tie_weights() is a no-op, so we don't actually tie any weights
    _tied_weights_keys = []

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """
        Enable or disable gradient checkpointing for the PI0 model.
        This propagates gradient checkpointing to all submodules.
        """
        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = checkpoint
            
        # Set gradient checkpointing on the main PI0FlowMatching module
        if hasattr(self, 'model'):
            if enable:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Override from_pretrained to handle problematic safetensors file.
        
        This temporarily moves the problematic model.safetensors file that has 
        bad metadata, allowing HuggingFace to use the sharded loading instead.
        
        Intelligently handles device_map and meta tensors based on training context.
        """
        logger.info(f"PI0 from_pretrained: Loading from {pretrained_model_name_or_path}")
        
        # Handle problematic safetensors files by attempting manual loading first
        single_file_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        index_file_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
        sharded_pattern = os.path.join(pretrained_model_name_or_path, "model-*.safetensors")
        has_single_file = os.path.exists(single_file_path)
        has_index_file = os.path.exists(index_file_path)
        sharded_parts = glob.glob(sharded_pattern)
        has_sharded_files = len(sharded_parts) > 0

        if has_single_file:
            try:
                with safe_open(single_file_path, framework='pt') as f:
                    metadata = f.metadata()
                invalid_metadata = not metadata or 'format' not in metadata or metadata.get('format') is None
            except Exception as exc:  # Broad except to ensure we still fall back to manual load
                logger.warning(
                    "Failed to read single model.safetensors metadata (%s) - attempting manual loading",
                    exc,
                )
                return cls._load_from_safetensors(pretrained_model_name_or_path, **kwargs)

            if invalid_metadata:
                logger.warning("Single model.safetensors has invalid metadata - attempting manual loading")
                return cls._load_from_safetensors(pretrained_model_name_or_path, **kwargs)
        elif has_index_file or has_sharded_files:
            logger.info("Detected HF-sharded checkpoint - will rely on HuggingFace loader")
        else:
            logger.warning(
                "No model.safetensors file detected; attempting standard HuggingFace loading."
            )
        
        # Smart handling of device_map and memory optimizations
        safe_kwargs = kwargs.copy()
        original_device_map = kwargs.get('device_map')
        
        # Detect distributed training context
        is_distributed = (
            os.environ.get('WORLD_SIZE') is not None or 
            os.environ.get('LOCAL_RANK') is not None or
            os.environ.get('RANK') is not None
        )
        
        # For distributed training (FSDP/DDP), let Accelerate handle device placement
        if is_distributed:
            logger.info("Detected distributed training - letting Accelerate handle device placement")
            safe_kwargs.update({
                'device_map': None,
                'low_cpu_mem_usage': False,  # Disable to avoid meta tensors in distributed setting
                'torch_dtype': kwargs.get('torch_dtype', torch.float32),
            })
        else:
            # For single GPU or inference, preserve original device_map behavior
            if original_device_map is not None:
                logger.info(f"Single GPU/inference mode - using device_map: {original_device_map}")
                safe_kwargs.update({
                    'torch_dtype': kwargs.get('torch_dtype', torch.float32),
                })
            else:
                logger.info("No device_map specified - using default CPU loading")
                safe_kwargs.update({
                    'device_map': None,
                    'low_cpu_mem_usage': False,
                    'torch_dtype': kwargs.get('torch_dtype', torch.float32),
                })

        # Always provide an explicit PI0Config so HF does not attempt to resolve other config classes
        if 'config' not in safe_kwargs or safe_kwargs['config'] is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as cfg_file:
                    config_dict = json.load(cfg_file)
                safe_kwargs['config'] = cls.config_class.from_dict(config_dict)
            else:
                loaded_config, _ = cls.config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    return_unused_kwargs=True,
                    trust_remote_code=kwargs.get('trust_remote_code', True),
                )
                safe_kwargs['config'] = loaded_config

        model = super().from_pretrained(pretrained_model_name_or_path, **safe_kwargs)
        logger.info("✓ Successfully loaded PI0 model using HuggingFace infrastructure")
        
        # Apply mixed-precision handling when not using parameter sharding
        # FSDP and DeepSpeed Zero-3 require uniform dtype; DDP/Zero-1/2 can use mixed
        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        if torch_dtype == torch.bfloat16 and not _requires_uniform_dtype():
            cls._apply_mixed_precision(model)
        elif _requires_uniform_dtype():
            logger.info("Parameter sharding detected (FSDP/Zero-3): using uniform dtype")
        
        return model

    @classmethod
    def _apply_mixed_precision(cls, model):
        """Apply mixed precision: keep action projection layers in float32 for flow matching.
        
        This is needed because noise and time in flow matching are float32, and matmul
        requires operands to have the same dtype. The backbone (vision/language) can stay
        in bfloat16 for memory efficiency.
        """
        # List of parameter name patterns that should stay in float32
        action_proj_patterns = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in action_proj_patterns):
                param.data = param.data.to(dtype=torch.float32)
                logger.debug(f"Kept {name} in float32 for flow matching compatibility")
        
        logger.info("Applied mixed precision: action projection layers kept in float32")

    @classmethod
    def _load_from_safetensors(cls, pretrained_model_name_or_path, **kwargs):
        """Manual loading from a single safetensors file with potentially corrupted metadata."""
        logger.info("Attempting manual loading from single safetensors file")
        
        # Load configuration - prefer kwargs config over config.json
        config = kwargs.get('config')
        if config is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = cls.config_class.from_dict(config_dict)
            else:
                raise ValueError(f"No config.json found in {pretrained_model_name_or_path} and no config provided in kwargs")
        
        # Create model from config
        logger.info("Creating model from config for manual loading")
        model = cls(config)
        
        # Load state dict manually from safetensors file
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        logger.info(f"Manually loading state dict from {safetensors_path}")
        
        try:
            # Load the safetensors file ignoring metadata issues
            state_dict = load_file(safetensors_path, device="cpu")
            logger.info(f"Successfully loaded {len(state_dict)} tensors from safetensors file")
            
            # Load into model
            # Check if state dict keys start with "model." to determine loading strategy
            sample_key = next(iter(state_dict.keys())) if state_dict else ""
            if sample_key.startswith("model."):
                logger.info("State dict keys start with 'model.' - loading into model directly")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            else:
                logger.info("State dict keys don't start with 'model.' - loading into model.model")
                missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading state dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys}")
            
            logger.info("✓ Successfully loaded model from single safetensors file")
            
            # Handle dtype - FSDP/Zero-3 require uniform dtype; DDP/Zero-1/2 can use mixed
            torch_dtype = kwargs.get('torch_dtype', torch.float32)
            
            if _requires_uniform_dtype():
                if torch_dtype == torch.bfloat16:
                    model.to(dtype=torch.bfloat16)
                    logger.info("Parameter sharding detected: uniform bfloat16 dtype")
                else:
                    model.to(dtype=torch.float32)
                    logger.info("Parameter sharding detected: uniform float32 dtype")
            else:
                # Non-sharding methods: use mixed precision (bf16 backbone + fp32 action layers)
                if torch_dtype == torch.bfloat16:
                    model.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
                elif torch_dtype == torch.float32:
                    model.model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to manually load from safetensors file: {e}")
            raise ValueError(f"Could not load model from {pretrained_model_name_or_path}: {e}")

class PI0ForConditionalGeneration(PI0PreTrainedModel, GenerationMixin):
    """
    PI0 model for conditional generation tasks.
    This model combines PaliGemma with Expert layers to perform Vision-Language-Action tasks.
    """

    def __init__(self, config: PI0Config):
        super().__init__(config)
        self.config = config
        
        # Core PI0 model
        self.model = PI0FlowMatching(config)
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        observation=None,
        actions=None,
        noise=None,
        time=None,
        # For transformers compatibility
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        """
        Forward pass for training and inference.
        
        Args:
            observation: Robot observation (images, language, state)
            actions: Target actions (for training)
            noise: Noise for flow matching (optional)
            time: Time steps for flow matching (optional)
            input_ids: For transformers compatibility (unused)
            attention_mask: For transformers compatibility (unused)
            return_dict: Whether to return a ModelOutput or tuple
            use_cache: Whether to use cache for generation
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            PI0CausalLMOutputWithPast containing losses or action predictions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if observation is None:
            raise ValueError("PI0 model requires 'observation' input")

        # Training mode - compute loss
        result = self.model.forward(
            observation=observation,
            actions=actions,
            noise=noise,
            time=time,
            **kwargs
        )
        
        if return_dict:
            return PI0CausalLMOutputWithPast(
                loss=result.loss.mean(),
                losses=result.loss, # non-reduced loss
                actions=None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            return (result.loss.mean(),)

    def generate(
        self,
        observation,
        noise=None,
        num_steps=10,
        **kwargs,
    ):
        """
        Generate actions given inputs.
        Compatible with HF generation interface.
        """
        device = next(self.parameters()).device
        return self.model.sample_actions(
            device=device,
            observation=observation,
            noise=noise,
            num_steps=num_steps,
        )

    def warmup(self, device: torch.device | str | None = None, tokenizer=None) -> None:
        """Run warmup inference to trigger torch.compile JIT compilation.
        
        Call this after model is on device and before serving to avoid
        compilation delay on first query.
        """
        if device is None:
            device = next(self.parameters()).device
        self.model.warmup(device=device, tokenizer=tokenizer)

    def get_input_embeddings(self):
        """Get input embeddings from the language model."""
        return self.model.paligemma_with_expert.paligemma.language_model.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings for the language model."""
        self.model.paligemma_with_expert.paligemma.language_model.embed_tokens = value

    def get_output_embeddings(self):
        """Get output embeddings (lm_head) from the language model."""
        return self.model.paligemma_with_expert.paligemma.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (lm_head) for the language model."""
        self.model.paligemma_with_expert.paligemma.lm_head = new_embeddings

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        pass

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resize token embeddings to accommodate new vocabulary size.
        
        Follows transformers library patterns for proper initialization of new embeddings.
        When mean_resizing=True, new embeddings are initialized from a multivariate normal
        distribution with old embeddings' mean and covariance (or just mean if covariance
        is not positive definite). See: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        
        Args:
            new_num_tokens: New vocabulary size
            pad_to_multiple_of: If set, pad vocab to multiple of this value
            mean_resizing: If True, init new embeddings from old embeddings' distribution
            
        Returns:
            The resized input embeddings module
        """
        if pad_to_multiple_of is not None:
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        
        old_embeddings = self.get_input_embeddings()
        old_lm_head = self.get_output_embeddings()
        old_num_tokens = old_embeddings.num_embeddings
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # Create new embeddings preserving device and dtype
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embeddings.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]
        new_lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[:num_tokens_to_copy]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        
        # Initialize new embeddings
        if new_num_tokens > old_num_tokens:
            added_num_tokens = new_num_tokens - old_num_tokens
            if mean_resizing:
                self._init_new_embeddings_with_mean(
                    old_embeddings.weight.data, new_embeddings.weight.data,
                    old_num_tokens, added_num_tokens
                )
                self._init_new_embeddings_with_mean(
                    old_lm_head.weight.data, new_lm_head.weight.data,
                    old_num_tokens, added_num_tokens
                )
                if old_lm_head.bias is not None:
                    bias_mean = old_lm_head.bias.data.mean()
                    new_lm_head.bias.data[old_num_tokens:] = bias_mean
            else:
                nn.init.normal_(new_embeddings.weight.data[old_num_tokens:], mean=0.0, std=0.02)
                nn.init.normal_(new_lm_head.weight.data[old_num_tokens:], mean=0.0, std=0.02)
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data[old_num_tokens:].zero_()
        
        self.set_input_embeddings(new_embeddings)
        self.set_output_embeddings(new_lm_head)
        
        # Update config
        if hasattr(self.config, 'vocab_size'):
            self.config.vocab_size = new_num_tokens
        
        return new_embeddings

    def _init_new_embeddings_with_mean(
        self,
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
        old_num_tokens: int,
        added_num_tokens: int,
    ) -> None:
        """
        Initialize new embedding weights using mean of old embeddings.
        
        Attempts to use multivariate normal with old embeddings' covariance.
        Falls back to just the mean if covariance is not positive definite.
        """
        old_weights_f32 = old_weights[:old_num_tokens].to(torch.float32)
        mean_embedding = old_weights_f32.mean(dim=0)
        
        # Try to compute covariance and sample from multivariate normal
        try:
            centered = old_weights_f32 - mean_embedding
            covariance = (centered.T @ centered) / old_num_tokens
            epsilon = 1e-9
            dist = torch.distributions.MultivariateNormal(
                mean_embedding, covariance_matrix=epsilon * covariance
            )
            new_weights[old_num_tokens:] = dist.sample((added_num_tokens,)).to(new_weights.dtype)
        except (ValueError, RuntimeError):
            # Covariance not positive definite, use mean only
            new_weights[old_num_tokens:] = mean_embedding.to(new_weights.dtype)


@dataclasses.dataclass
class PI0FlowMatchingOutput(CausalLMOutput):
    loss: torch.Tensor

class PI0FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        logger.info(f"PI05: {self.pi05}")

        paligemma_config = get_config(config.paligemma_variant)
        action_expert_config = get_config(config.action_expert_variant)
        
        logger.info(f"Creating PaliGemmaWithExpertModel with precision={config.dtype}")
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', False),
            train_expert_only=getattr(config, 'train_expert_only', False),
        )

        # Projection layers - no explicit dtype, inherits from default (matching OpenPI)
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        
        # Allow disabling torch.compile via environment variable for faster debugging/evaluation
        self._compile_enabled = os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1"
        if self._compile_enabled:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        else:
            logger.info("torch.compile disabled via TORCH_COMPILE_DISABLE=1")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        
        # Apply parameter freezing configuration
        self.set_requires_grad()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logger.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logger.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Use model weight dtype to avoid attention bias dtype mismatch errors
        # This matches the pattern used elsewhere in the codebase (line 791)
        dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        return torch.where(att_2d_masks_4d, torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device), 
                          torch.tensor(-2.3819763e38, dtype=dtype, device=att_2d_masks.device))

    def _preprocess_observation(self, observation):
        """Helper method to extract observation components (preprocessing done externally)."""
        # Handle both dict and object-style observations
        images = observation['images']
        image_masks = observation.get('image_masks', {})
        tokenized_prompt = observation['tokenized_prompt']
        tokenized_prompt_mask = observation['tokenized_prompt_mask']
        state = observation['state']
        
        # CRITICAL: Use sorted keys for consistent ordering between train and eval.
        # dict.values() order depends on insertion order, which may differ between
        # training datasets and inference environments.
        sorted_keys = sorted(images.keys())
        img_list = [images[k] for k in sorted_keys]
        img_mask_list = [image_masks.get(k, torch.tensor(True, dtype=torch.bool)) for k in sorted_keys]
        
        return (
            img_list,
            img_mask_list,
            tokenized_prompt,
            tokenized_prompt_mask,
            state,
        )

    def sample_noise(self, shape, device):
        # Keep as float32 - action projection layers are float32 (matching OpenPI)
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        # Keep as float32 - action projection layers are float32 (matching OpenPI)
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        # NOTE: Use lang_masks.shape[1] (same source as pad_masks) for consistency
        # This ensures att_masks and pad_masks have matching sequence lengths
        num_lang_embs = lang_masks.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # NOTE: Use action_time_dim (actual data shape) instead of self.config.action_horizon
        # to ensure consistency with pad_masks when data action_horizon differs from model config
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, **kwargs) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if noise.dtype != actions.dtype:
            noise = noise.to(actions.dtype)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        if time.dtype != actions.dtype:
            time = time.to(actions.dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")

        return PI0FlowMatchingOutput(
            loss=loss,
            logits=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # Handle both dict and object-style observation access
        if hasattr(observation, 'state'):
            bsize = observation.state.shape[0]
        elif isinstance(observation, dict) and 'state' in observation:
            bsize = observation['state'].shape[0]
        else:
            raise ValueError("observation must have 'state' attribute or key")
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        # Keep as float32 - action projection layers are float32 (matching OpenPI)
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)
    
    def set_requires_grad(self):
        """Set requires_grad for parameters based on configuration."""
        # Handle state projection freezing (only for PI0, not PI0.5)
        if not self.pi05 and hasattr(self, 'state_proj'):
            train_state_proj = getattr(self.config, 'train_state_proj', True)
            for param in self.state_proj.parameters():
                param.requires_grad = train_state_proj

    @torch.no_grad()
    def warmup(self, device: torch.device | str, tokenizer=None) -> None:
        """Run a warmup inference to trigger torch.compile JIT compilation.
        
        This should be called after model is moved to device and before serving,
        to avoid the first query taking a long time for compilation.
        
        Args:
            device: Device to run warmup on
            tokenizer: Optional tokenizer for creating dummy prompt tokens.
                       If None, will use simple dummy tokens.
        """
        if not self._compile_enabled:
            logger.info("Skipping warmup (torch.compile disabled)")
            return
        
        logger.info("Running warmup inference to trigger torch.compile...")
        
        # Create dummy observation matching expected shapes
        batch_size = 1
        state_dim = getattr(self.config, 'state_dim', 32)
        max_token_len = getattr(self.config, 'max_token_len', 200)
        
        # Dummy state
        state = torch.zeros(batch_size, state_dim, dtype=torch.bfloat16, device=device)
        
        # Dummy image (224x224 RGB, normalized to [-1, 1])
        dummy_image = torch.zeros(batch_size, 3, 224, 224, dtype=torch.bfloat16, device=device)
        
        # Dummy tokenized prompt
        if tokenizer is not None:
            tokenized = tokenizer("warmup", return_tensors="pt", padding="max_length", 
                                  truncation=True, max_length=max_token_len)
            tokenized_prompt = tokenized["input_ids"].to(device)
            tokenized_prompt_mask = tokenized["attention_mask"].bool().to(device)
        else:
            tokenized_prompt = torch.ones(batch_size, max_token_len, dtype=torch.long, device=device)
            tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool, device=device)
        
        dummy_observation = {
            "images": {"cam": dummy_image},
            "image_masks": {"cam": torch.tensor([True], dtype=torch.bool, device=device)},
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
        
        # Run sample_actions to trigger compilation
        _ = self.sample_actions(device=device, observation=dummy_observation)
        
        logger.info("Warmup complete - torch.compile JIT compilation finished")
