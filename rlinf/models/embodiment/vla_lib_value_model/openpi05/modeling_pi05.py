"""
PI0.5 Model with Extended Training Modes.

Extends PI0 to support three forwarding modes (auto-detected from inputs):
- VLA: Flow matching action prediction (actions provided, no token_loss_mask)
- VLM: Autoregressive cross-entropy loss (actions=None, token_loss_mask provided)
- VLM+VLA: Reasoning CE + action flow matching (both actions and token_loss_mask)

Key differences from PI0:
- No state projection (state discretized into language tokens)
- AdaRMSNorm for time conditioning in action expert
- token_ar_mask: attention pattern (0=bidirectional, 1=causal)
- token_loss_mask: which tokens contribute to CE loss
- token_kv_cache_mask: which tokens action expert attends to (excludes EOS)

KV Cache Masking for Action Expert
===================================

The token_kv_cache_mask controls which tokens are included in the KV cache
that the action expert attends to. This is set during tokenization and provides
a general mechanism to exclude certain tokens (like EOS) from action attention.

Training:
- EOS is appended to response tokens (both VLM and VLM+VLA modes)
- EOS has loss_mask=True, so model learns to generate it
- EOS has kv_cache_mask=False, so action expert doesn't attend to it
- Action expert attention mask is built using token_kv_cache_mask

Inference:
- generate_language() stops when EOS is produced
- EOS is stored in output_tokens but NOT added to KV cache
- sample_with_reasoning() uses EOS-excluded KV cache for action generation
- This ensures action expert never depends on EOS during either training or inference

Attention Pattern (token_ar_mask via cumsum)
--------------------------------------------
The cumsum of token_ar_mask determines attention blocks:

    tokens:  [IMG, IMG, "pick", "up", "ans", "red", "cup"]
    ar_mask: [  0,   0,     0,    0,     1,     1,     1]
    cumsum:  [  0,   0,     0,    0,     1,     2,     3]

Attention rule: token i can attend to token j if cumsum[j] <= cumsum[i]
- All prompt tokens (cumsum=0) see each other bidirectionally
- "ans" (cumsum=1) sees all prompt but starts causal chain
- "red" (cumsum=2) sees prompt + "ans" but not future answer tokens
- This creates prefix-LM: bidirectional on prompt, causal on response

FAST Dual-Objective Training (stop_gradient_to_vlm=True):
When enabled, implements knowledge insulation from Pi0 paper (Section 5.2). Attention
is computed separately for backbone and action queries:
- P_bb @ V_b: backbone self-attention with FULL gradients (CE loss trains all VLM params)
- P_ab @ sg(V_b) + P_aa @ V_a: action attention with stop gradient on cross-attention
This ensures CE loss has FULL gradients to VLM while flow matching only trains action expert.
"""

import math
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from rlinf.utils.dist_utils import get_logger
from ..openpi.configs import get_config
from ..openpi.modeling_pi0 import (
    PI0PreTrainedModel,
    PI0FlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from ..openpi.paligemma_with_expert import PaliGemmaWithExpertModel
from .configuration_pi05 import PI05Config
from .static_kv_cache import StaticKVCache, left_to_right_align

logger = get_logger(__name__)


@dataclass
class PI05Output(ModelOutput):
    """Output for PI0.5 model with support for multiple loss types."""
    loss: Optional[torch.FloatTensor] = None
    losses: Optional[torch.FloatTensor] = None
    action_loss: Optional[torch.FloatTensor] = None
    language_loss: Optional[torch.FloatTensor] = None
    actions: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    generated_tokens: Optional[torch.LongTensor] = None
    language_token_acc: Optional[torch.FloatTensor] = None
    language_loss_mask: Optional[torch.BoolTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PI05PreTrainedModel(PI0PreTrainedModel):
    config_class = PI05Config


class PI05ForConditionalGeneration(PI05PreTrainedModel, GenerationMixin):
    """PI0.5 model for Vision-Language-Action tasks with autoregressive language support."""

    def __init__(self, config: PI05Config):
        super().__init__(config)
        self.config = config
        self.model = PI05FlowMatching(config)
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings for resizing."""
        return self.model.paligemma_with_expert.paligemma.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings after resizing."""
        self.model.paligemma_with_expert.paligemma.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings (lm_head) for resizing."""
        return self.model.paligemma_with_expert.paligemma.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings after resizing."""
        self.model.paligemma_with_expert.paligemma.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        """
        Resize token embeddings, handling both input and output embeddings.
        
        Uses transformers library patterns for proper initialization of new embeddings.
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
        
        # Get paligemma language model
        lm = self.model.paligemma_with_expert.paligemma.language_model
        # Resize input embeddings (this handles mean_resizing properly)
        lm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        
        # Update lm_head to match with proper initialization
        old_lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        old_num_tokens = old_lm_head.weight.shape[0]
        
        if old_num_tokens != new_num_tokens:
            new_lm_head = nn.Linear(
                old_lm_head.in_features,
                new_num_tokens,
                bias=old_lm_head.bias is not None,
                device=old_lm_head.weight.device,
                dtype=old_lm_head.weight.dtype,
            )
            
            # Copy old weights
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            new_lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[:num_tokens_to_copy]
            if old_lm_head.bias is not None:
                new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
            
            # Initialize new weights
            if new_num_tokens > old_num_tokens:
                added_num_tokens = new_num_tokens - old_num_tokens
                if mean_resizing:
                    self._init_new_lm_head_with_mean(
                        old_lm_head.weight.data, new_lm_head.weight.data,
                        old_num_tokens, added_num_tokens
                    )
                    if old_lm_head.bias is not None:
                        bias_mean = old_lm_head.bias.data.mean()
                        new_lm_head.bias.data[old_num_tokens:] = bias_mean
                else:
                    nn.init.normal_(new_lm_head.weight.data[old_num_tokens:], mean=0.0, std=0.02)
                    if old_lm_head.bias is not None:
                        new_lm_head.bias.data[old_num_tokens:].zero_()
            
            self.model.paligemma_with_expert.paligemma.lm_head = new_lm_head
        
        # Update config
        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def _init_new_lm_head_with_mean(
        self,
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
        old_num_tokens: int,
        added_num_tokens: int,
    ) -> None:
        """
        Initialize new lm_head weights using mean of old weights.
        
        Attempts to use multivariate normal with old weights' covariance.
        Falls back to just the mean if covariance is not positive definite.
        """
        old_weights_f32 = old_weights[:old_num_tokens].to(torch.float32)
        mean_weight = old_weights_f32.mean(dim=0)
        
        # Try to compute covariance and sample from multivariate normal
        try:
            centered = old_weights_f32 - mean_weight
            covariance = (centered.T @ centered) / old_num_tokens
            epsilon = 1e-9
            dist = torch.distributions.MultivariateNormal(
                mean_weight, covariance_matrix=epsilon * covariance
            )
            new_weights[old_num_tokens:] = dist.sample((added_num_tokens,)).to(new_weights.dtype)
        except (ValueError, RuntimeError):
            # Covariance not positive definite, use mean only
            new_weights[old_num_tokens:] = mean_weight.to(new_weights.dtype)

    def forward(
        self,
        observation=None,
        actions=None,
        noise=None,
        time=None,
        x_t=None,
        return_intermediates=False,
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        **kwargs,
    ):
        """Unified forward pass - mode auto-detected from inputs.
        
        Args:
            noise: Optional noise tensor. If None, sampled internally.
            time: Optional timesteps tensor. If None, sampled internally.
            x_t: Optional noised actions. If None, computed from actions/noise/time.
            return_intermediates: If True, returns (v_t, noise, time, x_t) tuple.
                Used for NFT training where old/ref models need same inputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if observation is None:
            raise ValueError("PI0.5 model requires 'observation' input")
        
        result = self.model.forward(
            observation, actions, noise, time, x_t=x_t, return_intermediates=return_intermediates
        )
        
        if return_intermediates:
            return result  # (v_t, noise, time, x_t) tuple
        return result if return_dict else (result.loss,)

    def generate(
        self,
        observation,
        noise=None,
        num_steps=10,
        forward_mode: Optional[Literal["vla", "vlm", "vlm_vla"]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Generate actions and/or language based on mode."""
        mode = forward_mode or self.config.forward_mode
        device = next(self.parameters()).device
        eos_token_id = self.model._get_eos_token_id()

        if mode == "vla":
            return self.model.sample_actions(device, observation, noise, num_steps)
        elif mode == "vlm":
            return self.model.generate_language(
                observation,
                max_new_tokens or self.config.max_language_len,
                temperature if temperature is not None else self.config.language_temperature,
                eos_token_id,
            )
        elif mode == "vlm_vla":
            return self.model.sample_with_reasoning(
                device, observation, noise, num_steps,
                max_new_tokens or self.config.max_language_len,
                temperature if temperature is not None else self.config.language_temperature,
                eos_token_id,
            )
        else:
            raise ValueError(f"Unknown forward_mode: {mode}")


class PI05FlowMatching(PI0FlowMatching):
    """
    PI0.5 flow matching module extending PI0FlowMatching.

    Mode detection (in forward):
    - actions=None -> VLM mode (CE loss on token_loss_mask)
    - actions provided, token_loss_mask.any()=False -> VLA mode (flow matching only)
    - actions provided, token_loss_mask.any()=True -> CoT+VLA mode (CE + flow matching)
    """

    def __init__(self, config: PI05Config):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True

        paligemma_config = get_config(config.paligemma_variant)
        action_expert_config = get_config(config.action_expert_variant)

        logger.info(f"Creating PaliGemmaWithExpertModel with precision={config.dtype}")
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', False),
            train_expert_only=getattr(config, 'train_expert_only', False),
        )

        # Action projection layers
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # PI0.5: time MLP for adaRMSNorm (no state_proj)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self._compile_enabled = os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1"
        if self._compile_enabled:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        self.gradient_checkpointing_enabled = False

    # =========================================================================
    # Observation Processing
    # =========================================================================
    def _preprocess_observation(self, observation):
        """Extract observation components with unified masks."""
        images = observation['images']
        image_masks = observation.get('image_masks', {})
        tokenized_prompt = observation['tokenized_prompt']
        tokenized_prompt_mask = observation['tokenized_prompt_mask']

        batch_size, seq_len = tokenized_prompt.shape
        device = tokenized_prompt.device

        # token_ar_mask: 0=bidirectional, 1=causal (matches reference pi0_fast)
        token_ar_mask = observation.get('token_ar_mask')
        if token_ar_mask is None:
            token_ar_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        # token_loss_mask: True=include in CE loss
        token_loss_mask = observation.get('token_loss_mask')
        if token_loss_mask is None:
            token_loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        # token_kv_cache_mask: True=include in KV cache for action expert
        token_kv_cache_mask = observation.get('token_kv_cache_mask')
        if token_kv_cache_mask is None:
            # Default: all valid tokens are included in KV cache
            token_kv_cache_mask = tokenized_prompt_mask.clone()

        # CRITICAL: Use sorted keys for consistent ordering between train and eval.
        # dict.values() order depends on insertion order, which may differ between
        # training datasets and inference environments.
        sorted_keys = sorted(images.keys())
        img_list = [images[k] for k in sorted_keys]
        img_mask_list = [image_masks.get(k, torch.ones(batch_size, dtype=torch.bool, device=device)) for k in sorted_keys]

        return (
            img_list, img_mask_list,
            tokenized_prompt, tokenized_prompt_mask,
            token_ar_mask, token_loss_mask, token_kv_cache_mask,
        )

    # =========================================================================
    # Embedding Methods (aligned with reference pi0.py / pi0_fast.py)
    # =========================================================================
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, token_ar_mask=None):
        """
        Embed images and language tokens.

        Similar to pi0.embed_prefix + pi0_fast.embed_inputs:
        - Images get ar_mask=0 (bidirectional attention)
        - Language tokens use token_ar_mask from observation
        """
        embs, pad_masks, ar_masks = [], [], []
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # Embed images (ar_mask=0: image tokens attend to each other)
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            ar_masks.append(torch.zeros(bsize, num_img_embs, dtype=torch.long, device=device))

        # Embed language tokens (scaled by sqrt(dim))
        def embed_lang(tokens):
            emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return emb * math.sqrt(emb.shape[-1])
        lang_emb = self._apply_checkpoint(embed_lang, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Use token_ar_mask from observation (default: 0=bidirectional for prefix-lm style)
        if token_ar_mask is None:
            token_ar_mask = torch.zeros(bsize, lang_masks.shape[1], dtype=torch.long, device=device)
        ar_masks.append(token_ar_mask)

        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1), torch.cat(ar_masks, dim=1)

    def embed_suffix(self, noisy_actions, timestep):
        """
        Embed noisy actions and timestep for action expert.

        PI0.5 style (from reference pi0.py with pi05=True):
        - No state projection
        - Time MLP produces adaRMS conditioning
        - Action tokens: first has ar_mask=1, rest have ar_mask=0
        """
        # Time embedding with sinusoidal positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=timestep.device
        ).to(timestep.dtype)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        bsize, action_len = action_emb.shape[:2]
        pad_mask = torch.ones(bsize, action_len, dtype=torch.bool, device=timestep.device)
        # ar_mask: [1, 0, 0, ...] - first action token starts new causal block
        att_mask = torch.zeros(bsize, action_len, dtype=torch.long, device=timestep.device)
        att_mask[:, 0] = 1

        return action_emb, pad_mask, att_mask, adarms_cond

    def _get_eos_token_id(self) -> int:
        """
        Get EOS token ID from the underlying language model config.
        
        Priority:
        1. PaliGemma language model config (most authoritative)
        2. PI05Config.eos_token_id
        3. Default: 1 (PaliGemma default)
        """
        # Try to get from paligemma language model config
        try:
            paligemma_eos = self.paligemma_with_expert.paligemma.config.text_config.eos_token_id
            if paligemma_eos is not None:
                return paligemma_eos
        except AttributeError:
            pass
        
        # Fall back to PI05Config
        return getattr(self.config, 'eos_token_id', 1)

    def _mask_suffix_with_kv_cache_mask(
        self, attn_mask, token_kv_cache_mask, prefix_pad_masks, suffix_pad_masks
    ):
        """
        Modify attention mask to prevent action tokens from attending to excluded tokens.
        
        Uses token_kv_cache_mask (set during tokenization) to determine which tokens
        the action expert should NOT attend to (e.g., EOS tokens).
        
        This ensures train/inference consistency:
        - During training: action expert doesn't learn to depend on excluded tokens
        - During inference: excluded tokens are not in KV cache
        
        Args:
            attn_mask: [B, total_len, total_len] attention mask
            token_kv_cache_mask: [B, lang_len] which language tokens to include in KV cache
            prefix_pad_masks: [B, prefix_len] prefix padding mask
            suffix_pad_masks: [B, suffix_len] suffix padding mask
            
        Returns:
            Modified attention mask with suffix-to-excluded attention blocked
        """
        batch_size = attn_mask.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        total_len = attn_mask.shape[1]
        device = attn_mask.device
        
        # Calculate image token count
        lang_len = token_kv_cache_mask.shape[1]
        num_img_tokens = prefix_len - lang_len
        
        # Identify excluded positions (where kv_cache_mask is False)
        excluded_in_lang = ~token_kv_cache_mask  # [B, lang_len]
        excluded_positions = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        excluded_positions[:, num_img_tokens:prefix_len] = excluded_in_lang
        
        # Identify suffix positions
        suffix_positions = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        suffix_positions[:, prefix_len:] = True
        
        # Create mask: suffix positions attending to excluded positions
        suffix_to_excluded = suffix_positions[:, :, None] & excluded_positions[:, None, :]
        
        # One-time verification logging for kv_cache_mask
        if not getattr(self, '_verified_kv_cache_mask', False):
            self._verified_kv_cache_mask = True
            n_excluded = excluded_in_lang.sum(dim=1).float()
            logger.info("[Mask Verification] kv_cache_mask: excluding tokens from action expert attention:")
            for i in range(min(batch_size, 4)):
                logger.info("  Sample %d: %d tokens excluded (e.g. EOS)", i, int(n_excluded[i].item()))
        
        # Block suffix-to-excluded attention
        return attn_mask & ~suffix_to_excluded

    def _compute_position_ids_with_kv_cache_alignment(
        self, prefix_pad_masks, suffix_pad_masks, token_kv_cache_mask
    ):
        """
        Compute position IDs with train-eval alignment for action expert.
        
        When exclude_cot_from_kv_cache=True, CoT tokens are present during training
        but absent during eval. To ensure consistent position IDs for action tokens:
        
        - Prefix tokens: Natural cumsum positions (needed for CE loss on CoT)
        - Suffix tokens: Positions as if kv_cache_mask=False tokens don't exist
        
        This ensures action tokens see the same positions during training and eval,
        fixing the rotary embedding mismatch.
        
        Example with exclude_cot_from_kv_cache=True:
        - Prefix: [img0, img1, prompt0, prompt1, cot0, cot1] with kv_mask=[T,T,T,T,F,F]
        - Suffix: [action0, action1]
        
        Training (with this fix):
        - Prefix positions: [0, 1, 2, 3, 4, 5] (natural)
        - Suffix positions: [4, 5] (based on kv_mask=True count, i.e., 4)
        
        Eval (no CoT tokens):
        - Prefix: [img0, img1, prompt0, prompt1]
        - Suffix: [action0, action1]
        - Prefix positions: [0, 1, 2, 3]
        - Suffix positions: [4, 5] (matches training!)
        
        Args:
            prefix_pad_masks: [B, prefix_len] padding mask for prefix
            suffix_pad_masks: [B, suffix_len] padding mask for suffix  
            token_kv_cache_mask: [B, lang_len] which language tokens are in KV cache
            
        Returns:
            position_ids: [B, prefix_len + suffix_len] position IDs
        """
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        device = prefix_pad_masks.device
        
        # Prefix position IDs: natural cumsum (for CE loss on all prefix tokens)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Calculate number of image tokens in prefix
        lang_len = token_kv_cache_mask.shape[1]
        num_img_tokens = prefix_len - lang_len
        
        # Build full kv_cache_mask for prefix: images are always in cache
        img_kv_mask = torch.ones(batch_size, num_img_tokens, dtype=torch.bool, device=device)
        full_prefix_kv_mask = torch.cat([img_kv_mask, token_kv_cache_mask], dim=1)
        
        # Count tokens that action expert can attend to (kv_cache_mask=True)
        # This is the starting position for suffix tokens
        suffix_start_pos = full_prefix_kv_mask.sum(dim=1)  # [B]
        
        # Suffix position IDs: start from suffix_start_pos
        suffix_offsets = torch.cumsum(suffix_pad_masks, dim=1) - 1  # [B, suffix_len], 0-indexed
        suffix_position_ids = suffix_start_pos[:, None] + suffix_offsets  # [B, suffix_len]
        
        # Concatenate prefix and suffix position IDs
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)
        
        # One-time verification logging
        if not getattr(self, '_verified_position_alignment', False):
            self._verified_position_alignment = True
            n_excluded = (~token_kv_cache_mask).sum(dim=1).float()
            natural_suffix_start = prefix_len
            aligned_suffix_start = suffix_start_pos.float()
            logger.info("[Position Alignment] Train-eval position ID alignment enabled:")
            for i in range(min(batch_size, 2)):
                logger.info(
                    "  Sample %d: %d CoT tokens excluded, suffix starts at pos %d (vs natural %d)",
                    i, int(n_excluded[i].item()), 
                    int(aligned_suffix_start[i].item()), natural_suffix_start
                )
        
        return position_ids

    # =========================================================================
    # Unified Forward: Auto-detect mode from inputs
    # =========================================================================
    def forward(self, observation, actions=None, noise=None, time=None, x_t=None, return_intermediates=False):
        """
        Unified forward with auto mode detection:
        - actions=None -> VLM mode (CE loss)
        - actions + no token_loss_mask -> VLA mode (flow matching)
        - actions + token_loss_mask -> CoT+VLA mode (CE + flow matching)
        - return_intermediates=True -> Return (v_t, noise, time, x_t) for NFT training
        """
        (images, img_masks, lang_tokens, lang_masks,
         token_ar_mask, token_loss_mask, token_kv_cache_mask) = self._preprocess_observation(observation)

        # Sample-level action supervision mask (1.0 if sample has actions and
        # should contribute to flow-matching loss, 0.0 otherwise). This mirrors
        # language supervision which is controlled via token_loss_mask.
        action_mask = observation.get('action_mask', None)

        has_ce_loss = token_loss_mask.any()
        has_actions = (actions is not None) and (self.config.action_loss_weight != 0)

        # VLM-only mode (no actions)
        if not has_actions:
            return self._forward_vlm(images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask)

        # VLA or CoT+VLA mode (has actions)
        return self._forward_vla(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            token_loss_mask,
            token_kv_cache_mask,
            actions,
            noise,
            time,
            has_ce_loss,
            x_t=x_t,
            return_intermediates=return_intermediates,
            action_mask=action_mask,
        )

    def _compute_velocity_core(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        token_ar_mask,
        token_kv_cache_mask,
        x_t,
        time,
        stop_gradient_to_vlm: bool = False,
        debug: bool = False,
        debug_sample_idx: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Core velocity computation from pre-extracted observation components.
        
        Args:
            images, img_masks: Image tensors and masks
            lang_tokens, lang_masks: Language tokens and masks
            token_ar_mask: Attention pattern mask
            token_kv_cache_mask: KV cache inclusion mask
            x_t: Noised actions [B, action_horizon, action_dim]
            time: Timestep [B]
            stop_gradient_to_vlm: If True, detach prefix key/value in cross-attention
                to prevent flow matching gradients from affecting VLM backbone.
                This is safe for FSDP/DeepSpeed as detach() is a local operation.
            
        Returns:
            v_t: Predicted velocity [B, action_horizon, action_dim]
            prefix_out: Prefix outputs (for CE loss computation)
            prefix_pad_masks: Prefix padding masks (for CE loss computation)
        """
        # Embed prefix (images + language) and suffix (noisy actions).
        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks, adarms_cond = self.embed_suffix(x_t, time)

        # Cast to model dtype if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs, suffix_embs = prefix_embs.to(torch.bfloat16), suffix_embs.to(torch.bfloat16)

        # Build attention mask and position_ids
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask = make_att_2d_masks(pad_masks, ar_masks)

        attn_mask = self._mask_suffix_with_kv_cache_mask(
            attn_mask, token_kv_cache_mask, prefix_pad_masks, suffix_pad_masks
        )

        att_2d_masks_4d = self._prepare_attention_masks_4d(attn_mask)
        
        # Compute position_ids with train-eval alignment for action expert.
        # Problem: When exclude_cot_from_kv_cache=True, CoT tokens are in the sequence
        # during training but absent during eval. If we naively use cumsum(pad_masks),
        # action tokens get positions offset by CoT length during training, but at eval
        # they start right after prompt. This causes position ID mismatch.
        #
        # Solution: For suffix (action) tokens, compute positions as if tokens with
        # kv_cache_mask=False don't exist. This matches eval where those tokens are absent.
        # Prefix tokens keep their natural positions (needed for CE loss on CoT).
        position_ids = self._compute_position_ids_with_kv_cache_alignment(
            prefix_pad_masks, suffix_pad_masks, token_kv_cache_mask
        )

        # Forward through model
        # stop_gradient_to_vlm: When True, detaches prefix key/value in cross-attention
        # so flow matching gradients don't propagate to VLM backbone parameters.
        detach_prefix_kv = stop_gradient_to_vlm

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, detach_kv):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                detach_prefix_for_suffix=detach_kv,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
            detach_prefix_kv,
        )

        # Optional debug logging for flow-matching internals.
        if debug:
            # Clamp debug index into valid range.
            b = max(0, min(debug_sample_idx, lang_tokens.shape[0] - 1))

            try:
                def _summarize_1d(name: str, tensor: Tensor, max_items: int = 32) -> None:
                    arr = tensor.detach().cpu().flatten().tolist()
                    head = arr[:max_items]
                    logger.info(
                        "[FlowDebug] %s: shape=%s, head=%s%s",
                        name,
                        tuple(tensor.shape),
                        head,
                        " ... (truncated)" if len(arr) > max_items else "",
                    )

                logger.info("[FlowDebug] ===== Sample %d =====", b)
                logger.info(
                    "[FlowDebug] prefix_embs shape=%s, suffix_embs shape=%s",
                    tuple(prefix_embs.shape),
                    tuple(suffix_embs.shape),
                )
                logger.info(
                    "[FlowDebug] prefix_pad_masks shape=%s, suffix_pad_masks shape=%s",
                    tuple(prefix_pad_masks.shape),
                    tuple(suffix_pad_masks.shape),
                )

                _summarize_1d("lang_tokens[%d]" % b, lang_tokens[b])
                _summarize_1d("lang_masks[%d]" % b, lang_masks[b])
                _summarize_1d("token_ar_mask[%d]" % b, token_ar_mask[b])
                _summarize_1d("token_kv_cache_mask[%d]" % b, token_kv_cache_mask[b])
                _summarize_1d("prefix_pad_masks[%d]" % b, prefix_pad_masks[b])
                _summarize_1d("suffix_pad_masks[%d]" % b, suffix_pad_masks[b])

                # Attention mask is 2D for the sample; summarize flattened head.
                _summarize_1d("attn_mask[%d]" % b, attn_mask[b])
                _summarize_1d("adarms_cond[%d]" % b, adarms_cond[b])
                _summarize_1d("x_t[%d]" % b, x_t[b])
                _summarize_1d("prefix_out[%d]" % b, prefix_out[b])
                _summarize_1d("suffix_out[%d]" % b, suffix_out[b])
            except Exception as e:  # Defensive: avoid crashing if debug logging fails
                logger.error("[FlowDebug] Error while logging debug info: %s", e)

        # Project to velocity
        action_out = suffix_out[:, -self.config.action_horizon :]
        action_out = action_out.to(self.action_out_proj.weight.dtype)
        v_t = self._apply_checkpoint(self.action_out_proj, action_out)

        return v_t, prefix_out, prefix_pad_masks

    def _forward_vla(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        token_ar_mask,
        token_loss_mask,
        token_kv_cache_mask,
        actions,
        noise,
        time,
        compute_ce_loss,
        x_t=None,
        return_intermediates: bool = False,
        action_mask=None,
        debug: bool = False,
        debug_sample_idx: int = 0,
    ):
        """VLA forward with optional CE loss (for CoT+VLA mode).
        
        Gradient flow when stop_gradient_to_vlm=True (knowledge insulation):
        - CE loss: FULL gradients to all VLM parameters (Q, K, V, O, MLP, layernorms)
        - Flow loss: Updates action expert only, no gradients to VLM backbone
        
        This is achieved by splitting attention computation (Pi0 paper Eq. 5-6):
        backbone self-attention has full gradients, cross-attention uses stop gradient.
        
        Args:
            x_t: Pre-computed noised actions. If provided, noise/time are ignored for x_t computation.
            return_intermediates: If True, return (v_t, noise, time, x_t) tuple.
        """
        # Prepare flow matching (aligned with reference pi0.py)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        noise, time = noise.to(actions.dtype), time.to(actions.dtype)

        time_expanded = time[:, None, None]
        if x_t is None:
            x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Use stop_gradient_to_vlm from config to prevent flow matching gradients
        # from backpropagating to VLM backbone (for FAST dual-objective training)
        stop_gradient = getattr(self.config, 'stop_gradient_to_vlm', False)
        
        # One-time verification logging (first forward call only)
        if not getattr(self, '_verified_stop_gradient', False):
            self._verified_stop_gradient = True
            logger.info("[KI Verification] stop_gradient_to_vlm=%s (config=%s)", 
                       stop_gradient, getattr(self.config, 'stop_gradient_to_vlm', 'NOT_SET'))
        
        v_t, prefix_out, prefix_pad_masks = self._compute_velocity_core(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            token_kv_cache_mask,
            x_t,
            time,
            stop_gradient_to_vlm=stop_gradient,
            debug=debug,
            debug_sample_idx=debug_sample_idx,
        )

        if return_intermediates:
            return v_t, noise, time, x_t

        flow_loss = F.mse_loss(u_t, v_t, reduction='none').mean(dim=(-1, -2))

        # Sample-level masking for action (flow-matching) loss:
        # - action_mask=1.0: sample has actions, contributes to loss
        # - action_mask=0.0: no actions; flow_loss for this sample is zeroed
        n_action_samples = None
        if action_mask is not None:
            mask = action_mask.to(flow_loss.device).view(-1)
            n_action_samples = mask.sum()  # Count of VLA samples for proper averaging
            
            # One-time verification logging for action_mask
            if not getattr(self, '_verified_action_mask', False):
                self._verified_action_mask = True
                n_with_actions = mask.sum().item()
                n_without_actions = (1 - mask).sum().item()
                logger.info("[Mask Verification] action_mask: %d with actions, %d without (VLM-only)",
                           int(n_with_actions), int(n_without_actions))
            
            flow_loss = flow_loss * mask

        # CE loss (if token_loss_mask has True values)
        ce_loss, language_acc, logits = None, None, None
        if compute_ce_loss:
            # One-time verification logging for loss_mask
            if not getattr(self, '_verified_loss_mask', False):
                self._verified_loss_mask = True
                B = token_loss_mask.shape[0]
                tokens_with_loss = token_loss_mask.sum(dim=1).float()
                logger.info("[Mask Verification] token_loss_mask stats (first batch):")
                for i in range(min(B, 4)):  # Show first 4 samples
                    n_loss = tokens_with_loss[i].item()
                    logger.info("  Sample %d: %d tokens with loss_mask=True", i, int(n_loss))
            
            ce_loss, language_acc, logits = self._compute_ce_loss(
                prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask
            )

        # =====================================================================
        # Compute mean loss per sample type independently (ratio-independent).
        # Each type is averaged over its own samples, then combined with weights.
        # This ensures VLA/VLM contributions are controlled purely by loss weights.
        # =====================================================================

        # Determine sample counts for each type
        # action_mask: 1.0 for VLA samples, 0.0 for VLM samples
        # has_lang_target: True for samples with language targets (VLM samples)
        has_lang_target = token_loss_mask.any(dim=-1) if ce_loss is not None else None
        n_action = n_action_samples if n_action_samples is not None else action_mask.sum() if action_mask is not None else flow_loss.shape[0]
        n_lang = has_lang_target.sum() if has_lang_target is not None else 0

        # Clamp to avoid division by zero
        n_action_safe = n_action.clamp(min=1) if isinstance(n_action, torch.Tensor) else max(n_action, 1)
        n_lang_safe = n_lang.clamp(min=1) if isinstance(n_lang, torch.Tensor) else max(n_lang, 1)

        # Action loss: sum over VLA samples / count of VLA samples
        # flow_loss is already masked (VLM samples = 0)
        action_loss_mean = flow_loss.sum() / n_action_safe

        # Language loss: sum over VLM samples / count of VLM samples
        # ce_loss has 0 for VLA samples (due to all-False loss_mask)
        language_loss_mean = None
        language_acc_mean = None
        if ce_loss is not None:
            language_loss_mean = ce_loss.sum() / n_lang_safe
            language_acc_mean = language_acc.sum() / n_lang_safe

        # Combine: weighted sum of per-type means
        total_loss_scalar = self.config.action_loss_weight * action_loss_mean
        if language_loss_mean is not None:
            total_loss_scalar = total_loss_scalar + self.config.language_loss_weight * language_loss_mean

        # Per-sample losses (for logging only, not used for backward)
        # Normalize so that sum(per_sample) = total_loss_scalar
        if ce_loss is not None:
            per_sample_loss = (
                self.config.action_loss_weight * flow_loss / n_action_safe +
                self.config.language_loss_weight * ce_loss / n_lang_safe
            )
        else:
            per_sample_loss = flow_loss

        return PI05Output(
            loss=total_loss_scalar,
            losses=per_sample_loss,
            action_loss=action_loss_mean,
            language_loss=language_loss_mean,
            logits=logits,
            language_token_acc=language_acc_mean,
            language_loss_mask=token_loss_mask[:, 1:] if compute_ce_loss else None,
        )

    def _forward_vlm(self, images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask):
        """VLM-only forward (no actions, CE loss only)."""
        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )

        # For VLM-only, we don't input the last token (predicts next token)
        attn_mask = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask[:, :-1, :-1])
        position_ids = torch.cumsum(prefix_pad_masks[:, :-1], dim=1) - 1

        # Apply gradient checkpointing to reduce memory (matching VLA mode behavior)
        def forward_func(prefix_embs, attn_mask_4d, position_ids):
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            return prefix_out

        prefix_out = self._apply_checkpoint(
            forward_func, prefix_embs[:, :-1], attn_mask_4d, position_ids
        )

        ce_loss, language_acc, logits = self._compute_ce_loss(
            prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask, truncated_input=True
        )

        return PI05Output(
            loss=ce_loss.mean(),
            losses=ce_loss,
            language_loss=ce_loss,
            logits=logits,
            language_token_acc=language_acc,
            language_loss_mask=token_loss_mask[:, 1:],
        )

    def _compute_ce_loss(self, prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask, truncated_input=False):
        """
        Compute CE loss for language token prediction.

        Token flow for next-token prediction:
        - Input sequence: [img_0, ..., img_{M-1}, tok_0, ..., tok_{N-1}] where M=num_img, N=num_lang
        - Output at position i predicts token at position i+1
        - To predict lang_tokens[1:N], we need outputs at position_ids [M, M+1, ..., M+N-2]

        For VLA mode (full input, truncated_input=False):
        - Input length: M + N
        - Output length: M + N
        - lang_out = prefix_out[:, M:-1] gives outputs [M, ..., M+N-2], length N-1

        For VLM mode (truncated input, truncated_input=True):
        - Input length: M + N - 1 (last token removed for next-token prediction)
        - Output length: M + N - 1
        - lang_out = prefix_out[:, M:] gives outputs [M, ..., M+N-2], length N-1

        Args:
            prefix_out: Model output [B, seq_len, D]
            prefix_pad_masks: Padding masks [B, M + N] (original, NOT truncated)
            lang_tokens: Language token ids [B, N]
            token_loss_mask: Which tokens to compute loss on [B, N]
            truncated_input: Whether model input was truncated (VLM mode removes last token)
        """
        vocab_size = self.paligemma_with_expert.paligemma.language_model.config.vocab_size
        num_img_tokens = prefix_pad_masks.shape[1] - lang_tokens.shape[1]

        # Extract language outputs that predict tokens [1, 2, ..., N-1]
        if truncated_input:
            # VLM mode: input was truncated, prefix_out has length M+N-1
            # Outputs [M, M+1, ..., M+N-2] predict lang_tokens[1:N]
            lang_out = prefix_out[:, num_img_tokens:]
        else:
            # VLA mode: full input, prefix_out has length M+N
            # Outputs [M, M+1, ..., M+N-2] predict lang_tokens[1:N], exclude last output
            lang_out = prefix_out[:, num_img_tokens:-1]

        # Targets and loss mask (shifted by 1 for next-token prediction)
        target_ids = lang_tokens[:, 1:]
        loss_mask = token_loss_mask[:, 1:]
        denom = torch.clamp(loss_mask.sum(dim=-1), min=1)

        # Compute full logits for proper metrics computation
        logits = self.paligemma_with_expert.paligemma.lm_head(lang_out)  # [B, seq-1, vocab]

        # CE loss with integer targets
        ce_per_position = F.cross_entropy(
            logits.view(-1, vocab_size), target_ids.reshape(-1), reduction='none'
        ).view(target_ids.shape)
        ce_loss = (ce_per_position * loss_mask).sum(dim=-1) / denom

        # Token accuracy
        pred_ids = logits.argmax(dim=-1)
        correct = (pred_ids == target_ids).float()
        language_acc = (correct * loss_mask).sum(dim=-1) / denom

        return ce_loss, language_acc, logits

    # =========================================================================
    # Inference Methods
    # =========================================================================
    def _euler_sample(self, prefix_pad_masks, past_key_values, noise, device, num_steps):
        """Run Euler integration for flow matching sampling."""
        bsize = noise.shape[0]
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            timestep = time.expand(bsize)
            v_t = self._denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Sample actions using flow matching (VLA mode)."""
        bsize = observation['tokenized_prompt'].shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _ = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )

        # Fill KV cache with prefix
        attn_mask = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        return self._euler_sample(prefix_pad_masks, past_key_values, noise, device, num_steps)

    def _denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Single denoising step."""
        suffix_embs, suffix_pad_masks, suffix_ar_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        # Build attention mask for suffix attending to prefix (via cache) + suffix
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_attn_2d = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_mask = torch.cat([prefix_pad_2d, suffix_attn_2d], dim=2)
        full_attn_4d = self._prepare_attention_masks_4d(full_attn_mask)

        # Position_ids continue from prefix
        position_ids = prefix_pad_masks.sum(dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def generate_language(self, observation, max_new_tokens=50, temperature=0.0, eos_token_id=None):
        """
        Generate language autoregressively (VLM mode).
        
        Generates tokens until EOS is produced or max_new_tokens is reached.
        IMPORTANT: EOS token is stored in output but NOT added to KV cache,
        ensuring action expert won't attend to it during sample_with_reasoning.
        
        Returns:
            output_tokens: [B, actual_len] Generated tokens (exact length, includes EOS if generated)
            past_kv: KV cache (excludes EOS embedding, for use in action sampling)
            full_pad_mask: [B, prefix_len + tokens_in_cache] Padding mask for sequence in KV cache
            full_ar_mask: [B, prefix_len + tokens_in_cache] AR mask for sequence in KV cache
        """
        if eos_token_id is None:
            eos_token_id = self._get_eos_token_id()
            
        images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _ = self._preprocess_observation(observation)
        batch_size, device = lang_tokens.shape[0], lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )

        # Left-to-right align for efficient decoding
        prefix_attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_embs, prefix_pad_masks, prefix_attn_2d = left_to_right_align(
            prefix_embs, prefix_pad_masks, prefix_attn_2d
        )

        prefill_size = prefix_embs.shape[1]
        prefill_len = prefix_pad_masks.sum(dim=-1)
        prefix_start = prefill_size - prefill_len

        # Pad attention mask for decoding steps
        prefix_attn_2d = F.pad(prefix_attn_2d, (0, max_new_tokens, 0, 0))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Initialize static KV cache
        paligemma_config = self.paligemma_with_expert.paligemma.config.text_config
        static_cache = StaticKVCache(
            max_batch_size=batch_size,
            max_cache_len=prefill_size + max_new_tokens,
            num_layers=paligemma_config.num_hidden_layers,
            num_key_value_heads=paligemma_config.num_key_value_heads,
            head_dim=paligemma_config.head_dim,
            dtype=prefix_embs.dtype,
            device=device,
        )

        # Prefill
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_attn_2d),
            position_ids=position_ids,
            past_key_values=static_cache,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        # Decode loop - build tokens dynamically
        last_logits = F.log_softmax(self.paligemma_with_expert.paligemma.lm_head(prefix_out[:, -1:]), dim=-1)
        generated_tokens = []  # List of [B, 1] tensors
        tokens_in_cache = 0  # Track how many generated tokens are in KV cache

        for step in range(max_new_tokens):
            if temperature > 0:
                token = torch.multinomial(torch.exp(last_logits[:, 0] / temperature), num_samples=1)
            else:
                token = last_logits[:, 0].argmax(dim=-1, keepdim=True)
            generated_tokens.append(token)

            # Stop before adding EOS to KV cache
            if torch.all(token == eos_token_id):
                break

            # Embed new token and update KV cache (only for non-EOS tokens)
            tokens_in_cache += 1
            token_emb = self.paligemma_with_expert.embed_language_tokens(token)
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            step_position_ids = (prefill_len + step)[:, None]

            total_len = prefill_size + max_new_tokens
            j_indices = torch.arange(total_len, device=device)[None, None, :]
            mask = (j_indices >= prefix_start[:, None, None]) & (j_indices < (prefill_size + step + 1))

            (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(mask),
                position_ids=step_position_ids,
                past_key_values=past_kv,
                inputs_embeds=[token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            last_logits = F.log_softmax(self.paligemma_with_expert.paligemma.lm_head(prefix_out[:, -1:]), dim=-1)

        # Concatenate generated tokens: [B, actual_len]
        assert len(generated_tokens) > 0, "No tokens generated"
        output_tokens = torch.cat(generated_tokens, dim=1)

        # Build masks for sequence in KV cache (prefix + non-EOS generated tokens)
        gen_pad_mask = torch.ones((batch_size, tokens_in_cache), dtype=torch.bool, device=device)
        gen_ar_mask = torch.ones((batch_size, tokens_in_cache), dtype=torch.long, device=device)
        full_pad_mask = torch.cat([prefix_pad_masks, gen_pad_mask], dim=1)
        full_ar_mask = torch.cat([prefix_ar_masks, gen_ar_mask], dim=1)

        return output_tokens, past_kv, full_pad_mask, full_ar_mask

    @torch.no_grad()
    def predict_next_token_logits(self, observation) -> Tensor:
        """Get logits for next token prediction given observation."""
        images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _ = self._preprocess_observation(observation)
        device = lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )

        # Left-to-right align for consistent position handling
        prefix_attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_embs, prefix_pad_masks, prefix_attn_2d = left_to_right_align(
            prefix_embs, prefix_pad_masks, prefix_attn_2d
        )

        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prefill only (no generation loop needed)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_attn_2d),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )

        # Get logits at the last position (predicts the value token)
        logits = self.paligemma_with_expert.paligemma.lm_head(prefix_out[:, -1])
        return logits

    @torch.no_grad()
    def sample_with_reasoning(
        self, device, observation, noise=None, num_steps=10,
        max_reasoning_tokens=50, temperature=0.0, eos_token_id=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate reasoning language, then sample actions (VLM+VLA mode).
        
        Flow:
        1. Generate reasoning tokens autoregressively until EOS
        2. EOS is stored in output tokens but NOT in KV cache
        3. Use the EOS-excluded KV cache for action generation
        
        This ensures action expert never attends to EOS, maintaining consistency
        with training where we explicitly mask suffix-to-EOS attention.
        
        Args:
            device: Device for tensor creation
            observation: Input observation dict
            noise: Optional noise for flow matching
            num_steps: Number of Euler integration steps
            max_reasoning_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            eos_token_id: EOS token id (defaults to config value)
            
        Returns:
            actions: Sampled actions [B, action_horizon, action_dim]
            generated_tokens: Generated language tokens [B, max_reasoning_tokens]
        """
        if eos_token_id is None:
            eos_token_id = self._get_eos_token_id()
            
        # Generate language - past_kv excludes EOS embedding
        generated_tokens, past_kv, prefix_pad_masks, _ = self.generate_language(
            observation, max_reasoning_tokens, temperature, eos_token_id
        )
        batch_size = generated_tokens.shape[0]

        if noise is None:
            noise = self.sample_noise((batch_size, self.config.action_horizon, self.config.action_dim), device)

        # Action expert uses KV cache that doesn't contain EOS
        actions = self._euler_sample(prefix_pad_masks, past_kv, noise, device, num_steps)
        return actions, generated_tokens
